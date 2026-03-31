"""
ConversationCoordinator — punto de entrada principal del agente HiperForge User.

El ConversationCoordinator es el orquestador central del sistema. Implementa
ConversationPort y coordina todos los servicios de la capa de aplicación para
procesar cada turno de conversación del usuario.

Responsabilidades:
  1. Recibir el mensaje del usuario desde la UI
  2. Restaurar o crear la sesión y el contexto
  3. Construir el LLMContext apropiado
  4. Llamar al LLM y obtener la respuesta
  5. Enrutar la respuesta al flujo correcto (DirectReply / Tool / Plan)
  6. Coordinar la ejecución de tools a través del TaskExecutor
  7. Generar la síntesis final si hubo tools
  8. Persistir el turno en la sesión
  9. Registrar en el audit log
  10. Retornar la AssistantResponse a la UI

El Coordinator NO contiene lógica de negocio. Es un orquestador puro —
delega todo a servicios especializados. Si crece más de ~300 líneas de
lógica real (sin docstrings ni comentarios), es señal de que se debe
extraer un servicio nuevo.

Diseño de manejo de errores:
  - Los errores de LLM se convierten en AssistantResponse con tipo ERROR.
  - Los errores de tool execution producen resultados de fallo, no excepciones.
  - Los errores internos del coordinator se loguean y producen ERROR response.
  - NUNCA se lanza una excepción al caller de la UI.

Diseño de concurrencia:
  - El coordinator es stateless entre llamadas — todo el estado vive en la sesión.
  - Múltiples sesiones pueden estar activas simultáneamente (asyncio safe).
  - El SessionManager usa locks por session_id para prevenir race conditions.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

from forge_core.errors.types import (
    ForgeDomainError,
    ForgeLLMError,
    ForgeStorageError,
)
from forge_core.observability.logging import get_logger

from src.domain.entities.artifact import ArtifactSummary
from src.domain.entities.session import TurnReference
from src.domain.entities.user_profile import UserProfile
from src.domain.value_objects.identifiers import (
    ArtifactId,
    ApprovalId,
    RequestId,
    SessionId,
    TurnId,
    UserId,
)
from src.domain.value_objects.intent import (
    DirectReply,
    MultipleToolCalls,
    PlanningRequired,
    RoutingDecision,
    SingleToolCall,
)
from src.domain.value_objects.token_budget import TokenEstimator
from src.ports.inbound.conversation_port import (
    AdminPort,
    ApprovalResponseRequest,
    AssistantResponse,
    AssistantResponseType,
    ArtifactInfo as UIArtifactInfo,
    CancelTaskRequest,
    ConversationPort,
    CreateSessionRequest,
    SessionInfo,
    SessionListResponse,
    StreamingChunk,
    SwitchSessionRequest,
    ThinkingIndicator,
    UserMessageRequest,
)
from src.ports.outbound.llm_port import (
    ConversationContext,
    SynthesisContext,
    SystemPromptBuilder,
    UserLLMPort,
)
from src.ports.outbound.policy_port import ApprovalPort, PolicyEnginePort
from src.ports.outbound.storage_port import (
    ArtifactStorePort,
    AuditLogPort,
    SessionStorePort,
    UserProfileStorePort,
)
from src.ports.outbound.tool_port import ToolDispatchPort, ToolRegistryPort


logger = get_logger(__name__, component="conversation_coordinator")


class ConversationCoordinator(ConversationPort):
    """
    Orquestador central del agente HiperForge User.

    Implementa ConversationPort coordinando todos los servicios de la capa
    de aplicación. Es el único lugar donde se ensamblan los resultados de
    múltiples servicios para producir la respuesta final al usuario.

    Inyección de dependencias:
    Todas las dependencias se inyectan en el constructor. El ConversationCoordinator
    no crea ningún servicio internamente — el bootstrap (DI container) los provee.
    Esto garantiza testabilidad completa: cada dependencia puede ser mockeada.

    El coordinator es stateless entre requests — no mantiene ningún estado mutable
    a nivel de instancia. Todo el estado de la conversación vive en la Session
    (persistida por SessionStorePort) y en el UserProfile.
    """

    def __init__(
        self,
        llm_port: UserLLMPort,
        session_store: SessionStorePort,
        artifact_store: ArtifactStorePort,
        audit_log: AuditLogPort,
        profile_store: UserProfileStorePort,
        tool_registry: ToolRegistryPort,
        tool_dispatch: ToolDispatchPort,
        policy_engine: PolicyEnginePort,
        approval_port: ApprovalPort,
        *,
        planning_threshold: int = 2,
        max_tool_calls_per_turn: int = 20,
        thinking_indicators_enabled: bool = True,
    ) -> None:
        """
        Inicializa el ConversationCoordinator con sus dependencias.

        Args:
            llm_port:                   Puerto LLM del producto.
            session_store:              Puerto de persistencia de sesiones.
            artifact_store:             Puerto de persistencia de artifacts.
            audit_log:                  Puerto del audit log.
            profile_store:              Puerto de persistencia de perfiles.
            tool_registry:              Registro de tools disponibles.
            tool_dispatch:              Dispatcher de ejecución de tools.
            policy_engine:              Motor de políticas de seguridad.
            approval_port:              Puerto de gestión de aprobaciones.
            planning_threshold:         Tool calls para activar planificación.
            max_tool_calls_per_turn:    Límite de tool calls por turno.
            thinking_indicators_enabled: Habilitar indicadores de progreso.
        """
        self._llm = llm_port
        self._session_store = session_store
        self._artifact_store = artifact_store
        self._audit_log = audit_log
        self._profile_store = profile_store
        self._tool_registry = tool_registry
        self._tool_dispatch = tool_dispatch
        self._policy_engine = policy_engine
        self._approval_port = approval_port
        self._planning_threshold = planning_threshold
        self._max_tool_calls_per_turn = max_tool_calls_per_turn
        self._thinking_enabled = thinking_indicators_enabled
        self._system_prompt_builder = SystemPromptBuilder()

    # =========================================================================
    # ConversationPort — operación principal
    # =========================================================================

    async def send_message(
        self,
        request: UserMessageRequest,
    ) -> AssistantResponse:
        """
        Procesa un mensaje del usuario y retorna la respuesta completa.

        Flujo completo:
          1. Validar el request
          2. Cargar sesión y perfil de usuario
          3. Construir el contexto de conversación
          4. Llamar al LLM
          5. Enrutar la respuesta (DirectReply / Tool / Plan)
          6. Ejecutar tools si aplica
          7. Sintetizar la respuesta final
          8. Persistir el turno
          9. Registrar en audit log
          10. Retornar AssistantResponse
        """
        start_time = time.perf_counter()
        request_id = RequestId.generate()

        logger.info(
            "turno_iniciado",
            session_id=request.session_id.to_str(),
            request_id=request_id.to_str(),
            has_message=bool(request.message.strip()),
            attachments_count=len(request.attachments),
        )

        try:
            return await self._process_turn(request, request_id, start_time)

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "turno_fallido_inesperadamente",
                session_id=request.session_id.to_str(),
                error=exc,
                duration_ms=round(duration_ms, 2),
            )
            return self._build_error_response(
                session_id=request.session_id,
                message="Ha ocurrido un error inesperado. Por favor, intenta de nuevo.",
                is_retryable=True,
                duration_ms=duration_ms,
            )

    async def send_message_streaming(
        self,
        request: UserMessageRequest,
    ) -> AsyncIterator[StreamingChunk]:
        """
        Versión streaming de send_message.

        Emite chunks progresivos durante el procesamiento.
        El último chunk tiene is_final=True con la respuesta completa.
        """
        start_time = time.perf_counter()
        request_id = RequestId.generate()

        try:
            # Emitir indicador de procesamiento inicial
            if self._thinking_enabled:
                yield StreamingChunk(
                    chunk_type="thinking",
                    thinking_indicator=ThinkingIndicator(
                        message="Procesando tu mensaje...",
                    ),
                )

            # Procesar el turno completo
            response = await self._process_turn(request, request_id, start_time)

            # Emitir el contenido de texto en chunks si hay texto
            if response.text_content:
                # Simular streaming del texto por oraciones para UX fluida
                sentences = self._split_into_streaming_chunks(response.text_content)
                for sentence in sentences:
                    yield StreamingChunk(
                        chunk_type="text_delta",
                        text_delta=sentence,
                    )

            # Emitir artifacts producidos
            for artifact_info in response.artifacts_produced:
                yield StreamingChunk(
                    chunk_type="artifact_ready",
                    artifact=artifact_info,
                )

            # Emitir solicitud de aprobación si la hay
            if response.approval_request is not None:
                yield StreamingChunk(
                    chunk_type="approval_needed",
                    approval_request=response.approval_request,
                )

            # Chunk final con la respuesta completa
            yield StreamingChunk(
                chunk_type="final",
                is_final=True,
                final_response=response,
            )

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "streaming_fallido",
                session_id=request.session_id.to_str(),
                error=exc,
                duration_ms=round(duration_ms, 2),
            )
            error_response = self._build_error_response(
                session_id=request.session_id,
                message="Error durante el streaming de la respuesta.",
                is_retryable=True,
                duration_ms=duration_ms,
            )
            yield StreamingChunk(
                chunk_type="error",
                error_message="Error durante el streaming de la respuesta.",
                is_final=True,
                final_response=error_response,
            )

    async def respond_to_approval(
        self,
        request: ApprovalResponseRequest,
    ) -> AssistantResponse:
        """
        Procesa la respuesta del usuario a una solicitud de aprobación.

        Resuelve la aprobación pendiente y permite que el TaskExecutor
        continúe o cancele la ejecución según la decisión.
        """
        start_time = time.perf_counter()

        logger.info(
            "aprobacion_respondida",
            session_id=request.session_id.to_str(),
            approval_id=request.approval_id.to_str(),
            granted=request.granted,
            remember=request.remember_decision,
        )

        try:
            # Resolver la aprobación en el ApprovalPort
            await self._approval_port.resolve(
                request.approval_id,
                granted=request.granted,
                remember_decision=request.remember_decision,
            )

            # Si el usuario aprobó y quiere recordar, actualizar el perfil
            if request.granted and request.remember_decision:
                await self._update_profile_with_approval(
                    request.session_id,
                    request.approval_id,
                )

            duration_ms = (time.perf_counter() - start_time) * 1000

            if request.granted:
                return AssistantResponse(
                    response_type=AssistantResponseType.THINKING,
                    session_id=request.session_id,
                    thinking_indicator=ThinkingIndicator(
                        message="Aprobado. Continuando con la acción...",
                    ),
                    duration_ms=duration_ms,
                )
            else:
                return AssistantResponse(
                    response_type=AssistantResponseType.TEXT,
                    session_id=request.session_id,
                    text_content=(
                        "Entendido, he cancelado esa acción. "
                        "¿En qué más puedo ayudarte?"
                    ),
                    duration_ms=duration_ms,
                )

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "aprobacion_fallo",
                session_id=request.session_id.to_str(),
                error=exc,
            )
            return self._build_error_response(
                session_id=request.session_id,
                message="Error al procesar tu respuesta. Intenta de nuevo.",
                is_retryable=True,
                duration_ms=duration_ms,
            )

    async def cancel_current_task(
        self,
        request: CancelTaskRequest,
    ) -> AssistantResponse:
        """Cancela la tarea activa de la sesión."""
        try:
            session = await self._session_store.load_session(request.session_id)
            if session is None:
                return self._build_error_response(
                    session_id=request.session_id,
                    message="Sesión no encontrada.",
                    is_retryable=False,
                )

            # Cancelar aprobaciones pendientes
            pending = await self._approval_port.get_pending(request.session_id)
            for approval_req in pending:
                await self._approval_port.cancel(approval_req.approval_id)

            return AssistantResponse(
                response_type=AssistantResponseType.TEXT,
                session_id=request.session_id,
                text_content="He cancelado la tarea actual. ¿En qué más puedo ayudarte?",
            )
        except Exception as exc:
            logger.error("cancelacion_fallida", error=exc)
            return self._build_error_response(
                session_id=request.session_id,
                message="Error al cancelar. Intenta de nuevo.",
                is_retryable=True,
            )

    async def create_session(
        self,
        request: CreateSessionRequest,
    ) -> tuple[SessionId, AssistantResponse]:
        """Crea una nueva sesión conversacional."""
        from src.domain.entities.session import Session
        from src.domain.value_objects.token_budget import BUDGET_128K

        try:
            # Cargar perfil del usuario
            profile = await self._load_or_create_profile(request.user_id)

            # Determinar budget según el modelo configurado
            budget = BUDGET_128K  # el bootstrap puede inyectar el correcto

            # Crear la sesión
            session = Session.create(
                user_id=request.user_id,
                token_budget=budget,
            )
            session.activate()

            # Persistir
            await self._session_store.save_session(session)
            await self._profile_store.increment_session_count(request.user_id)

            # Limpiar eventos de dominio
            session.clear_domain_events()

            logger.info(
                "sesion_creada",
                session_id=session.session_id.to_str(),
                user_id=request.user_id.to_str(),
            )

            # Mensaje de bienvenida personalizado
            display_name = profile.preferences.display_name
            welcome_text = self._build_welcome_message(display_name, profile)

            return session.session_id, AssistantResponse(
                response_type=AssistantResponseType.TEXT,
                session_id=session.session_id,
                text_content=welcome_text,
            )

        except Exception as exc:
            logger.error("creacion_sesion_fallida", error=exc)
            # Retornar un SessionId dummy para que la UI pueda manejar el error
            dummy_id = SessionId.generate()
            return dummy_id, self._build_error_response(
                session_id=dummy_id,
                message="Error al crear la sesión. Por favor, reinicia la aplicación.",
                is_retryable=False,
            )

    async def switch_session(
        self,
        request: SwitchSessionRequest,
    ) -> AssistantResponse:
        """Cambia a una sesión existente."""
        try:
            session = await self._session_store.load_session(request.session_id)
            if session is None:
                return self._build_error_response(
                    session_id=request.session_id,
                    message="La sesión solicitada no existe.",
                    is_retryable=False,
                )

            if not session.is_active:
                session.activate(was_restored=True)
                await self._session_store.save_session(session)

            return AssistantResponse(
                response_type=AssistantResponseType.TEXT,
                session_id=request.session_id,
                text_content=(
                    f"He retomado tu sesión anterior con {session.turn_count} "
                    f"mensajes. ¿En qué quieres continuar?"
                ),
                session_turn_count=session.turn_count,
            )
        except Exception as exc:
            logger.error("cambio_sesion_fallido", error=exc)
            return self._build_error_response(
                session_id=request.session_id,
                message="Error al cambiar de sesión.",
                is_retryable=True,
            )

    async def close_session(self, session_id: SessionId) -> AssistantResponse:
        """Cierra la sesión activa."""
        try:
            session = await self._session_store.load_session(session_id)
            if session is None:
                return self._build_error_response(
                    session_id=session_id,
                    message="Sesión no encontrada.",
                    is_retryable=False,
                )

            if not session.is_closed:
                session.close()
                await self._session_store.save_session(session)

            return AssistantResponse(
                response_type=AssistantResponseType.SESSION_CLOSED,
                session_id=session_id,
                text_content=(
                    f"Sesión cerrada. Tuvimos {session.turn_count} intercambios "
                    f"y produjimos {session.artifact_count} artefactos."
                ),
            )
        except Exception as exc:
            logger.error("cierre_sesion_fallido", error=exc)
            return self._build_error_response(
                session_id=session_id,
                message="Error al cerrar la sesión.",
                is_retryable=True,
            )

    async def list_sessions(
        self,
        user_id: UserId,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> SessionListResponse:
        """Lista las sesiones del usuario."""
        from src.ports.outbound.storage_port import SessionQuery

        try:
            query = SessionQuery(
                user_id=user_id,
                limit=limit,
                offset=offset,
                order_by="last_activity",
                order_dir="desc",
            )
            result = await self._session_store.list_sessions(query)

            session_infos = [
                SessionInfo(
                    session_id=s.session_id,
                    status=s.status,
                    created_at=s.created_at,
                    last_activity=s.last_activity,
                    turn_count=s.turn_count,
                    artifact_count=s.artifact_count,
                    duration_minutes=s.duration_minutes,
                )
                for s in result.items
            ]

            return SessionListResponse(
                sessions=session_infos,
                total_count=result.total_count,
                has_more=result.has_more,
            )
        except Exception as exc:
            logger.error("listado_sesiones_fallido", error=exc)
            return SessionListResponse(sessions=[], total_count=0, has_more=False)

    async def get_session_artifacts(
        self,
        session_id: SessionId,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ArtifactSummary]:
        """Retorna los artifacts de una sesión."""
        try:
            result = await self._artifact_store.list_by_session(
                session_id,
                limit=limit,
                offset=offset,
            )
            return result.items
        except Exception as exc:
            logger.error("artifacts_sesion_fallido", error=exc)
            return []

    async def get_artifact_content(
        self,
        artifact_id: ArtifactId,
        session_id: SessionId,
    ) -> dict[str, Any]:
        """Retorna el contenido completo de un artifact."""
        try:
            artifact = await self._artifact_store.load(artifact_id)
            if artifact is None or artifact.session_id != session_id:
                return {"error": "Artifact no encontrado."}
            return {
                "artifact_id": artifact.artifact_id.to_str(),
                "type": artifact.artifact_type.value,
                "display_name": artifact.display_name,
                "content": artifact.content.raw_content,
                "content_type": artifact.content.content_type.value,
                "metadata": artifact.metadata.to_index_dict(),
                "created_at": artifact.created_at.isoformat(),
            }
        except Exception as exc:
            logger.error("contenido_artifact_fallido", error=exc)
            return {"error": "Error al cargar el artifact."}

    async def export_artifact(
        self,
        artifact_id: ArtifactId,
        session_id: SessionId,
        *,
        format: str = "markdown",
    ) -> bytes:
        """Exporta un artifact en el formato especificado."""
        try:
            artifact = await self._artifact_store.load(artifact_id)
            if artifact is None or artifact.session_id != session_id:
                return b""

            content = artifact.content.raw_content

            if format == "json":
                import json
                export_data = {
                    "name": artifact.display_name,
                    "type": artifact.artifact_type.value,
                    "created_at": artifact.created_at.isoformat(),
                    "content": content,
                }
                return json.dumps(export_data, ensure_ascii=False, indent=2).encode("utf-8")

            elif format == "txt":
                return content.encode("utf-8")

            else:  # markdown (default)
                header = f"# {artifact.display_name}\n\n"
                metadata = f"*Tipo: {artifact.artifact_type.value} | Creado: {artifact.created_at.strftime('%Y-%m-%d %H:%M')}*\n\n"
                return (header + metadata + content).encode("utf-8")

        except Exception as exc:
            logger.error("exportacion_artifact_fallida", error=exc)
            return b""

    async def get_current_session_state(
        self,
        session_id: SessionId,
    ) -> dict[str, Any]:
        """Retorna el estado actual de la sesión para sincronización de la UI."""
        try:
            session = await self._session_store.load_session(session_id)
            if session is None:
                return {"error": "Sesión no encontrada."}

            pending = await self._approval_port.get_pending(session_id)
            return {
                "session_id": session_id.to_str(),
                "status": session.status.value,
                "turn_count": session.turn_count,
                "artifact_count": session.artifact_count,
                "task_count": session.task_count,
                "pending_approvals": len(pending),
                "has_compaction": session.compaction_summary is not None,
                "last_activity": session.last_activity.isoformat(),
            }
        except Exception as exc:
            logger.error("estado_sesion_fallido", error=exc)
            return {"error": "Error al obtener el estado de la sesión."}

    # =========================================================================
    # FLUJO INTERNO DE PROCESAMIENTO
    # =========================================================================

    async def _process_turn(
        self,
        request: UserMessageRequest,
        request_id: RequestId,
        start_time: float,
    ) -> AssistantResponse:
        """
        Flujo interno completo de procesamiento de un turno.

        Orquesta todos los servicios para procesar el mensaje del usuario
        y producir la respuesta final. Toda la lógica de coordinación está aquí.

        Args:
            request:    UserMessageRequest del usuario.
            request_id: ID de la request para trazabilidad.
            start_time: perf_counter del inicio del turno.

        Returns:
            AssistantResponse completa para la UI.
        """
        # 1. Cargar sesión y perfil
        session = await self._session_store.load_session(request.session_id)
        if session is None:
            return self._build_error_response(
                session_id=request.session_id,
                message="La sesión no existe o fue cerrada. Crea una nueva sesión.",
                is_retryable=False,
            )

        if not session.is_active:
            session.activate(was_restored=True)

        profile = await self._load_or_create_profile(session.user_id)

        # 2. Cargar historial de turns
        turns_data = await self._session_store.load_recent_turns(
            request.session_id,
            max_tokens=session._token_budget.compute_allocation().available_for_history,
        )

        # 3. Obtener schemas de tools disponibles
        from forge_core.tools.protocol import Platform
        platform = Platform(self._llm.provider_name) if hasattr(Platform, self._llm.provider_name.upper()) else Platform.ALL
        available_tools = self._tool_registry.get_schemas_for_llm(
            Platform.LINUX if "linux" in __import__("sys").platform else Platform.WINDOWS,
            profile.permissions,
        )

        # 4. Construir el ProfileSummary para el LLM
        profile_summary = profile.to_profile_summary()

        # 5. Construir el system prompt
        system_prompt = self._system_prompt_builder.build(
            profile_summary,
            available_tool_count=len(available_tools),
        )

        # 6. Construir mensajes del historial
        from forge_core.llm.protocol import LLMMessage
        history_messages = self._build_history_messages(turns_data)

        # Añadir el system prompt al inicio
        all_messages = [LLMMessage.system(system_prompt)] + history_messages

        # 7. Construir el contexto de conversación
        budget_allocation = session._token_budget.compute_allocation(
            active_tools_token_estimate=self._tool_registry.estimate_schemas_token_count(
                Platform.LINUX,
                profile.permissions,
            )
        )

        conversation_context = ConversationContext(
            session_id=request.session_id,
            user_message=request.get_full_text_for_context(),
            conversation_history=all_messages,
            profile_summary=profile_summary,
            available_tools=available_tools,
            budget_allocation=budget_allocation,
            compaction_summary=session.compaction_summary,
            request_id=request_id,
            attachments_text=self._extract_image_context(request),
        )

        # 8. Llamar al LLM
        try:
            llm_response = await self._llm.generate_conversation(conversation_context)
        except ForgeLLMError as e:
            logger.error("llm_fallo", error=e)
            return self._build_error_response(
                session_id=request.session_id,
                message=self._llm_error_to_user_message(e),
                is_retryable=e.is_retryable(),
            )

        # 9. Enrutar la respuesta del LLM
        routing = self._llm.get_routing_decision(
            llm_response,
            mutation_tool_ids=self._tool_registry.get_mutation_tool_ids(),
            planning_threshold=self._planning_threshold,
        )

        # 10. Ejecutar según el routing
        artifacts_produced: list[UIArtifactInfo] = []
        final_text = ""
        approval_info = None

        if isinstance(routing, DirectReply):
            final_text = routing.content

        elif isinstance(routing, SingleToolCall):
            result = await self._execute_single_tool(
                routing, request, profile, session
            )
            artifacts_produced = result.get("artifacts", [])
            approval_info = result.get("approval_info")
            final_text = result.get("text", "")

            # Si hay approval pendiente, retornar inmediatamente
            if approval_info is not None:
                return AssistantResponse(
                    response_type=AssistantResponseType.APPROVAL_REQUEST,
                    session_id=request.session_id,
                    approval_request=approval_info,
                    duration_ms=(time.perf_counter() - start_time) * 1000,
                )

        elif isinstance(routing, MultipleToolCalls):
            result = await self._execute_multiple_tools(
                routing, request, profile, session
            )
            artifacts_produced = result.get("artifacts", [])
            final_text = result.get("text", "")

        elif isinstance(routing, PlanningRequired):
            result = await self._execute_with_planning(
                routing, request, profile, session
            )
            artifacts_produced = result.get("artifacts", [])
            final_text = result.get("text", "")

        # 11. Persistir el turno
        turn_id = TurnId.generate()
        sequence_number = session.turn_count + 1
        estimated_tokens = TokenEstimator.estimate(
            request.message + final_text
        )

        turn_ref = TurnReference(
            turn_id=turn_id,
            sequence_number=sequence_number,
            estimated_tokens=estimated_tokens,
            created_at=datetime.now(tz=timezone.utc),
            has_tool_calls=not isinstance(routing, DirectReply),
        )

        if session.is_active:
            session.add_turn(turn_ref)
            await self._session_store.save_session(session)
            await self._session_store.append_turn(
                request.session_id,
                turn_ref,
                {
                    "user_message": request.message,
                    "assistant_response": final_text,
                    "routing_type": type(routing).__name__,
                    "tool_calls_count": len(getattr(routing, "tool_calls", [])),
                },
            )

        # 12. Determinar tipo de respuesta
        response_type = (
            AssistantResponseType.TEXT_WITH_ARTIFACTS
            if artifacts_produced
            else AssistantResponseType.TEXT
        )

        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "turno_completado",
            session_id=request.session_id.to_str(),
            routing=type(routing).__name__,
            artifacts_count=len(artifacts_produced),
            duration_ms=round(duration_ms, 2),
        )

        return AssistantResponse(
            response_type=response_type,
            session_id=request.session_id,
            text_content=final_text,
            artifacts_produced=artifacts_produced,
            turn_id=turn_id.to_str(),
            duration_ms=duration_ms,
            session_turn_count=session.turn_count,
            suggested_actions=self._generate_suggested_actions(
                routing, artifacts_produced
            ),
        )

    async def _execute_single_tool(
        self,
        routing: SingleToolCall,
        request: UserMessageRequest,
        profile: UserProfile,
        session: Any,
    ) -> dict[str, Any]:
        """
        Ejecuta un single tool call con enforcement de policy.

        Retorna un dict con 'text', 'artifacts', y 'approval_info' si hay
        una aprobación pendiente.
        """
        from src.domain.value_objects.identifiers import InvocationId
        from src.ports.outbound.tool_port import ToolExecutionRequest

        tool_call = routing.tool_call
        invocation_id = InvocationId.generate()

        exec_request = ToolExecutionRequest(
            tool_id=tool_call.name,
            arguments=tool_call.arguments,
            session_id=request.session_id,
            invocation_id=invocation_id,
        )

        # Ejecutar la tool (el ToolDispatchPort maneja policy + approval)
        exec_result = await self._tool_dispatch.dispatch(
            exec_request,
            profile.permissions,
        )

        # Verificar si hay una aprobación pendiente
        pending = await self._approval_port.get_pending(request.session_id)
        if pending:
            # Hay una aprobación pendiente — devolver al coordinador
            from src.ports.inbound.conversation_port import ApprovalRequestInfo
            from forge_core.tools.protocol import RiskLevel
            req = pending[0]
            return {
                "approval_info": ApprovalRequestInfo(
                    approval_id=req.approval_id,
                    description=req.description,
                    risk_label=req.risk_label,
                    tool_name=req.tool_display_name,
                    can_remember=req.can_remember_decision,
                    context_summary=req.context_summary,
                    risk_explanation=req.risk_explanation,
                    expires_in_seconds=max(0, int(req.seconds_remaining)),
                    is_first_time=req.is_first_time,
                ),
                "text": "",
                "artifacts": [],
            }

        if not exec_result.success:
            return {
                "text": f"No pude completar la acción: {exec_result.error_message or 'Error desconocido'}",
                "artifacts": [],
            }

        # Sintetizar la respuesta con el resultado de la tool
        from forge_core.llm.protocol import LLMMessage, ToolResult
        profile_summary = profile.to_profile_summary()

        tool_result = ToolResult.success(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content=exec_result.output_for_llm,
        )

        synthesis_context = SynthesisContext(
            session_id=request.session_id,
            original_user_message=request.message,
            tool_results=[tool_result],
            conversation_history=[],
            profile_summary=profile_summary,
            budget_allocation=session._token_budget.compute_allocation(),
        )

        try:
            synthesis_response = await self._llm.generate_synthesis(synthesis_context)
            return {
                "text": synthesis_response.content,
                "artifacts": [],
            }
        except ForgeLLMError as e:
            return {
                "text": exec_result.output_for_llm[:2000],  # fallback al output crudo
                "artifacts": [],
            }

    async def _execute_multiple_tools(
        self,
        routing: MultipleToolCalls,
        request: UserMessageRequest,
        profile: UserProfile,
        session: Any,
    ) -> dict[str, Any]:
        """
        Ejecuta múltiples tool calls secuencialmente.
        """
        from src.domain.value_objects.identifiers import InvocationId
        from src.ports.outbound.tool_port import ToolExecutionRequest
        from forge_core.llm.protocol import ToolResult

        tool_results = []
        for tool_call in routing.tool_calls:
            invocation_id = InvocationId.generate()
            exec_request = ToolExecutionRequest(
                tool_id=tool_call.name,
                arguments=tool_call.arguments,
                session_id=request.session_id,
                invocation_id=invocation_id,
            )
            result = await self._tool_dispatch.dispatch(
                exec_request,
                profile.permissions,
            )
            tool_results.append(ToolResult.success(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                content=result.output_for_llm,
            ) if result.success else ToolResult.error(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                error_message=result.error_message or "Error desconocido",
            ))

        # Sintetizar todos los resultados
        profile_summary = profile.to_profile_summary()
        synthesis_context = SynthesisContext(
            session_id=request.session_id,
            original_user_message=request.message,
            tool_results=tool_results,
            conversation_history=[],
            profile_summary=profile_summary,
            budget_allocation=session._token_budget.compute_allocation(),
        )

        try:
            synthesis_response = await self._llm.generate_synthesis(synthesis_context)
            return {"text": synthesis_response.content, "artifacts": []}
        except ForgeLLMError:
            combined = "\n\n".join(r.content for r in tool_results)
            return {"text": combined[:3000], "artifacts": []}

    async def _execute_with_planning(
        self,
        routing: PlanningRequired,
        request: UserMessageRequest,
        profile: UserProfile,
        session: Any,
    ) -> dict[str, Any]:
        """
        Ejecuta una tarea compleja con planificación multi-step.

        Delega al LightweightPlanner para generar el plan y al TaskExecutor
        para ejecutarlo. En esta implementación inicial, ejecuta los tool calls
        semilla directamente si los hay, y luego sintetiza.
        """
        # En la implementación completa, esto usaría el LightweightPlanner
        # Por ahora, ejecutar los tool calls semilla directamente
        if routing.initial_tool_calls:
            from forge_core.llm.protocol import ToolResult
            from src.domain.value_objects.identifiers import InvocationId
            from src.ports.outbound.tool_port import ToolExecutionRequest

            tool_results = []
            for tool_call in routing.initial_tool_calls:
                invocation_id = InvocationId.generate()
                exec_request = ToolExecutionRequest(
                    tool_id=tool_call.name,
                    arguments=tool_call.arguments,
                    session_id=request.session_id,
                    invocation_id=invocation_id,
                )
                result = await self._tool_dispatch.dispatch(
                    exec_request,
                    profile.permissions,
                )
                tool_results.append(ToolResult.success(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    content=result.output_for_llm,
                ) if result.success else ToolResult.error(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    error_message=result.error_message or "Error",
                ))

            profile_summary = profile.to_profile_summary()
            synthesis_context = SynthesisContext(
                session_id=request.session_id,
                original_user_message=request.message,
                tool_results=tool_results,
                conversation_history=[],
                profile_summary=profile_summary,
                budget_allocation=session._token_budget.compute_allocation(),
                task_description=routing.complexity_hint,
            )

            try:
                synthesis_response = await self._llm.generate_synthesis(synthesis_context)
                return {"text": synthesis_response.content, "artifacts": []}
            except ForgeLLMError:
                pass

        return {"text": "He procesado tu solicitud.", "artifacts": []}

    # =========================================================================
    # HELPERS INTERNOS
    # =========================================================================

    async def _load_or_create_profile(self, user_id: UserId) -> UserProfile:
        """Carga el perfil del usuario o crea uno por defecto si no existe."""
        profile = await self._profile_store.load(user_id)
        if profile is None:
            profile = UserProfile.create_default()
            # Ajustar el user_id al del usuario actual
            profile._user_id = user_id
            await self._profile_store.save(profile)
        return profile

    def _build_history_messages(
        self,
        turns_data: list[dict[str, Any]],
    ) -> list[Any]:
        """Construye los mensajes del historial desde los datos de turns."""
        from forge_core.llm.protocol import LLMMessage
        messages = []
        for turn in turns_data:
            user_msg = turn.get("user_message", "")
            assistant_msg = turn.get("assistant_response", "")
            if user_msg:
                messages.append(LLMMessage.user(user_msg))
            if assistant_msg:
                messages.append(LLMMessage.assistant(assistant_msg))
        return messages

    def _extract_image_context(self, request: UserMessageRequest) -> str | None:
        """Extrae el contexto de imágenes adjuntas si las hay."""
        image_attachments = [a for a in request.attachments if a.is_image]
        if not image_attachments:
            return None
        return "\n".join(
            f"[Imagen adjunta: {a.filename}]"
            for a in image_attachments
        )

    def _llm_error_to_user_message(self, error: ForgeLLMError) -> str:
        """Convierte un error LLM en un mensaje legible para el usuario."""
        from forge_core.errors.types import (
            LLMConnectionError,
            LLMRateLimitError,
            LLMTimeoutError,
            LLMContentFilterError,
        )
        if isinstance(error, LLMConnectionError):
            return "No puedo conectarme al servicio de IA en este momento. Verifica tu conexión a internet."
        if isinstance(error, LLMRateLimitError):
            return "El servicio de IA está muy ocupado en este momento. Intenta en unos segundos."
        if isinstance(error, LLMTimeoutError):
            return "La respuesta tardó demasiado. Intenta con una pregunta más corta."
        if isinstance(error, LLMContentFilterError):
            return "No puedo procesar esta solicitud. Por favor, reformula tu mensaje."
        return "Ha ocurrido un error al procesar tu mensaje. Intenta de nuevo."

    def _build_welcome_message(
        self,
        display_name: str,
        profile: UserProfile,
    ) -> str:
        """Construye el mensaje de bienvenida personalizado."""
        if profile.session_count <= 1:
            greeting = f"¡Hola{', ' + display_name if display_name != 'Usuario' else ''}! "
            greeting += (
                "Soy Forge, tu asistente personal. Puedo ayudarte a buscar información, "
                "analizar documentos, organizar ideas y ejecutar acciones en tu computadora. "
                "¿En qué puedo ayudarte hoy?"
            )
        else:
            greeting = f"¡Bienvenido de vuelta{', ' + display_name if display_name != 'Usuario' else ''}! "
            greeting += "¿En qué puedo ayudarte hoy?"
        return greeting

    def _generate_suggested_actions(
        self,
        routing: RoutingDecision,
        artifacts: list[UIArtifactInfo],
    ) -> list[str]:
        """Genera sugerencias de acciones siguientes según el contexto."""
        suggestions: list[str] = []

        # Sugerencias basadas en artifacts producidos
        for artifact in artifacts[:2]:  # Máximo 2 sugerencias de artifacts
            if artifact.artifact_type.value == "summary":
                suggestions.append("Generar flashcards del resumen")
                suggestions.append("Crear un quiz")
            elif artifact.artifact_type.value == "search_results":
                suggestions.append("Profundizar en uno de los resultados")

        return suggestions[:3]  # Máximo 3 sugerencias

    def _build_error_response(
        self,
        session_id: SessionId,
        message: str,
        is_retryable: bool = False,
        duration_ms: float = 0.0,
    ) -> AssistantResponse:
        """Construye una AssistantResponse de error."""
        return AssistantResponse(
            response_type=AssistantResponseType.ERROR,
            session_id=session_id,
            error_message=message,
            error_is_retryable=is_retryable,
            duration_ms=duration_ms,
        )

    async def _update_profile_with_approval(
        self,
        session_id: SessionId,
        approval_id: ApprovalId,
    ) -> None:
        """Actualiza el perfil del usuario con el permiso aprobado."""
        # Se implementará cuando tengamos el ProfileStore completo
        # Por ahora, solo loguear
        logger.info(
            "permiso_recordado",
            session_id=session_id.to_str(),
            approval_id=approval_id.to_str(),
        )

    @staticmethod
    def _split_into_streaming_chunks(text: str) -> list[str]:
        """
        Divide el texto en chunks para simulación de streaming.

        En una implementación real, el streaming viene directo del LLM.
        Este método es para el caso de batch → streaming conversion.
        """
        # Dividir por oraciones para chunks naturales
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s + " " for s in sentences if s]


# =============================================================================
# ADMIN PORT STUB
# =============================================================================


class ConversationAdminPort(AdminPort):
    """
    Implementación del AdminPort para el debug CLI.

    Accede a todos los servicios internos directamente para proporcionar
    diagnósticos completos al administrador del sistema.
    """

    def __init__(
        self,
        session_store: SessionStorePort,
        artifact_store: ArtifactStorePort,
        audit_log: AuditLogPort,
        tool_registry: ToolRegistryPort,
        policy_engine: PolicyEnginePort,
    ) -> None:
        self._session_store = session_store
        self._artifact_store = artifact_store
        self._audit_log = audit_log
        self._tool_registry = tool_registry
        self._policy_engine = policy_engine

    async def get_session_list(
        self,
        *,
        limit: int = 20,
        include_closed: bool = True,
    ) -> list[dict[str, Any]]:
        from src.ports.outbound.storage_port import SessionQuery
        query = SessionQuery(limit=limit, order_by="last_activity", order_dir="desc")
        result = await self._session_store.list_sessions(query)
        return [s.to_display_dict() for s in result.items]

    async def inspect_session(self, session_id: SessionId) -> dict[str, Any]:
        session = await self._session_store.load_session(session_id)
        if session is None:
            return {"error": "Sesión no encontrada."}
        return session.to_state_dict()

    async def query_audit_log(
        self,
        *,
        session_id: str | None = None,
        tool_id: str | None = None,
        last_hours: int = 24,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        from src.ports.outbound.storage_port import AuditQuery
        from src.domain.value_objects.identifiers import SessionId as SID
        query = AuditQuery(
            session_id=SID.from_str(session_id) if session_id else None,
            tool_id=tool_id,
            limit=limit,
        )
        result = await self._audit_log.query(query)
        return [e.to_display_dict() for e in result.items]

    async def get_policy_status(self) -> dict[str, Any]:
        policies = self._policy_engine.get_active_policies()
        return {
            "active_policies": len(policies),
            "circuit_breaker": self._policy_engine.get_circuit_breaker_state(),
            "total_evaluations": self._policy_engine.total_evaluations,
            "total_denials": self._policy_engine.total_denials,
            "total_security_events": self._policy_engine.total_security_events,
            "policies": [
                {
                    "name": p.name,
                    "priority": p.priority,
                    "enabled": p.enabled,
                    "description": p.description,
                }
                for p in policies
            ],
        }

    async def get_tool_list(
        self,
        *,
        category: str | None = None,
        include_disabled: bool = False,
    ) -> list[dict[str, Any]]:
        from forge_core.tools.protocol import ToolCategory
        from src.ports.outbound.tool_port import ToolFilter

        tool_filter = None
        if category:
            try:
                cat = ToolCategory(category)
                tool_filter = ToolFilter.by_category(cat)
            except ValueError:
                pass

        registrations = self._tool_registry.list(tool_filter)
        return [
            {
                "tool_id": r.tool_id,
                "category": r.category.value,
                "risk_level": r.risk_level.name,
                "enabled": r.schema.enabled,
                "invocations": r.invocation_count,
                "failures": r.failure_count,
                "healthy": r.is_healthy,
            }
            for r in registrations
            if include_disabled or r.schema.enabled
        ]

    async def get_security_report(
        self,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        if session_id:
            return await self._policy_engine.get_security_report(session_id)
        return {
            "circuit_breaker_open": self._policy_engine.is_circuit_breaker_open,
            "total_evaluations": self._policy_engine.total_evaluations,
            "total_denials": self._policy_engine.total_denials,
            "total_security_events": self._policy_engine.total_security_events,
        }

    async def get_metrics_summary(self) -> dict[str, Any]:
        return {
            "tools_registered": self._tool_registry.registered_count,
            "tools_enabled": self._tool_registry.enabled_count,
            "policy_evaluations": self._policy_engine.total_evaluations,
            "policy_denials": self._policy_engine.total_denials,
        }

    async def get_config(self) -> dict[str, Any]:
        return {"note": "Configuración disponible en el bootstrap del sistema."}

    async def reset_session_security(self, session_id: str) -> dict[str, Any]:
        self._policy_engine.reset_session_security(session_id)
        return {"status": "ok", "session_id": session_id, "message": "Seguridad de sesión reiniciada."}

    async def force_session_close(self, session_id: SessionId) -> dict[str, Any]:
        session = await self._session_store.load_session(session_id)
        if session is None:
            return {"error": "Sesión no encontrada."}
        if not session.is_closed:
            session.close()
            await self._session_store.save_session(session)
        return {"status": "closed", "session_id": session_id.to_str()}