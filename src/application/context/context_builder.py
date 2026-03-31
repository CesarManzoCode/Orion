"""
ContextBuilder — ensamblador del contexto para cada invocación al LLM.

El ContextBuilder es el servicio de aplicación responsable de construir el
LLMContext óptimo para cada turno de conversación. Es una pieza crítica del
sistema — un contexto mal construido produce respuestas de baja calidad,
excesos del context window, o pérdida de información conversacional relevante.

Responsabilidades:
  1. Distribuir el token budget entre los componentes del contexto
  2. Seleccionar qué turns del historial incluir (los más recientes que quepan)
  3. Incluir el system prompt con personalidad + preferencias del usuario
  4. Añadir los schemas de tools disponibles para function calling
  5. Incluir el resumen compactado del historial si existe
  6. Añadir metadata de artifacts activos relevantes
  7. Incorporar el contenido de archivos adjuntos del usuario
  8. Calcular el TokenBudget exacto para el adapter del LLM

Estrategia de construcción del contexto (orden fijo):
  [1] System prompt           (~2,000 tokens, fijo)
  [2] Tool schemas            (~500-3,000 tokens, según tools activas)
  [3] User profile summary    (~200-300 tokens, fijo)
  [4] Compaction summary      (~500 tokens, si existe)
  [5] Conversation history    (variable, hasta agotar el budget restante)
  [6] Current user message    (variable, incluye adjuntos)

La estrategia prioriza siempre los turns más recientes. Si el historial
completo no cabe, se trunca desde el más antiguo hacia adelante.

Principios de diseño:
  1. El budget se calcula antes de construir el contexto — no se descubre
     al final que se excedió el límite.
  2. Cada componente tiene un límite máximo independiente que el builder respeta.
  3. Los tokens se estiman con heurística rápida — la precisión exacta la
     provee el adapter del LLM si es necesaria.
  4. Los adjuntos de imagen se manejan como contenido multimodal separado.
  5. La construcción es determinista: dado el mismo estado, produce el mismo contexto.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any

from forge_core.llm.protocol import (
    DocumentContent,
    ImageContent,
    LLMContext,
    LLMMessage,
    MessageRole,
    TextContent,
    TokenBudget,
    ToolDefinition,
)
from forge_core.observability.logging import get_logger
from forge_core.tools.protocol import Platform

from src.domain.entities.artifact import Artifact, ArtifactType
from src.domain.entities.session import Session
from src.domain.entities.user_profile import ProfileSummary, UserProfile
from src.domain.value_objects.identifiers import RequestId, SessionId, TaskId
from src.domain.value_objects.token_budget import (
    BudgetAllocation,
    ContextTokenBudget,
    TokenEstimator,
)
from src.ports.inbound.conversation_port import AttachedFile
from src.ports.outbound.llm_port import (
    CompactionContext,
    ConversationContext,
    PlanningContext,
    SynthesisContext,
    SystemPromptBuilder,
)
from src.ports.outbound.storage_port import ArtifactStorePort
from src.ports.outbound.tool_port import ToolRegistryPort


logger = get_logger(__name__, component="context_builder")


# Límites de tamaño de componentes del contexto (en tokens estimados)
_MAX_SYSTEM_PROMPT_TOKENS = 2_500
_MAX_TOOL_SCHEMAS_TOKENS = 4_000
_MAX_USER_PROFILE_TOKENS = 400
_MAX_COMPACTION_SUMMARY_TOKENS = 700
_MAX_ARTIFACT_CONTEXT_TOKENS = 1_500
_MAX_ATTACHMENT_TOKENS = 8_000
_MAX_SINGLE_TURN_TOKENS = 4_000


class ContextBuilder:
    """
    Ensamblador del contexto LLM para HiperForge User.

    Construye el LLMContext óptimo para cada turno de conversación
    distribuyendo el token budget de forma inteligente entre todos
    los componentes del contexto.

    El ContextBuilder es stateless — no mantiene estado entre invocaciones.
    Puede usarse como singleton o instanciarse por request.

    Inyección de dependencias:
    - ToolRegistryPort: para obtener los schemas de tools
    - ArtifactStorePort: para cargar artifacts activos de la sesión
    - SystemPromptBuilder: para construir el system prompt personalizado
    """

    def __init__(
        self,
        tool_registry: ToolRegistryPort,
        artifact_store: ArtifactStorePort,
        *,
        max_artifacts_in_context: int = 3,
        artifact_content_max_tokens: int = 500,
    ) -> None:
        """
        Args:
            tool_registry:              Registro de tools disponibles.
            artifact_store:             Store de artifacts.
            max_artifacts_in_context:   Máximo de artifacts a incluir en el contexto.
            artifact_content_max_tokens: Tokens máximos por artifact en el contexto.
        """
        self._tool_registry = tool_registry
        self._artifact_store = artifact_store
        self._system_prompt_builder = SystemPromptBuilder()
        self._max_artifacts = max_artifacts_in_context
        self._artifact_max_tokens = artifact_content_max_tokens

        # Detectar plataforma actual
        platform_map = {"linux": Platform.LINUX, "win32": Platform.WINDOWS}
        self._platform = platform_map.get(sys.platform, Platform.LINUX)

    # =========================================================================
    # CONSTRUCCIÓN DE CONTEXTO CONVERSACIONAL
    # =========================================================================

    async def build_conversation_context(
        self,
        session: Session,
        profile: UserProfile,
        user_message: str,
        turns_data: list[dict[str, Any]],
        *,
        attachments: list[AttachedFile] | None = None,
        tool_results_messages: list[LLMMessage] | None = None,
        active_artifact_ids: list[str] | None = None,
        request_id: RequestId | None = None,
        task_id: TaskId | None = None,
    ) -> ConversationContext:
        """
        Construye el ConversationContext completo para un turno normal.

        Es el método principal del ContextBuilder. Ensambla todos los
        componentes del contexto y calcula el BudgetAllocation exacto.

        Args:
            session:                Session activa.
            profile:                Perfil del usuario.
            user_message:           Mensaje del usuario para este turno.
            turns_data:             Datos de turns del historial (desde SessionManager).
            attachments:            Archivos adjuntos del usuario.
            tool_results_messages:  Resultados de tools (para síntesis).
            active_artifact_ids:    IDs de artifacts activos a incluir.
            request_id:             ID de la request para trazabilidad.
            task_id:                ID de la tarea activa.

        Returns:
            ConversationContext completo para pasar al UserLLMPort.
        """
        # 1. Obtener schemas de tools disponibles
        available_tools = self._tool_registry.get_schemas_for_llm(
            self._platform,
            profile.permissions,
        )

        # 2. Calcular el BudgetAllocation con la estimación real de tools
        tool_tokens_estimate = self._tool_registry.estimate_schemas_token_count(
            self._platform,
            profile.permissions,
        )
        allocation = session._token_budget.compute_allocation(
            active_tools_token_estimate=tool_tokens_estimate,
        )

        if not allocation.is_viable:
            logger.warning(
                "budget_no_viable",
                session_id=session.session_id.to_str(),
                available_tokens=allocation.available_for_history,
            )

        # 3. Construir el ProfileSummary
        profile_summary = profile.to_profile_summary()

        # 4. Construir el historial de mensajes
        history_messages = self._build_history_messages(
            turns_data=turns_data,
            compaction_summary=session.compaction_summary,
            budget_for_history=allocation.available_for_history,
            tool_results_messages=tool_results_messages,
        )

        # 5. Añadir artifacts activos al final del historial como contexto
        if active_artifact_ids:
            artifact_messages = await self._build_artifact_context_messages(
                active_artifact_ids,
                session.session_id,
            )
            history_messages.extend(artifact_messages)

        # 6. Construir el mensaje del usuario con adjuntos
        user_message_content = self._build_user_message_content(
            user_message=user_message,
            attachments=attachments or [],
        )

        # Añadir el mensaje del usuario actual al historial
        if user_message_content:
            history_messages.append(LLMMessage.user(user_message_content))

        # 7. Construir el system prompt completo
        system_prompt = self._system_prompt_builder.build(
            profile_summary,
            available_tool_count=len(available_tools),
        )

        # 8. Construir la lista completa de mensajes (system + historial)
        all_messages = [LLMMessage.system(system_prompt)] + history_messages

        # 9. Calcular el TokenBudget para el adapter
        token_budget = allocation.to_forge_token_budget()

        logger.debug(
            "contexto_construido",
            session_id=session.session_id.to_str(),
            history_turns=len(turns_data),
            messages_total=len(all_messages),
            tools_count=len(available_tools),
            has_compaction=session.compaction_summary is not None,
            has_attachments=bool(attachments),
        )

        return ConversationContext(
            session_id=session.session_id,
            user_message=user_message,
            conversation_history=all_messages,
            profile_summary=profile_summary,
            available_tools=available_tools,
            budget_allocation=allocation,
            compaction_summary=session.compaction_summary,
            request_id=request_id,
            task_id=task_id,
        )

    async def build_synthesis_context(
        self,
        session: Session,
        profile: UserProfile,
        original_user_message: str,
        tool_results_messages: list[LLMMessage],
        turns_data: list[dict[str, Any]],
        *,
        task_description: str = "",
        request_id: RequestId | None = None,
        task_id: TaskId | None = None,
    ) -> SynthesisContext:
        """
        Construye el SynthesisContext para sintetizar resultados de tools.

        La síntesis no incluye tool schemas (las tools ya se ejecutaron)
        pero sí el historial reciente para que el LLM tenga contexto de
        qué preguntó el usuario y qué resultó.

        Args:
            session:               Session activa.
            profile:               Perfil del usuario.
            original_user_message: Mensaje original que originó las tools.
            tool_results_messages: Mensajes con los resultados de las tools.
            turns_data:            Historial de turns para contexto.
            task_description:      Descripción de la tarea (para planes).
            request_id:            ID de la request.
            task_id:               ID de la tarea.

        Returns:
            SynthesisContext para el UserLLMPort.
        """
        allocation = session._token_budget.compute_allocation()
        profile_summary = profile.to_profile_summary()

        # Historial sin tool schemas (síntesis es más compacta)
        history_messages = self._build_history_messages(
            turns_data=turns_data,
            compaction_summary=session.compaction_summary,
            budget_for_history=allocation.available_for_history // 2,  # más conservador
        )

        # Extraer ToolResult de los mensajes
        from forge_core.llm.protocol import ToolResult
        tool_results = []
        for msg in tool_results_messages:
            if msg.role == MessageRole.TOOL and msg.tool_call_id:
                tool_results.append(ToolResult(
                    tool_call_id=msg.tool_call_id,
                    tool_name=msg.name or "unknown",
                    content=msg.get_text(),
                ))

        return SynthesisContext(
            session_id=session.session_id,
            original_user_message=original_user_message,
            tool_results=tool_results,
            conversation_history=history_messages,
            profile_summary=profile_summary,
            budget_allocation=allocation,
            task_description=task_description,
            request_id=request_id,
            task_id=task_id,
        )

    async def build_planning_context(
        self,
        session: Session,
        profile: UserProfile,
        task_description: str,
        available_tool_ids: list[str],
        max_steps: int = 10,
        *,
        request_id: RequestId | None = None,
        task_id: TaskId | None = None,
    ) -> PlanningContext:
        """
        Construye el PlanningContext para generación de planes multi-step.

        La planificación necesita un contexto más compacto — el LLM debe
        enfocarse en el task_description y las tools disponibles, no en
        el historial completo.

        Args:
            session:            Session activa.
            profile:            Perfil del usuario.
            task_description:   Descripción de la tarea a planificar.
            available_tool_ids: IDs de tools disponibles para el plan.
            max_steps:          Máximo de pasos permitidos en el plan.
            request_id:         ID de la request.
            task_id:            ID de la tarea.

        Returns:
            PlanningContext para el UserLLMPort.
        """
        profile_summary = profile.to_profile_summary()

        # Obtener schemas de tools disponibles para el plan
        available_tools = self._tool_registry.get_schemas_for_llm(
            self._platform,
            profile.permissions,
        )

        # Filtrar solo las tools especificadas en available_tool_ids
        if available_tool_ids:
            available_tools = [
                t for t in available_tools
                if t.name in available_tool_ids
            ]

        # Resumen de conversación compacto para el planner
        allocation = session._token_budget.compute_allocation()
        conversation_summary = (
            session.compaction_summary
            or f"Conversación activa con {session.turn_count} turnos."
        )

        return PlanningContext(
            session_id=session.session_id,
            task_description=task_description,
            available_tools=available_tools,
            max_steps=max_steps,
            conversation_summary=conversation_summary,
            profile_summary=profile_summary,
            request_id=request_id,
            task_id=task_id,
        )

    async def build_compaction_context(
        self,
        session: Session,
        turns_to_compact: list[dict[str, Any]],
        max_summary_tokens: int = 500,
    ) -> CompactionContext:
        """
        Construye el CompactionContext para generación de resúmenes.

        Convierte los datos crudos de turns en LLMMessages para que
        el LLM pueda generar el resumen de compactación.

        Args:
            session:             Session activa.
            turns_to_compact:    Datos de turns a compactar.
            max_summary_tokens:  Longitud máxima del resumen.

        Returns:
            CompactionContext para el UserLLMPort.
        """
        # Convertir turns a LLMMessages para el contexto de compactación
        turns_messages = self._build_history_messages(
            turns_data=turns_to_compact,
            compaction_summary=None,  # no incluir summary previo
            budget_for_history=max_summary_tokens * 10,  # budget amplio para compactar
        )

        # Detectar idioma del perfil (para el summary en el mismo idioma)
        # Sin acceso al perfil aquí, usamos español por defecto
        language = "es"

        return CompactionContext(
            session_id=session.session_id,
            turns_to_compact=turns_messages,
            max_summary_tokens=max_summary_tokens,
            language=language,
        )

    # =========================================================================
    # ESTIMACIÓN DE TOKENS
    # =========================================================================

    def estimate_context_tokens(
        self,
        session: Session,
        profile: UserProfile,
        user_message: str,
        turns_data: list[dict[str, Any]],
    ) -> dict[str, int]:
        """
        Estima el total de tokens de un contexto sin construirlo.

        Útil para decisiones de compactación sin el overhead de
        construir el contexto completo.

        Args:
            session:      Session activa.
            profile:      Perfil del usuario.
            user_message: Mensaje del usuario.
            turns_data:   Datos del historial.

        Returns:
            Dict con estimaciones por componente y total.
        """
        tool_schemas_estimate = self._tool_registry.estimate_schemas_token_count(
            self._platform,
            profile.permissions,
        )

        system_prompt_est = _MAX_SYSTEM_PROMPT_TOKENS
        profile_est = _MAX_USER_PROFILE_TOKENS
        compaction_est = (
            TokenEstimator.estimate(session.compaction_summary)
            if session.compaction_summary
            else 0
        )

        history_est = sum(
            TokenEstimator.estimate_message(
                t.get("role", "user"),
                t.get("user_message", "") + t.get("assistant_response", ""),
            )
            for t in turns_data
        )

        user_message_est = TokenEstimator.estimate(user_message)

        total = (
            system_prompt_est
            + tool_schemas_estimate
            + profile_est
            + compaction_est
            + history_est
            + user_message_est
        )

        return {
            "system_prompt": system_prompt_est,
            "tool_schemas": tool_schemas_estimate,
            "user_profile": profile_est,
            "compaction_summary": compaction_est,
            "conversation_history": history_est,
            "user_message": user_message_est,
            "total_estimated": total,
            "context_window": session._token_budget.model_context_window,
            "available_for_history": (
                session._token_budget.compute_allocation().available_for_history
            ),
        }

    def calculate_turns_fitting_budget(
        self,
        session: Session,
        turns_data: list[dict[str, Any]],
    ) -> int:
        """
        Calcula cuántos turns del historial caben en el budget disponible.

        Itera desde el más reciente hacia el más antiguo contando tokens.
        Retorna el número de turns que caben sin exceder el budget.

        Args:
            session:    Session activa.
            turns_data: Lista de turns en orden cronológico.

        Returns:
            Número de turns que caben (desde el más reciente).
        """
        # Usar solo el budget base sin tools — profile no está disponible aquí
        allocation = session._token_budget.compute_allocation()

        turns_with_tokens = [
            (i, TokenEstimator.estimate_message(
                "user",
                t.get("user_message", "") + " " + t.get("assistant_response", ""),
            ))
            for i, t in enumerate(turns_data)
        ]

        return session._token_budget.max_turns_from_end(
            turns_with_tokens,
            allocation,
        )

    # =========================================================================
    # CONSTRUCCIÓN DE COMPONENTES INTERNOS
    # =========================================================================

    def _build_history_messages(
        self,
        turns_data: list[dict[str, Any]],
        compaction_summary: str | None,
        budget_for_history: int,
        tool_results_messages: list[LLMMessage] | None = None,
    ) -> list[LLMMessage]:
        """
        Construye la lista de LLMMessages del historial de conversación.

        Incluye el resumen de compactación (si existe) seguido de los
        turns más recientes que caben en el budget.

        Estrategia de selección:
          1. Incluir siempre el resumen de compactación si existe (~500 tokens)
          2. Del budget restante, incluir los N turns más recientes que quepan
          3. Truncar los turns individuales que excedan MAX_SINGLE_TURN_TOKENS

        Args:
            turns_data:             Datos de turns (los más recientes al final).
            compaction_summary:     Resumen de compactación previo si existe.
            budget_for_history:     Tokens disponibles para el historial.
            tool_results_messages:  Mensajes adicionales de resultados de tools.

        Returns:
            Lista ordenada de LLMMessages lista para incluir en el contexto.
        """
        messages: list[LLMMessage] = []
        remaining_budget = budget_for_history

        # 1. Incluir resumen de compactación si existe
        if compaction_summary:
            summary_tokens = TokenEstimator.estimate(compaction_summary)
            if summary_tokens <= min(remaining_budget, _MAX_COMPACTION_SUMMARY_TOKENS * 2):
                summary_message = LLMMessage.system(
                    f"[RESUMEN DE CONVERSACIÓN ANTERIOR]\n{compaction_summary}"
                )
                messages.append(summary_message)
                remaining_budget -= summary_tokens

        # 2. Seleccionar turns que caben en el budget restante
        # Iterar desde el más reciente hacia el más antiguo
        selected_turns: list[dict[str, Any]] = []
        accumulated_tokens = 0

        for turn in reversed(turns_data):
            user_msg = turn.get("user_message", "")
            assistant_msg = turn.get("assistant_response", "")
            combined_text = user_msg + " " + assistant_msg
            turn_tokens = TokenEstimator.estimate_message("user", combined_text)

            # Truncar si el turn individual es demasiado grande
            if turn_tokens > _MAX_SINGLE_TURN_TOKENS:
                # Truncar el mensaje del asistente para reducir tokens
                max_chars = _MAX_SINGLE_TURN_TOKENS * 4  # aprox
                if len(assistant_msg) > max_chars:
                    assistant_msg = assistant_msg[:max_chars] + "...[truncado]"
                    turn = {**turn, "assistant_response": assistant_msg}
                turn_tokens = _MAX_SINGLE_TURN_TOKENS

            if accumulated_tokens + turn_tokens > remaining_budget:
                break

            selected_turns.insert(0, turn)  # insertar al principio para mantener orden
            accumulated_tokens += turn_tokens

        # 3. Construir los mensajes del historial seleccionado
        for turn in selected_turns:
            user_msg = turn.get("user_message", "")
            assistant_msg = turn.get("assistant_response", "")

            if user_msg:
                messages.append(LLMMessage.user(user_msg))
            if assistant_msg:
                messages.append(LLMMessage.assistant(assistant_msg))

        # 4. Añadir resultados de tools si los hay (al final, son los más recientes)
        if tool_results_messages:
            messages.extend(tool_results_messages)

        return messages

    def _build_user_message_content(
        self,
        user_message: str,
        attachments: list[AttachedFile],
    ) -> str | list[TextContent | ImageContent | DocumentContent]:
        """
        Construye el contenido del mensaje del usuario.

        Si hay solo texto, retorna el string directamente.
        Si hay adjuntos, retorna una lista de bloques de contenido multimodal.

        Args:
            user_message: Texto del mensaje del usuario.
            attachments:  Archivos adjuntos preprocesados.

        Returns:
            String para mensaje simple, lista de bloques para multimodal.
        """
        if not attachments:
            return user_message

        # Verificar si hay imágenes entre los adjuntos
        has_images = any(a.is_image for a in attachments)
        has_text_attachments = any(
            not a.is_image and a.has_content for a in attachments
        )

        if not has_images and not has_text_attachments:
            return user_message

        # Construir contenido multimodal
        content_blocks: list[TextContent | ImageContent | DocumentContent] = []

        # Mensaje principal de texto
        if user_message.strip():
            content_blocks.append(TextContent(text=user_message))

        # Documentos de texto adjuntos
        for attachment in attachments:
            if not attachment.is_image and attachment.extracted_text:
                # Truncar si es muy largo
                text = attachment.extracted_text
                max_chars = _MAX_ATTACHMENT_TOKENS * 4
                if len(text) > max_chars:
                    text = text[:max_chars] + f"\n\n[...archivo truncado — {len(attachment.extracted_text)} caracteres totales]"

                content_blocks.append(DocumentContent(
                    title=attachment.filename,
                    text=text,
                    source_filename=attachment.filename,
                ))

        # Imágenes adjuntas
        for attachment in attachments:
            if attachment.is_image and attachment.image_base64:
                from forge_core.llm.protocol import ContentType
                content_blocks.append(ImageContent(
                    type=ContentType.IMAGE_BASE64,
                    data=attachment.image_base64,
                    media_type=attachment.image_media_type or "image/jpeg",
                ))

        # Si solo tenemos texto (sin multimodal real), retornar string simple
        if not has_images and len(content_blocks) == 1 and isinstance(content_blocks[0], TextContent):
            return content_blocks[0].text

        # Combinar texto + documentos como texto enriquecido si no hay imágenes
        if not has_images:
            combined_parts = []
            for block in content_blocks:
                if isinstance(block, TextContent):
                    combined_parts.append(block.text)
                elif isinstance(block, DocumentContent):
                    combined_parts.append(
                        f"\n--- Archivo: {block.title} ---\n{block.text}"
                    )
            return "\n\n".join(combined_parts)

        return content_blocks

    async def _build_artifact_context_messages(
        self,
        artifact_ids: list[str],
        session_id: SessionId,
    ) -> list[LLMMessage]:
        """
        Carga artifacts activos y los convierte en mensajes de contexto.

        Los artifacts relevantes se incluyen en el contexto del LLM para
        que pueda referirse a ellos cuando el usuario los mencione.

        Args:
            artifact_ids: IDs de artifacts a incluir (como strings).
            session_id:   ID de la sesión (para verificación de propiedad).

        Returns:
            Lista de LLMMessages con el contexto de los artifacts.
        """
        from src.domain.value_objects.identifiers import ArtifactId

        messages: list[LLMMessage] = []
        total_tokens = 0

        for artifact_id_str in artifact_ids[:self._max_artifacts]:
            if total_tokens >= _MAX_ARTIFACT_CONTEXT_TOKENS:
                break

            try:
                artifact_id = ArtifactId.from_str(artifact_id_str)
                artifact = await self._artifact_store.load(artifact_id)

                if artifact is None or artifact.session_id != session_id:
                    continue

                # Construir la representación del artifact para el LLM
                artifact_text = artifact.for_llm_context(
                    max_chars=self._artifact_max_tokens * 4,
                    include_metadata=True,
                )

                artifact_tokens = TokenEstimator.estimate(artifact_text)
                if total_tokens + artifact_tokens > _MAX_ARTIFACT_CONTEXT_TOKENS:
                    break

                # Incluir como mensaje de sistema para diferenciarlo del historial
                messages.append(LLMMessage.system(
                    f"[CONTEXTO DE SESIÓN]\n{artifact_text}"
                ))
                total_tokens += artifact_tokens

            except Exception as exc:
                logger.warning(
                    "artifact_contexto_fallo",
                    artifact_id=artifact_id_str,
                    error=str(exc),
                )

        return messages

    def _select_relevant_artifacts(
        self,
        session: Session,
        user_message: str,
    ) -> list[str]:
        """
        Selecciona los IDs de artifacts más relevantes para el contexto actual.

        En V1, incluye simplemente los N artifacts más recientes.
        En V2+, usaría embedding similarity para seleccionar los más relevantes.

        Args:
            session:      Session activa.
            user_message: Mensaje del usuario para selección por relevancia.

        Returns:
            Lista de IDs de artifacts a incluir.
        """
        artifact_refs = session.get_artifact_references()

        # Ordenar por creación descendente (más reciente primero)
        sorted_refs = sorted(
            artifact_refs,
            key=lambda r: r.created_at,
            reverse=True,
        )

        # Filtrar por tipos más útiles para el contexto
        priority_types = {
            "summary", "analysis", "comparison_table",
            "flashcards", "study_plan", "note",
        }

        # Priorizar artifacts de alta utilidad en el contexto
        priority_ids = [
            r.artifact_id.to_str()
            for r in sorted_refs
            if r.artifact_type in priority_types
        ]
        other_ids = [
            r.artifact_id.to_str()
            for r in sorted_refs
            if r.artifact_type not in priority_types
        ]

        selected = (priority_ids + other_ids)[:self._max_artifacts]
        return selected

    def _calculate_safe_history_budget(
        self,
        allocation: BudgetAllocation,
        compaction_summary: str | None,
        user_message: str,
        attachments: list[AttachedFile] | None,
    ) -> int:
        """
        Calcula el budget efectivo disponible para el historial después
        de descontar el resumen de compactación, el mensaje actual y adjuntos.

        Args:
            allocation:         BudgetAllocation calculado.
            compaction_summary: Resumen de compactación si existe.
            user_message:       Mensaje del usuario.
            attachments:        Adjuntos del usuario.

        Returns:
            Tokens disponibles para el historial de conversación.
        """
        budget = allocation.available_for_history

        # Descontar el resumen de compactación
        if compaction_summary:
            budget -= min(
                TokenEstimator.estimate(compaction_summary),
                _MAX_COMPACTION_SUMMARY_TOKENS,
            )

        # Descontar el mensaje actual del usuario
        user_msg_tokens = TokenEstimator.estimate(user_message)
        budget -= min(user_msg_tokens, _MAX_SINGLE_TURN_TOKENS)

        # Descontar adjuntos de texto
        if attachments:
            for attachment in attachments:
                if not attachment.is_image and attachment.extracted_text:
                    attach_tokens = TokenEstimator.estimate(
                        attachment.extracted_text[:_MAX_ATTACHMENT_TOKENS * 4]
                    )
                    budget -= min(attach_tokens, _MAX_ATTACHMENT_TOKENS)

        return max(0, budget)