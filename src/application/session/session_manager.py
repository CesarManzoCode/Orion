"""
SessionManager — gestor del ciclo de vida de sesiones conversacionales.

El SessionManager es el servicio de aplicación responsable de todo lo relacionado
con las sesiones: crearlas, restaurarlas, pausarlas por inactividad, cerrarlas,
y coordinar la compactación del historial cuando el context window se agota.

Responsabilidades:
  1. Crear nuevas sesiones con el budget de tokens correcto para el modelo activo
  2. Restaurar sesiones desde el storage al arrancar la app
  3. Persistir el estado del aggregate Session después de cada modificación
  4. Detectar sesiones inactivas y pausarlas automáticamente
  5. Coordinar la compactación de historial con el MemoryManager
  6. Publicar eventos de dominio de la sesión al EventBus (cuando exista)
  7. Gestionar el estado del UserProfile al cerrar sesiones

Diseño:
  - Stateless entre llamadas: todo el estado vive en el aggregate Session
    y en el storage. El SessionManager no cachea sesiones en memoria.
  - Async-safe: operaciones con la misma sesión usan el lock del storage.
  - Fail-safe: errores de storage producen respuestas degradadas, no crashes.
  - El SessionManager coordina pero no contiene lógica de negocio —
    las reglas están en los aggregates del dominio y en los value objects.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from forge_core.errors.types import ForgeDomainError, ForgeStorageError
from forge_core.observability.logging import get_logger

from src.domain.entities.session import (
    ArtifactReference,
    Session,
    SessionStatus,
    SessionSummary,
    TaskReference,
    TurnReference,
)
from src.domain.entities.user_profile import UserProfile
from src.domain.value_objects.identifiers import (
    ArtifactId,
    SessionId,
    TurnId,
    UserId,
)
from src.domain.value_objects.token_budget import (
    BUDGET_128K,
    ContextTokenBudget,
    TokenEstimator,
)
from src.ports.outbound.storage_port import (
    AuditLogPort,
    SessionQuery,
    SessionStorePort,
    UserProfileStorePort,
)


logger = get_logger(__name__, component="session_manager")


class SessionManager:
    """
    Gestor del ciclo de vida de sesiones conversacionales.

    Coordina la creación, restauración, persistencia y cierre de sesiones.
    Es el único punto de acceso a las sesiones desde la capa de aplicación
    — ningún otro servicio accede al SessionStorePort directamente.

    El SessionManager NO mantiene sesiones en caché — cada operación carga
    el aggregate desde el storage. Esto garantiza que el estado siempre sea
    consistente incluso si múltiples coroutines acceden a la misma sesión.

    Nota sobre concurrencia: en V1 hay exactamente una sesión activa a la vez.
    Los locks son por session_id para el futuro multi-sesión de V3.
    """

    def __init__(
        self,
        session_store: SessionStorePort,
        profile_store: UserProfileStorePort,
        audit_log: AuditLogPort,
        *,
        default_token_budget: ContextTokenBudget = BUDGET_128K,
        inactivity_timeout_minutes: int = 240,
        auto_pause_enabled: bool = True,
    ) -> None:
        """
        Inicializa el SessionManager.

        Args:
            session_store:              Puerto de persistencia de sesiones.
            profile_store:              Puerto de persistencia de perfiles.
            audit_log:                  Puerto del audit log.
            default_token_budget:       Budget de tokens por defecto.
            inactivity_timeout_minutes: Minutos de inactividad para auto-pause.
            auto_pause_enabled:         Habilitar pausa automática por inactividad.
        """
        self._session_store = session_store
        self._profile_store = profile_store
        self._audit_log = audit_log
        self._default_budget = default_token_budget
        self._inactivity_timeout = inactivity_timeout_minutes
        self._auto_pause_enabled = auto_pause_enabled

        # Locks por session_id para operaciones concurrentes
        self._session_locks: dict[str, asyncio.Lock] = {}

    # =========================================================================
    # CICLO DE VIDA DE SESIONES
    # =========================================================================

    async def create_session(
        self,
        user_id: UserId,
        *,
        token_budget: ContextTokenBudget | None = None,
    ) -> Session:
        """
        Crea una nueva sesión conversacional para el usuario.

        Genera un SessionId único, crea el aggregate Session, lo activa,
        y lo persiste en el storage. También actualiza el contador de sesiones
        del UserProfile.

        Args:
            user_id:      ID del usuario propietario de la sesión.
            token_budget: Budget de tokens a usar. None usa el default configurado.

        Returns:
            Nuevo aggregate Session en estado ACTIVE.

        Raises:
            ForgeStorageError: Si no se puede persistir la sesión.
        """
        budget = token_budget or self._default_budget

        session = Session.create(
            user_id=user_id,
            token_budget=budget,
        )
        session.activate()

        try:
            await self._session_store.save_session(session)
            await self._profile_store.increment_session_count(user_id)
        except ForgeStorageError as e:
            logger.error(
                "sesion_creacion_fallo_storage",
                session_id=session.session_id.to_str(),
                error=e,
            )
            raise

        # Publicar eventos de dominio
        events = session.clear_domain_events()
        self._log_domain_events(events, session.session_id)

        logger.info(
            "sesion_creada",
            session_id=session.session_id.to_str(),
            user_id=user_id.to_str(),
            budget_tokens=budget.model_context_window,
        )

        return session

    async def get_session(self, session_id: SessionId) -> Session | None:
        """
        Carga una sesión desde el storage por su ID.

        Args:
            session_id: ID de la sesión a cargar.

        Returns:
            Aggregate Session restaurado, o None si no existe.
        """
        try:
            return await self._session_store.load_session(session_id)
        except ForgeStorageError as e:
            logger.error(
                "sesion_carga_fallo",
                session_id=session_id.to_str(),
                error=e,
            )
            return None

    async def get_or_create_session(
        self,
        session_id: SessionId,
        user_id: UserId,
    ) -> Session:
        """
        Obtiene la sesión especificada o crea una nueva si no existe.

        Útil en el ConversationCoordinator cuando el ID de sesión puede
        corresponder a una sesión existente o a una nueva.

        Args:
            session_id: ID de la sesión a obtener.
            user_id:    ID del usuario (para crear si no existe).

        Returns:
            Session existente o nueva en estado ACTIVE.
        """
        session = await self.get_session(session_id)
        if session is not None:
            if not session.is_active:
                await self.activate_session(session)
            return session

        # La sesión no existe, crear una nueva
        logger.warning(
            "sesion_no_encontrada_creando_nueva",
            session_id=session_id.to_str(),
            user_id=user_id.to_str(),
        )
        return await self.create_session(user_id)

    async def activate_session(self, session: Session) -> Session:
        """
        Activa una sesión que estaba en estado CREATED o PAUSED.

        Args:
            session: Session a activar.

        Returns:
            Session activada y persistida.

        Raises:
            ForgeDomainError:   Si la transición de estado no es válida.
            ForgeStorageError:  Si no se puede persistir.
        """
        was_paused = session.status == SessionStatus.PAUSED

        session.activate(was_restored=was_paused)

        await self._session_store.save_session(session)

        events = session.clear_domain_events()
        self._log_domain_events(events, session.session_id)

        logger.info(
            "sesion_activada",
            session_id=session.session_id.to_str(),
            was_restored=was_paused,
        )

        return session

    async def pause_session(
        self,
        session: Session,
        *,
        reason: str = "inactivity",
    ) -> Session:
        """
        Pausa una sesión activa.

        Args:
            session: Session a pausar.
            reason:  Razón de la pausa ('inactivity', 'user_requested').

        Returns:
            Session pausada y persistida.
        """
        if not session.is_active:
            logger.debug(
                "sesion_pausa_skip",
                session_id=session.session_id.to_str(),
                status=session.status.value,
            )
            return session

        if session.turn_count == 0:
            logger.debug(
                "sesion_pausa_sin_turns_skip",
                session_id=session.session_id.to_str(),
            )
            return session

        try:
            session.pause(reason=reason)
            await self._session_store.save_session(session)

            events = session.clear_domain_events()
            self._log_domain_events(events, session.session_id)

            logger.info(
                "sesion_pausada",
                session_id=session.session_id.to_str(),
                reason=reason,
                turns=session.turn_count,
            )

        except ForgeDomainError as e:
            # La sesión no estaba en estado pausable — ignorar silenciosamente
            logger.debug("sesion_pausa_skip_domain", error=e)

        return session

    async def close_session(
        self,
        session: Session,
        profile: UserProfile | None = None,
    ) -> Session:
        """
        Cierra una sesión y actualiza las estadísticas del usuario.

        Args:
            session: Session a cerrar.
            profile: UserProfile del usuario (para actualizar estadísticas).

        Returns:
            Session cerrada y persistida.
        """
        if session.is_closed:
            return session

        turn_count = session.turn_count

        try:
            session.close()
            await self._session_store.save_session(session)

            # Actualizar estadísticas del perfil de usuario
            if profile is not None:
                await self._profile_store.add_turns_to_total(
                    session.user_id,
                    turn_count,
                )

            events = session.clear_domain_events()
            self._log_domain_events(events, session.session_id)

            logger.info(
                "sesion_cerrada",
                session_id=session.session_id.to_str(),
                total_turns=turn_count,
                total_tasks=session.task_count,
                total_artifacts=session.artifact_count,
            )

        except ForgeDomainError as e:
            logger.warning(
                "sesion_cierre_domain_error",
                session_id=session.session_id.to_str(),
                error=e,
            )

        return session

    async def get_last_active_session(
        self,
        user_id: UserId,
    ) -> Session | None:
        """
        Retorna la última sesión activa del usuario.

        Se usa al arrancar la app para restaurar la sesión anterior.

        Args:
            user_id: ID del usuario.

        Returns:
            La última Session activa, o None si no hay ninguna.
        """
        try:
            session_id = await self._session_store.get_last_active_session(user_id)
            if session_id is None:
                return None
            return await self._session_store.load_session(session_id)
        except ForgeStorageError as e:
            logger.error("ultima_sesion_activa_fallo", user_id=user_id.to_str(), error=e)
            return None

    # =========================================================================
    # GESTIÓN DE TURNS Y EVIDENCIA
    # =========================================================================

    async def persist_turn(
        self,
        session: Session,
        *,
        user_message: str,
        assistant_response: str,
        routing_type: str,
        tool_calls_count: int = 0,
        has_tool_calls: bool = False,
    ) -> TurnReference:
        """
        Persiste un nuevo turn en la sesión y actualiza el aggregate.

        Calcula la estimación de tokens del turn, crea la TurnReference,
        actualiza el aggregate Session, y persiste tanto el aggregate como
        el contenido completo del turn.

        Args:
            session:           Session activa.
            user_message:      Mensaje del usuario.
            assistant_response: Respuesta del asistente.
            routing_type:      Tipo de routing usado ('DirectReply', etc.).
            tool_calls_count:  Número de tool calls ejecutados.
            has_tool_calls:    True si hubo tool calls en el turno.

        Returns:
            TurnReference del turn persistido.

        Raises:
            ForgeStorageError: Si no se puede persistir.
        """
        from src.domain.value_objects.identifiers import TurnId

        turn_id = TurnId.generate()
        sequence_number = session.turn_count + 1

        # Estimar tokens del turn completo
        combined_text = user_message + " " + assistant_response
        estimated_tokens = TokenEstimator.estimate(combined_text)

        turn_ref = TurnReference(
            turn_id=turn_id,
            sequence_number=sequence_number,
            estimated_tokens=estimated_tokens,
            created_at=datetime.now(tz=timezone.utc),
            has_tool_calls=has_tool_calls,
        )

        turn_content = {
            "turn_id": turn_id.to_str(),
            "sequence_number": sequence_number,
            "user_message": user_message,
            "assistant_response": assistant_response,
            "routing_type": routing_type,
            "tool_calls_count": tool_calls_count,
            "estimated_tokens": estimated_tokens,
            "created_at": turn_ref.created_at.isoformat(),
        }

        # Actualizar el aggregate primero (valida invariantes de dominio)
        session.add_turn(turn_ref)

        # Luego persistir
        try:
            await self._session_store.save_session(session)
            await self._session_store.append_turn(
                session.session_id,
                turn_ref,
                turn_content,
            )
        except ForgeStorageError as e:
            logger.error(
                "persistencia_turn_fallo",
                session_id=session.session_id.to_str(),
                turn_id=turn_id.to_str(),
                error=e,
            )
            raise

        # Verificar si se necesita compactación
        compaction_decision = session._token_budget.should_compact(
            current_history_tokens=session.current_history_tokens,
            total_turns=session.turn_count,
        )

        if compaction_decision.needs_compaction:
            logger.info(
                "compactacion_requerida",
                session_id=session.session_id.to_str(),
                turns_to_compact=compaction_decision.turns_to_compact,
                urgency=round(compaction_decision.urgency, 2),
            )

        # Publicar eventos de dominio
        events = session.clear_domain_events()
        self._log_domain_events(events, session.session_id)

        return turn_ref

    async def register_task(
        self,
        session: Session,
        task_ref: TaskReference,
    ) -> None:
        """
        Registra una tarea completada en la sesión.

        Args:
            session:  Session activa.
            task_ref: Referencia de la tarea a registrar.
        """
        session.add_task(task_ref)
        try:
            await self._session_store.save_session(session)
            await self._session_store.append_task_reference(
                session.session_id,
                task_ref,
            )
        except ForgeStorageError as e:
            logger.error("registro_tarea_fallo", error=e)

    async def register_artifact(
        self,
        session: Session,
        artifact_ref: ArtifactReference,
    ) -> None:
        """
        Registra un artifact producido en la sesión.

        Args:
            session:       Session activa.
            artifact_ref:  Referencia del artifact a registrar.
        """
        session.add_artifact(artifact_ref)
        try:
            await self._session_store.save_session(session)
            await self._session_store.append_artifact_reference(
                session.session_id,
                artifact_ref,
            )
        except ForgeStorageError as e:
            logger.error("registro_artifact_fallo", error=e)

    # =========================================================================
    # CARGA DE HISTORIAL PARA EL CONTEXTO LLM
    # =========================================================================

    async def load_turns_for_context(
        self,
        session: Session,
        *,
        max_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Carga los turns más recientes que caben en el budget del context window.

        Retorna el historial en orden cronológico, comenzando por el más
        antiguo que aún cabe dentro del budget especificado.

        Args:
            session:    Session activa.
            max_tokens: Budget máximo de tokens. None usa el disponible del session budget.

        Returns:
            Lista de dicts de turns en orden cronológico (más antiguo primero).
        """
        budget = max_tokens
        if budget is None:
            allocation = session._token_budget.compute_allocation()
            budget = allocation.available_for_history

        try:
            return await self._session_store.load_recent_turns(
                session.session_id,
                max_tokens=budget,
            )
        except ForgeStorageError as e:
            logger.error(
                "carga_turns_fallo",
                session_id=session.session_id.to_str(),
                error=e,
            )
            return []

    async def load_all_turns_for_compaction(
        self,
        session: Session,
        turns_count: int,
    ) -> list[dict[str, Any]]:
        """
        Carga los turns más antiguos para compactación.

        A diferencia de load_turns_for_context (que carga los más recientes),
        este método carga los más antiguos que se van a compactar.

        Args:
            session:     Session activa.
            turns_count: Número de turns a cargar desde el más antiguo.

        Returns:
            Lista de dicts de turns a compactar, en orden cronológico.
        """
        try:
            return await self._session_store.load_turns(
                session.session_id,
                limit=turns_count,
                offset=0,
                order="asc",
            )
        except ForgeStorageError as e:
            logger.error("carga_turns_compactacion_fallo", error=e)
            return []

    # =========================================================================
    # COMPACTACIÓN DE HISTORIAL
    # =========================================================================

    async def apply_compaction(
        self,
        session: Session,
        *,
        compaction_summary: str,
        compacted_turn_ids: list[TurnId],
        freed_tokens: int,
    ) -> None:
        """
        Aplica el resultado de la compactación al aggregate y lo persiste.

        Se llama después de que el LLM generó el resumen de compactación
        para actualizar el estado del historial.

        Args:
            session:             Session activa.
            compaction_summary:  Resumen generado.
            compacted_turn_ids:  IDs de turns compactados.
            freed_tokens:        Tokens estimados liberados.

        Raises:
            ForgeDomainError:   Si la compactación no es válida para la sesión.
            ForgeStorageError:  Si no se puede persistir.
        """
        # Actualizar el aggregate (valida que los turn_ids existen)
        session.apply_compaction(
            compaction_summary=compaction_summary,
            compacted_turn_ids=compacted_turn_ids,
            freed_tokens=freed_tokens,
        )

        try:
            await self._session_store.update_compaction_state(
                session.session_id,
                compaction_summary=compaction_summary,
                compacted_turn_ids=compacted_turn_ids,
                freed_tokens=freed_tokens,
            )
            await self._session_store.save_session(session)
        except ForgeStorageError as e:
            logger.error(
                "compactacion_persistencia_fallo",
                session_id=session.session_id.to_str(),
                error=e,
            )
            raise

        logger.info(
            "compactacion_aplicada",
            session_id=session.session_id.to_str(),
            turns_compacted=len(compacted_turn_ids),
            freed_tokens=freed_tokens,
            summary_length=len(compaction_summary),
        )

    # =========================================================================
    # MANTENIMIENTO Y LISTADO
    # =========================================================================

    async def list_sessions(
        self,
        user_id: UserId | None = None,
        *,
        status: SessionStatus | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[SessionSummary]:
        """
        Lista sesiones del usuario según criterios de filtrado.

        Args:
            user_id: ID del usuario. None lista todas (admin).
            status:  Filtrar por estado. None lista todas.
            limit:   Máximo de resultados.
            offset:  Desplazamiento para paginación.

        Returns:
            Lista de SessionSummary ordenada por última actividad descendente.
        """
        query = SessionQuery(
            user_id=user_id,
            status=status,
            order_by="last_activity",
            order_dir="desc",
            limit=limit,
            offset=offset,
        )

        try:
            result = await self._session_store.list_sessions(query)
            return result.items
        except ForgeStorageError as e:
            logger.error("listado_sesiones_fallo", error=e)
            return []

    async def detect_and_pause_inactive_sessions(
        self,
        user_id: UserId,
    ) -> list[SessionId]:
        """
        Detecta sesiones inactivas y las pausa automáticamente.

        Se llama periódicamente por el sistema (ej: cada 15 minutos)
        para limpiar sesiones que quedaron activas sin uso.

        Args:
            user_id: ID del usuario cuyas sesiones verificar.

        Returns:
            Lista de SessionIds de sesiones que fueron pausadas.
        """
        if not self._auto_pause_enabled:
            return []

        paused_sessions: list[SessionId] = []

        try:
            active_sessions = await self.list_sessions(
                user_id,
                status=SessionStatus.ACTIVE,
                limit=10,
            )

            for summary in active_sessions:
                session = await self._session_store.load_session(summary.session_id)
                if session is None:
                    continue

                if session.is_inactive(timeout_minutes=self._inactivity_timeout):
                    await self.pause_session(session, reason="inactivity")
                    paused_sessions.append(summary.session_id)

                    logger.info(
                        "sesion_pausada_por_inactividad",
                        session_id=summary.session_id.to_str(),
                        last_activity=summary.last_activity.isoformat(),
                    )

        except ForgeStorageError as e:
            logger.error("deteccion_inactividad_fallo", user_id=user_id.to_str(), error=e)

        return paused_sessions

    async def cleanup_old_sessions(
        self,
        user_id: UserId,
        *,
        retention_days: int = 90,
    ) -> int:
        """
        Archiva o elimina sesiones cerradas más antiguas que retention_days.

        Args:
            user_id:        ID del usuario.
            retention_days: Días de retención para sesiones cerradas.

        Returns:
            Número de sesiones archivadas.
        """
        archived_count = 0
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=retention_days)

        try:
            closed_sessions = await self.list_sessions(
                user_id,
                status=SessionStatus.CLOSED,
                limit=50,
            )

            for summary in closed_sessions:
                if summary.last_activity < cutoff:
                    session = await self._session_store.load_session(summary.session_id)
                    if session is not None and not session.is_closed:
                        continue

                    if session is not None:
                        try:
                            session.archive()
                            await self._session_store.save_session(session)
                            archived_count += 1
                        except ForgeDomainError:
                            pass  # ya archivada o en estado inválido

        except ForgeStorageError as e:
            logger.error("limpieza_sesiones_fallo", error=e)

        if archived_count > 0:
            logger.info(
                "sesiones_archivadas",
                user_id=user_id.to_str(),
                archived_count=archived_count,
            )

        return archived_count

    # =========================================================================
    # HELPERS INTERNOS
    # =========================================================================

    def _get_session_lock(self, session_id: SessionId) -> asyncio.Lock:
        """Obtiene o crea el lock de una sesión específica."""
        sid_str = session_id.to_str()
        if sid_str not in self._session_locks:
            self._session_locks[sid_str] = asyncio.Lock()
        return self._session_locks[sid_str]

    def _log_domain_events(
        self,
        events: list[Any],
        session_id: SessionId,
    ) -> None:
        """
        Loguea los eventos de dominio de la sesión.

        En V2, esto publicará los eventos en el EventBus para que
        otros componentes puedan reaccionar (ej: notificaciones).
        En V1, solo se loguean para observabilidad.
        """
        for event in events:
            event_name = type(event).__name__
            logger.debug(
                f"evento_dominio_{event_name.lower()}",
                session_id=session_id.to_str(),
                event_type=event_name,
            )

    async def get_session_stats(
        self,
        session_id: SessionId,
    ) -> dict[str, Any]:
        """
        Retorna estadísticas de una sesión para debugging y métricas.

        Args:
            session_id: ID de la sesión.

        Returns:
            Dict con estadísticas de la sesión.
        """
        session = await self.get_session(session_id)
        if session is None:
            return {"error": "Sesión no encontrada."}

        allocation = session._token_budget.compute_allocation()
        snapshot = session.get_token_usage_snapshot()

        return {
            "session_id": session_id.to_str(),
            "status": session.status.value,
            "turn_count": session.turn_count,
            "task_count": session.task_count,
            "artifact_count": session.artifact_count,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "token_budget": {
                "total_context_window": allocation.total_context_window,
                "available_for_history": allocation.available_for_history,
                "current_history_tokens": session.current_history_tokens,
                "usage_ratio": round(
                    allocation.usage_ratio(session.current_history_tokens), 3
                ),
            },
            "compaction": {
                "has_summary": session.compaction_summary is not None,
                "compacted_turns": session.compacted_turns_count,
                "needs_compaction": session.needs_compaction(),
            },
            "is_inactive": session.is_inactive(
                timeout_minutes=self._inactivity_timeout
            ),
        }