"""
Entidad Session del dominio HiperForge User.

La Session es el aggregate root central del sistema. Encapsula toda la
conversación del usuario con el agente: el historial de turns, las tareas
ejecutadas, los artifacts producidos, y el perfil de usuario activo.

Una Session no es un simple contenedor de datos — es un aggregate con
reglas de negocio concretas:
  - Las transiciones de estado son explícitas y validadas.
  - Los turns se añaden de forma append-only (nunca se modifican).
  - Los artifacts se asocian a la sesión que los produjo.
  - La sesión conoce su propio estado de "necesita compactación".

Ciclo de vida de una Session:

  CREATED → ACTIVE ──────────────────────────────┐
                │                                 │
                ├──(inactividad > N min)──→ PAUSED│
                │                          │      │
                │                          └──→ ACTIVE
                │                                 │
                └─────────────────────────────────┴──→ CLOSED
                                                        │
                                                     ARCHIVED
                                                  (después de N días)

Principios de diseño:
  1. Session es un aggregate — todas las modificaciones al historial pasan
     por métodos de Session, no directamente a sus listas internas.
  2. El dominio no sabe cómo se persiste — el SessionManager usa el
     SessionStorePort para eso.
  3. Las invariantes se verifican en cada modificación, no solo en la
     construcción. Un aggregate inválido no puede existir.
  4. Los eventos de dominio se acumulan en _domain_events para que el
     bus de eventos los procese después de persistir el estado.
  5. La sesión es la fuente de verdad del contexto conversacional actual.

Relaciones del aggregate:
  Session 1──* ConversationTurn  (historial de la conversación)
  Session 1──* TaskReference     (tareas ejecutadas)
  Session 1──* ArtifactReference (artifacts producidos)
  Session 1──1 UserProfile       (perfil del usuario activo)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Any

from forge_core.errors.types import ForgeDomainError, InvalidStateTransitionError

from src.domain.value_objects.identifiers import (
    ArtifactId,
    SessionId,
    TaskId,
    TurnId,
    UserId,
)
from src.domain.value_objects.token_budget import ContextTokenBudget, TokenUsageSnapshot


# =============================================================================
# STATUS Y TRANSICIONES
# =============================================================================


class SessionStatus(Enum):
    """
    Estado del ciclo de vida de una Session.

    Cada estado define qué operaciones son válidas y qué transiciones
    están permitidas. Las transiciones inválidas lanzan
    InvalidStateTransitionError — nunca silencian el error.
    """

    CREATED = "created"
    """
    Estado inicial al crear la sesión.
    Aún no tiene turns — el primer turn la transiciona a ACTIVE.
    """

    ACTIVE = "active"
    """
    La sesión está activa y en uso. El usuario está interactuando.
    Solo las sesiones ACTIVE pueden recibir nuevos turns y ejecutar tareas.
    """

    PAUSED = "paused"
    """
    La sesión fue pausada por inactividad o por el usuario.
    Puede reanudarse (→ ACTIVE). No acepta nuevos turns mientras está pausada.
    """

    CLOSED = "closed"
    """
    La sesión fue cerrada explícitamente.
    Estado terminal — no puede volver a ACTIVE.
    Se genera un resumen y se archiva después de N días.
    """

    ARCHIVED = "archived"
    """
    La sesión fue archivada después del período de retención.
    Estado terminal — solo lectura para consulta histórica.
    """


# Mapa de transiciones válidas: estado_actual → {estados_siguientes_válidos}
_VALID_TRANSITIONS: dict[SessionStatus, frozenset[SessionStatus]] = {
    SessionStatus.CREATED: frozenset({SessionStatus.ACTIVE}),
    SessionStatus.ACTIVE: frozenset({SessionStatus.PAUSED, SessionStatus.CLOSED}),
    SessionStatus.PAUSED: frozenset({SessionStatus.ACTIVE, SessionStatus.CLOSED}),
    SessionStatus.CLOSED: frozenset({SessionStatus.ARCHIVED}),
    SessionStatus.ARCHIVED: frozenset(),  # estado terminal
}


def _validate_transition(
    from_status: SessionStatus,
    to_status: SessionStatus,
    session_id: SessionId,
) -> None:
    """
    Valida que una transición de estado sea válida según el ciclo de vida.

    Args:
        from_status: Estado actual.
        to_status:   Estado destino.
        session_id:  ID de la sesión (para el mensaje de error).

    Raises:
        InvalidStateTransitionError: Si la transición no está permitida.
    """
    allowed = _VALID_TRANSITIONS.get(from_status, frozenset())
    if to_status not in allowed:
        raise InvalidStateTransitionError(
            entity="Session",
            from_state=from_status.value,
            to_state=to_status.value,
            entity_id=session_id.to_str(),
        )


# =============================================================================
# REFERENCIAS LIGERAS (para el aggregate sin cargar datos completos)
# =============================================================================


@dataclass(frozen=True)
class TurnReference:
    """
    Referencia ligera a un ConversationTurn.

    El aggregate Session no carga los turns completos en memoria —
    almacena solo las referencias con metadata mínima para decisiones
    de compactación y auditoría. Los turns completos se cargan bajo demanda.
    """

    turn_id: TurnId
    """ID único del turn."""

    sequence_number: int
    """Número de secuencia del turn en la sesión (1-indexed)."""

    estimated_tokens: int
    """
    Estimación de tokens del turn (user message + assistant response + tool results).
    Usado para cálculos de budget sin cargar el turn completo.
    """

    created_at: datetime
    """Timestamp de creación del turn (UTC)."""

    has_tool_calls: bool = False
    """True si el turno involucró ejecución de tools."""

    is_compacted: bool = False
    """True si el turn fue incluido en un resumen de compactación."""

    def is_recent(self, within_minutes: int = 60) -> bool:
        """
        Verifica si el turn es reciente.

        Args:
            within_minutes: Ventana de tiempo en minutos.

        Returns:
            True si el turn fue creado hace menos de within_minutes.
        """
        age = datetime.now(tz=timezone.utc) - self.created_at
        return age < timedelta(minutes=within_minutes)


@dataclass(frozen=True)
class TaskReference:
    """
    Referencia ligera a una Task ejecutada en la sesión.

    Como con TurnReference, el aggregate no carga las tareas completas.
    Solo mantiene referencias para auditoría y estadísticas de sesión.
    """

    task_id: TaskId
    """ID único de la tarea."""

    turn_id: TurnId
    """Turn que originó esta tarea."""

    status: str
    """Estado final de la tarea ('completed', 'failed', 'cancelled')."""

    tool_calls_count: int
    """Número de tool calls ejecutados en la tarea."""

    created_at: datetime
    """Timestamp de creación de la tarea (UTC)."""

    completed_at: datetime | None = None
    """Timestamp de finalización de la tarea. None si no completó."""

    @property
    def duration_seconds(self) -> float | None:
        """Duración de la tarea en segundos. None si no completó."""
        if self.completed_at is None:
            return None
        return (self.completed_at - self.created_at).total_seconds()


@dataclass(frozen=True)
class ArtifactReference:
    """
    Referencia a un artifact producido en la sesión.

    Los artifacts son los productos del trabajo del agente: resúmenes,
    reportes, archivos, resultados de búsqueda, etc.
    """

    artifact_id: ArtifactId
    """ID único del artifact."""

    artifact_type: str
    """Tipo de artifact ('summary', 'report', 'search_results', etc.)."""

    display_name: str
    """Nombre legible del artifact para la UI."""

    created_at: datetime
    """Timestamp de creación (UTC)."""

    size_tokens: int = 0
    """Tamaño aproximado del artifact en tokens."""


# =============================================================================
# EVENTOS DE DOMINIO
# =============================================================================


@dataclass(frozen=True)
class SessionEvent:
    """Base de los eventos de dominio de Session."""

    session_id: SessionId
    """ID de la sesión que emitió el evento."""

    occurred_at: datetime
    """Timestamp del evento (UTC)."""


@dataclass(frozen=True)
class SessionActivated(SessionEvent):
    """La sesión fue activada (CREATED→ACTIVE o PAUSED→ACTIVE)."""

    was_restored: bool = False
    """True si la activación fue por restauración desde storage."""


@dataclass(frozen=True)
class SessionPaused(SessionEvent):
    """La sesión fue pausada."""

    reason: str = "inactivity"
    """Razón de la pausa: 'inactivity', 'user_requested'."""


@dataclass(frozen=True)
class SessionClosed(SessionEvent):
    """La sesión fue cerrada."""

    total_turns: int = 0
    """Total de turns al cerrar."""

    total_tasks: int = 0
    """Total de tareas ejecutadas."""


@dataclass(frozen=True)
class TurnAdded(SessionEvent):
    """Un nuevo turn fue añadido a la sesión."""

    turn_id: TurnId
    """ID del turn añadido."""

    estimated_tokens: int = 0
    """Tokens estimados del turn."""


@dataclass(frozen=True)
class CompactionRequired(SessionEvent):
    """El historial de la sesión requiere compactación."""

    turns_to_compact: int = 0
    """Número de turns a compactar."""

    urgency: float = 0.0
    """Urgencia de la compactación [0.0, 1.0]."""


# =============================================================================
# AGGREGATE SESSION
# =============================================================================


class Session:
    """
    Aggregate root de la sesión conversacional.

    Encapsula toda la conversación del usuario: historial de turns,
    tareas ejecutadas, artifacts producidos, y el estado del context window.

    Las invariantes del aggregate son:
    1. Una sesión PAUSED o CLOSED no puede recibir nuevos turns.
    2. El sequence_number de los turns es siempre consecutivo y sin gaps.
    3. La sesión tiene al menos un turn antes de poder pausarse.
    4. El token budget se recalcula con cada nuevo turn.
    5. Los artifacts son inmutables una vez añadidos — no se modifican.
    6. La sesión ARCHIVED no puede ser modificada de ninguna forma.

    Nota sobre el estado mutable:
    A diferencia de los value objects (frozen dataclasses), Session es mutable.
    Es un aggregate que evoluciona a lo largo de su ciclo de vida. La
    inmutabilidad se garantiza a nivel de invariantes y transiciones de estado,
    no a nivel de Python frozen.
    """

    def __init__(
        self,
        session_id: SessionId,
        user_id: UserId,
        token_budget: ContextTokenBudget,
        *,
        created_at: datetime | None = None,
        status: SessionStatus = SessionStatus.CREATED,
    ) -> None:
        """
        Inicializa una nueva sesión.

        El constructor no valida la coherencia completa — usa Session.create()
        o Session.restore() para crear instancias válidas.

        Args:
            session_id:   ID único de la sesión.
            user_id:      ID del usuario propietario de la sesión.
            token_budget: Presupuesto de tokens configurado para el modelo activo.
            created_at:   Timestamp de creación. None = ahora en UTC.
            status:       Estado inicial. Por defecto CREATED.
        """
        self._session_id = session_id
        self._user_id = user_id
        self._token_budget = token_budget
        self._status = status
        self._created_at = created_at or datetime.now(tz=timezone.utc)
        self._last_activity = self._created_at
        self._updated_at = self._created_at

        # Historial de turns (referencias ligeras)
        self._turns: list[TurnReference] = []

        # Tareas ejecutadas en la sesión
        self._tasks: list[TaskReference] = []

        # Artifacts producidos
        self._artifacts: list[ArtifactReference] = []

        # Resumen compactado del historial (si se ha compactado)
        self._compaction_summary: str | None = None
        self._compacted_turns_count: int = 0

        # Tokens actuales del historial (suma de TurnReference.estimated_tokens)
        self._current_history_tokens: int = 0

        # Eventos de dominio acumulados (se vacían después de persistir)
        self._domain_events: list[SessionEvent] = []

    # =========================================================================
    # FACTORIES
    # =========================================================================

    @classmethod
    def create(
        cls,
        user_id: UserId,
        token_budget: ContextTokenBudget,
    ) -> Session:
        """
        Factory: crea una nueva sesión en estado CREATED.

        Este es el punto de entrada correcto para crear una sesión nueva.
        Genera un SessionId único y establece el timestamp de creación.

        Args:
            user_id:      ID del usuario propietario.
            token_budget: Presupuesto de tokens para el modelo activo.

        Returns:
            Nueva Session en estado CREATED.
        """
        return cls(
            session_id=SessionId.generate(),
            user_id=user_id,
            token_budget=token_budget,
            status=SessionStatus.CREATED,
        )

    @classmethod
    def restore(
        cls,
        session_id: SessionId,
        user_id: UserId,
        token_budget: ContextTokenBudget,
        *,
        status: SessionStatus,
        created_at: datetime,
        last_activity: datetime,
        turns: list[TurnReference],
        tasks: list[TaskReference],
        artifacts: list[ArtifactReference],
        compaction_summary: str | None = None,
        compacted_turns_count: int = 0,
        current_history_tokens: int = 0,
    ) -> Session:
        """
        Factory: restaura una sesión desde el storage.

        Se usa en el SessionManager al cargar una sesión persistida.
        No emite eventos de dominio — la restauración no es un evento nuevo.

        Args:
            session_id:             ID de la sesión.
            user_id:                ID del usuario propietario.
            token_budget:           Presupuesto de tokens (puede diferir del original
                                    si cambió el modelo LLM).
            status:                 Estado de la sesión al persistirse.
            created_at:             Timestamp original de creación.
            last_activity:          Timestamp de última actividad.
            turns:                  Lista de TurnReference restauradas.
            tasks:                  Lista de TaskReference restauradas.
            artifacts:              Lista de ArtifactReference restauradas.
            compaction_summary:     Resumen de compactación si existe.
            compacted_turns_count:  Turns incluidos en el resumen.
            current_history_tokens: Tokens del historial al momento de persistir.

        Returns:
            Session restaurada con el estado exacto del storage.
        """
        session = cls(
            session_id=session_id,
            user_id=user_id,
            token_budget=token_budget,
            status=status,
            created_at=created_at,
        )
        session._last_activity = last_activity
        session._updated_at = last_activity
        session._turns = list(turns)
        session._tasks = list(tasks)
        session._artifacts = list(artifacts)
        session._compaction_summary = compaction_summary
        session._compacted_turns_count = compacted_turns_count
        session._current_history_tokens = current_history_tokens

        return session

    # =========================================================================
    # PROPIEDADES (solo lectura)
    # =========================================================================

    @property
    def session_id(self) -> SessionId:
        """ID único de la sesión."""
        return self._session_id

    @property
    def user_id(self) -> UserId:
        """ID del usuario propietario."""
        return self._user_id

    @property
    def status(self) -> SessionStatus:
        """Estado actual de la sesión."""
        return self._status

    @property
    def created_at(self) -> datetime:
        """Timestamp de creación (UTC)."""
        return self._created_at

    @property
    def last_activity(self) -> datetime:
        """Timestamp de la última actividad (UTC)."""
        return self._last_activity

    @property
    def updated_at(self) -> datetime:
        """Timestamp de la última actualización del aggregate (UTC)."""
        return self._updated_at

    @property
    def turn_count(self) -> int:
        """Número de turns en el historial de la sesión."""
        return len(self._turns)

    @property
    def task_count(self) -> int:
        """Número de tareas ejecutadas en la sesión."""
        return len(self._tasks)

    @property
    def artifact_count(self) -> int:
        """Número de artifacts producidos en la sesión."""
        return len(self._artifacts)

    @property
    def current_history_tokens(self) -> int:
        """Tokens totales estimados del historial actual."""
        return self._current_history_tokens

    @property
    def compaction_summary(self) -> str | None:
        """Resumen compactado del historial, si existe."""
        return self._compaction_summary

    @property
    def compacted_turns_count(self) -> int:
        """Número de turns incluidos en el resumen de compactación."""
        return self._compacted_turns_count

    @property
    def is_active(self) -> bool:
        """True si la sesión está en estado ACTIVE."""
        return self._status == SessionStatus.ACTIVE

    @property
    def is_closed(self) -> bool:
        """True si la sesión está en estado CLOSED o ARCHIVED."""
        return self._status in {SessionStatus.CLOSED, SessionStatus.ARCHIVED}

    @property
    def domain_events(self) -> list[SessionEvent]:
        """
        Eventos de dominio acumulados desde la última persistencia.

        El SessionManager los lee después de persistir el aggregate
        para publicarlos en el EventBus.
        """
        return list(self._domain_events)

    # =========================================================================
    # COMANDOS (modificaciones del aggregate)
    # =========================================================================

    def activate(self, *, was_restored: bool = False) -> None:
        """
        Transiciona la sesión a estado ACTIVE.

        Válido desde CREATED y PAUSED.

        Args:
            was_restored: True si la activación es por restauración desde storage.

        Raises:
            InvalidStateTransitionError: Si la sesión no está en CREATED o PAUSED.
        """
        _validate_transition(self._status, SessionStatus.ACTIVE, self._session_id)
        self._status = SessionStatus.ACTIVE
        self._touch()
        self._emit(SessionActivated(
            session_id=self._session_id,
            occurred_at=self._updated_at,
            was_restored=was_restored,
        ))

    def pause(self, *, reason: str = "inactivity") -> None:
        """
        Transiciona la sesión a estado PAUSED.

        Solo válido desde ACTIVE.

        Args:
            reason: Razón de la pausa ('inactivity', 'user_requested').

        Raises:
            InvalidStateTransitionError: Si la sesión no está ACTIVE.
            ForgeDomainError: Si la sesión no tiene turns (no se puede pausar vacía).
        """
        _validate_transition(self._status, SessionStatus.PAUSED, self._session_id)
        if not self._turns:
            raise ForgeDomainError(
                "No se puede pausar una sesión sin ningún turno de conversación.",
                context={"session_id": self._session_id.to_str()},
            )
        self._status = SessionStatus.PAUSED
        self._touch()
        self._emit(SessionPaused(
            session_id=self._session_id,
            occurred_at=self._updated_at,
            reason=reason,
        ))

    def close(self) -> None:
        """
        Transiciona la sesión a estado CLOSED.

        Válido desde ACTIVE y PAUSED. Estado terminal — no puede reactivarse.

        Raises:
            InvalidStateTransitionError: Si la transición no es válida.
        """
        _validate_transition(self._status, SessionStatus.CLOSED, self._session_id)
        self._status = SessionStatus.CLOSED
        self._touch()
        self._emit(SessionClosed(
            session_id=self._session_id,
            occurred_at=self._updated_at,
            total_turns=len(self._turns),
            total_tasks=len(self._tasks),
        ))

    def archive(self) -> None:
        """
        Transiciona la sesión a estado ARCHIVED.

        Solo válido desde CLOSED. Se llama automáticamente después del
        período de retención configurado.

        Raises:
            InvalidStateTransitionError: Si la sesión no está CLOSED.
        """
        _validate_transition(self._status, SessionStatus.ARCHIVED, self._session_id)
        self._status = SessionStatus.ARCHIVED
        self._touch()

    def add_turn(self, turn_ref: TurnReference) -> None:
        """
        Añade un nuevo turn al historial de la sesión.

        Verifica que la sesión esté ACTIVE y que el sequence_number
        del turn sea el siguiente esperado en la secuencia.

        Args:
            turn_ref: Referencia al turn a añadir.

        Raises:
            ForgeDomainError: Si la sesión no está ACTIVE.
            ForgeDomainError: Si el sequence_number es incorrecto.
        """
        if self._status != SessionStatus.ACTIVE:
            raise ForgeDomainError(
                f"No se pueden añadir turns a una sesión en estado '{self._status.value}'. "
                f"La sesión debe estar ACTIVE.",
                context={
                    "session_id": self._session_id.to_str(),
                    "current_status": self._status.value,
                },
            )

        # Validar sequence_number (debe ser consecutivo)
        expected_seq = len(self._turns) + 1
        if turn_ref.sequence_number != expected_seq:
            raise ForgeDomainError(
                f"Sequence number del turn incorrecto: se esperaba {expected_seq}, "
                f"se recibió {turn_ref.sequence_number}.",
                context={
                    "session_id": self._session_id.to_str(),
                    "expected_sequence": expected_seq,
                    "received_sequence": turn_ref.sequence_number,
                    "turn_id": turn_ref.turn_id.to_str(),
                },
            )

        self._turns.append(turn_ref)
        self._current_history_tokens += turn_ref.estimated_tokens
        self._touch()

        self._emit(TurnAdded(
            session_id=self._session_id,
            occurred_at=self._updated_at,
            turn_id=turn_ref.turn_id,
            estimated_tokens=turn_ref.estimated_tokens,
        ))

        # Verificar si se necesita compactación después de añadir el turn
        compaction = self._token_budget.should_compact(
            current_history_tokens=self._current_history_tokens,
            total_turns=len(self._turns),
        )
        if compaction.needs_compaction:
            self._emit(CompactionRequired(
                session_id=self._session_id,
                occurred_at=self._updated_at,
                turns_to_compact=compaction.turns_to_compact,
                urgency=compaction.urgency,
            ))

    def add_task(self, task_ref: TaskReference) -> None:
        """
        Registra una tarea ejecutada en la sesión.

        Args:
            task_ref: Referencia a la tarea a registrar.

        Raises:
            ForgeDomainError: Si la sesión no está ACTIVE.
        """
        if self._status != SessionStatus.ACTIVE:
            raise ForgeDomainError(
                f"No se pueden registrar tareas en una sesión '{self._status.value}'.",
                context={"session_id": self._session_id.to_str()},
            )
        self._tasks.append(task_ref)
        self._touch()

    def add_artifact(self, artifact_ref: ArtifactReference) -> None:
        """
        Registra un artifact producido en la sesión.

        Args:
            artifact_ref: Referencia al artifact a registrar.

        Raises:
            ForgeDomainError: Si la sesión no está ACTIVE.
        """
        if self._status != SessionStatus.ACTIVE:
            raise ForgeDomainError(
                f"No se pueden registrar artifacts en una sesión '{self._status.value}'.",
                context={"session_id": self._session_id.to_str()},
            )
        self._artifacts.append(artifact_ref)
        self._touch()

    def apply_compaction(
        self,
        summary: str,
        compacted_turn_ids: list[TurnId],
        freed_tokens: int,
    ) -> None:
        """
        Aplica el resultado de una compactación al aggregate.

        Después de que el MemoryManager genera el resumen compactado,
        el SessionManager llama a este método para actualizar el estado
        del aggregate: marca los turns como compactados y actualiza el
        token count.

        Args:
            summary:            Resumen generado de los turns compactados.
            compacted_turn_ids: IDs de los turns incluidos en el resumen.
            freed_tokens:       Tokens liberados por la compactación.

        Raises:
            ForgeDomainError: Si la sesión no está ACTIVE.
            ForgeDomainError: Si alguno de los turn_ids no existe en la sesión.
        """
        if self._status != SessionStatus.ACTIVE:
            raise ForgeDomainError(
                "Solo se puede compactar el historial de una sesión ACTIVE.",
                context={"session_id": self._session_id.to_str()},
            )

        # Validar que los turn_ids existen en la sesión
        existing_ids = {t.turn_id for t in self._turns}
        unknown_ids = {tid for tid in compacted_turn_ids if tid not in existing_ids}
        if unknown_ids:
            raise ForgeDomainError(
                f"Se intentó compactar turns que no existen en la sesión: "
                f"{[t.to_str() for t in unknown_ids]}",
                context={"session_id": self._session_id.to_str()},
            )

        # Marcar turns como compactados
        compacted_set = set(compacted_turn_ids)
        self._turns = [
            TurnReference(
                turn_id=t.turn_id,
                sequence_number=t.sequence_number,
                estimated_tokens=0 if t.turn_id in compacted_set else t.estimated_tokens,
                created_at=t.created_at,
                has_tool_calls=t.has_tool_calls,
                is_compacted=t.turn_id in compacted_set,
            )
            for t in self._turns
        ]

        # Actualizar resumen y contadores
        self._compaction_summary = summary
        self._compacted_turns_count = len(compacted_turn_ids)
        self._current_history_tokens = max(
            0,
            self._current_history_tokens - freed_tokens,
        )

        self._touch()

    def clear_domain_events(self) -> list[SessionEvent]:
        """
        Retorna y vacía los eventos de dominio acumulados.

        El SessionManager llama a este método después de persistir el
        aggregate y publicar los eventos en el EventBus.

        Returns:
            Lista de eventos de dominio acumulados desde la última llamada.
        """
        events = list(self._domain_events)
        self._domain_events.clear()
        return events

    # =========================================================================
    # CONSULTAS
    # =========================================================================

    def get_turn_references(self) -> list[TurnReference]:
        """
        Retorna todas las referencias de turns del historial.

        Returns:
            Lista inmutable de TurnReference en orden cronológico.
        """
        return list(self._turns)

    def get_active_turn_refs(self) -> list[TurnReference]:
        """
        Retorna solo los turns que no han sido compactados.

        Returns:
            Lista de TurnReference no compactadas en orden cronológico.
        """
        return [t for t in self._turns if not t.is_compacted]

    def get_artifact_references(self) -> list[ArtifactReference]:
        """
        Retorna todas las referencias de artifacts de la sesión.

        Returns:
            Lista de ArtifactReference en orden de creación.
        """
        return list(self._artifacts)

    def get_task_references(self) -> list[TaskReference]:
        """
        Retorna todas las referencias de tareas de la sesión.

        Returns:
            Lista de TaskReference en orden de creación.
        """
        return list(self._tasks)

    def get_token_usage_snapshot(self) -> TokenUsageSnapshot:
        """
        Crea un snapshot del uso actual de tokens en la sesión.

        El ContextBuilder usa este snapshot para decidir cuántos turns
        incluir en el contexto y si necesita compactar.

        Returns:
            TokenUsageSnapshot con el estado actual de tokens.
        """
        active_turns = [t for t in self._turns if not t.is_compacted]
        return TokenUsageSnapshot(
            history_tokens_used=self._current_history_tokens,
            turns_included=len(active_turns),
            total_turns_in_session=len(self._turns),
            has_compacted_summary=self._compaction_summary is not None,
            compacted_turns_count=self._compacted_turns_count,
        )

    def needs_compaction(self) -> bool:
        """
        Verifica si el historial actual necesita compactación.

        Consulta el ContextTokenBudget con el estado actual de la sesión.

        Returns:
            True si el historial supera el umbral de compactación.
        """
        decision = self._token_budget.should_compact(
            current_history_tokens=self._current_history_tokens,
            total_turns=len(self._turns),
        )
        return decision.needs_compaction

    def is_inactive(self, timeout_minutes: int = 240) -> bool:
        """
        Verifica si la sesión está inactiva más allá del timeout.

        Args:
            timeout_minutes: Minutos de inactividad para considerar la sesión inactiva.

        Returns:
            True si la sesión está ACTIVE pero sin actividad en timeout_minutes.
        """
        if self._status != SessionStatus.ACTIVE:
            return False
        inactive_for = datetime.now(tz=timezone.utc) - self._last_activity
        return inactive_for > timedelta(minutes=timeout_minutes)

    def to_summary(self) -> SessionSummary:
        """
        Crea un SessionSummary con la metadata de alto nivel de la sesión.

        Usado por el SessionManager para el listado de sesiones sin cargar
        el historial completo.

        Returns:
            SessionSummary con metadata de la sesión.
        """
        last_turn_at = self._turns[-1].created_at if self._turns else None
        return SessionSummary(
            session_id=self._session_id,
            user_id=self._user_id,
            status=self._status,
            created_at=self._created_at,
            last_activity=self._last_activity,
            turn_count=len(self._turns),
            task_count=len(self._tasks),
            artifact_count=len(self._artifacts),
            last_turn_at=last_turn_at,
            has_compaction=self._compaction_summary is not None,
        )

    def to_state_dict(self) -> dict[str, Any]:
        """
        Serializa el estado del aggregate para persistencia.

        El SessionStorePort usa este dict para construir las queries SQL.
        No incluye los turns completos — solo las referencias ligeras.

        Returns:
            Diccionario con el estado completo del aggregate.
        """
        return {
            "session_id": self._session_id.to_str(),
            "user_id": self._user_id.to_str(),
            "status": self._status.value,
            "created_at": self._created_at.isoformat(),
            "last_activity": self._last_activity.isoformat(),
            "updated_at": self._updated_at.isoformat(),
            "turn_count": len(self._turns),
            "task_count": len(self._tasks),
            "artifact_count": len(self._artifacts),
            "current_history_tokens": self._current_history_tokens,
            "compaction_summary": self._compaction_summary,
            "compacted_turns_count": self._compacted_turns_count,
        }

    # =========================================================================
    # HELPERS INTERNOS
    # =========================================================================

    def _touch(self) -> None:
        """Actualiza los timestamps de última actividad y última modificación."""
        now = datetime.now(tz=timezone.utc)
        self._last_activity = now
        self._updated_at = now

    def _emit(self, event: SessionEvent) -> None:
        """Acumula un evento de dominio para publicación posterior."""
        self._domain_events.append(event)

    def __repr__(self) -> str:
        return (
            f"Session("
            f"id={self._session_id.to_str()!r}, "
            f"status={self._status.value!r}, "
            f"turns={len(self._turns)}, "
            f"tasks={len(self._tasks)}"
            f")"
        )


# =============================================================================
# SESSION SUMMARY (para listados)
# =============================================================================


@dataclass(frozen=True)
class SessionSummary:
    """
    Resumen de alto nivel de una sesión para listados y navegación.

    Se usa en la UI cuando el usuario quiere ver sus sesiones anteriores.
    No incluye el historial completo — solo metadata para mostrar en un listado.
    """

    session_id: SessionId
    user_id: UserId
    status: SessionStatus
    created_at: datetime
    last_activity: datetime
    turn_count: int
    task_count: int
    artifact_count: int
    last_turn_at: datetime | None
    has_compaction: bool = False

    @property
    def duration_minutes(self) -> float:
        """Duración de la sesión desde creación hasta última actividad, en minutos."""
        return (self.last_activity - self.created_at).total_seconds() / 60

    @property
    def is_empty(self) -> bool:
        """True si la sesión no tiene ningún turn."""
        return self.turn_count == 0

    def to_display_dict(self) -> dict[str, Any]:
        """Serializa el resumen para la UI."""
        return {
            "session_id": self.session_id.to_str(),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "duration_minutes": round(self.duration_minutes, 1),
            "turn_count": self.turn_count,
            "task_count": self.task_count,
            "artifact_count": self.artifact_count,
            "is_empty": self.is_empty,
        }