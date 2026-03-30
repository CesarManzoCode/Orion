"""
Puertos de persistencia de HiperForge User.

Este módulo define los contratos de acceso a datos para todos los subsistemas
de persistencia del producto. En V1, todas las implementaciones usan SQLite
local. Los puertos están diseñados para que la migración a PostgreSQL (V3
multi-usuario) no requiera cambios en la capa de aplicación.

Puertos definidos:

  SessionStorePort     — sesiones y turns de conversación
  ArtifactStorePort    — artifacts producidos por el agente
  AuditLogPort         — registro inmutable de acciones (append-only)
  UserProfileStorePort — perfil del usuario y permisos

Principios de diseño:
  1. Todos los métodos son async — ninguna operación de I/O bloquea.
  2. Los puertos operan con tipos del dominio, no con dicts crudos de SQL.
     Los adapters hacen el mapeo ORM↔dominio internamente.
  3. Los errores se convierten en ForgeStorageError antes de propagarse.
  4. El AuditLog es estrictamente append-only — sin UPDATE ni DELETE.
  5. Los puertos incluyen paginación en todas las operaciones de listado
     para prevenir cargas masivas de memoria.
  6. Operaciones de escritura usan transacciones implícitas — o todo
     se persiste o nada se persiste.

Convenciones de paginación:
  - limit: máximo de resultados a retornar (default: 50)
  - offset: desplazamiento desde el inicio del resultado (default: 0)
  - order: 'asc' o 'desc' por timestamp de creación

Jerarquía de tipos:

  PaginatedResult[T]      — wrapper de resultados paginados
  SessionQuery            — criterios de búsqueda de sesiones
  ArtifactQuery           — criterios de búsqueda de artifacts
  AuditQuery              — criterios de búsqueda en audit log
  AuditEntry              — entrada individual del audit log
  SessionStorePort        — persistencia de sesiones y turns
  ArtifactStorePort       — persistencia de artifacts
  AuditLogPort            — audit log append-only
  UserProfileStorePort    — persistencia del perfil de usuario
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar

from forge_core.errors.types import ForgeStorageError  # noqa: F401 — re-exportado

from src.domain.entities.artifact import Artifact, ArtifactSummary, ArtifactType
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


# =============================================================================
# TIPOS GENÉRICOS Y DE CONSULTA
# =============================================================================

T = TypeVar("T")


@dataclass(frozen=True)
class PaginatedResult(Generic[T]):
    """
    Resultado paginado genérico para operaciones de listado.

    Wrap estándar para cualquier operación que pueda retornar múltiples
    resultados. Incluye metadata de paginación para que la UI pueda
    implementar carga progresiva o navegación por páginas.
    """

    items: list[T]
    """Items de la página actual."""

    total_count: int
    """Total de items que cumplen los criterios (sin paginación)."""

    limit: int
    """Tamaño máximo de página solicitado."""

    offset: int
    """Desplazamiento desde el inicio."""

    @property
    def has_more(self) -> bool:
        """True si hay más items después de esta página."""
        return (self.offset + len(self.items)) < self.total_count

    @property
    def page_count(self) -> int:
        """Número total de páginas con el limit actual."""
        if self.limit <= 0:
            return 1
        return max(1, (self.total_count + self.limit - 1) // self.limit)

    @property
    def current_page(self) -> int:
        """Número de página actual (1-indexed)."""
        if self.limit <= 0:
            return 1
        return (self.offset // self.limit) + 1


@dataclass(frozen=True)
class SessionQuery:
    """
    Criterios de búsqueda y filtrado de sesiones.

    Todos los campos son opcionales — un SessionQuery vacío retorna todas
    las sesiones del usuario ordenadas por última actividad descendente.
    """

    user_id: UserId | None = None
    """Filtrar por usuario (requerido en V3 multi-usuario)."""

    status: SessionStatus | None = None
    """Filtrar por estado. None retorna todas."""

    created_after: datetime | None = None
    """Solo sesiones creadas después de esta fecha."""

    created_before: datetime | None = None
    """Solo sesiones creadas antes de esta fecha."""

    has_artifacts: bool | None = None
    """True para sesiones con artifacts, False para las que no tienen."""

    min_turn_count: int | None = None
    """Solo sesiones con al menos este número de turns."""

    order_by: str = "last_activity"
    """Campo de ordenación: 'last_activity', 'created_at', 'turn_count'."""

    order_dir: str = "desc"
    """Dirección de ordenación: 'asc' o 'desc'."""

    limit: int = 20
    """Máximo de sesiones a retornar."""

    offset: int = 0
    """Desplazamiento para paginación."""


@dataclass(frozen=True)
class ArtifactQuery:
    """
    Criterios de búsqueda y filtrado de artifacts.
    """

    session_id: SessionId | None = None
    """Filtrar por sesión. None busca en todas las sesiones del usuario."""

    artifact_type: ArtifactType | None = None
    """Filtrar por tipo de artifact."""

    tags: list[str] = field(default_factory=list)
    """
    Filtrar por tags. Un artifact debe tener TODOS los tags especificados.
    Lista vacía no filtra por tags.
    """

    source_filename: str | None = None
    """Filtrar por nombre de archivo origen."""

    text_search: str | None = None
    """
    Búsqueda de texto libre en display_name y content_summary.
    Implementado como LIKE query en SQLite (no full-text search en V1).
    """

    created_after: datetime | None = None
    """Solo artifacts creados después de esta fecha."""

    created_before: datetime | None = None
    """Solo artifacts creados antes de esta fecha."""

    order_by: str = "created_at"
    """Campo de ordenación: 'created_at', 'display_name', 'artifact_type'."""

    order_dir: str = "desc"
    """Dirección de ordenación."""

    limit: int = 50
    """Máximo de artifacts a retornar."""

    offset: int = 0
    """Desplazamiento para paginación."""


@dataclass(frozen=True)
class AuditQuery:
    """
    Criterios de búsqueda en el audit log.
    """

    session_id: SessionId | None = None
    """Filtrar por sesión."""

    tool_id: str | None = None
    """Filtrar por tool ejecutada."""

    risk_level: str | None = None
    """Filtrar por nivel de riesgo ('none', 'low', 'medium', 'high', 'critical')."""

    policy_decision: str | None = None
    """Filtrar por decisión de policy ('allow', 'deny', 'require_approval')."""

    success: bool | None = None
    """True para éxitos, False para fallos, None para todos."""

    is_security_event: bool | None = None
    """True para filtrar solo eventos de seguridad."""

    occurred_after: datetime | None = None
    """Solo entradas después de esta fecha."""

    occurred_before: datetime | None = None
    """Solo entradas antes de esta fecha."""

    limit: int = 100
    """Máximo de entradas a retornar."""

    offset: int = 0
    """Desplazamiento para paginación."""


@dataclass(frozen=True)
class AuditEntry:
    """
    Entrada del audit log inmutable.

    Representa una acción ejecutada por el agente con todos sus metadatos
    de seguridad. Una vez creada, nunca se modifica — el audit log es
    append-only por diseño.

    Diseño de los campos:
    - Los campos de identificación (session_id, tool_id) son siempre strings
      para facilitar la serialización y evitar dependencias de tipos del dominio.
    - input_summary y output_summary son resúmenes SANITIZADOS — nunca
      contienen el contenido completo ni datos sensibles.
    - Los campos booleanos (success, sandbox_used, is_security_event) permiten
      queries rápidas sin parsear strings.
    """

    entry_id: str
    """ID único de la entrada (ULID sin tipo)."""

    occurred_at: datetime
    """Timestamp del evento (UTC)."""

    session_id: str
    """ID de la sesión en la que ocurrió."""

    event_type: str
    """
    Tipo de evento:
    'tool_execution' | 'policy_decision' | 'approval' |
    'security_event' | 'session_lifecycle' | 'compaction'
    """

    # --- Campos de ejecución de tool (si event_type='tool_execution') ---
    tool_id: str | None = None
    """ID de la tool ejecutada."""

    risk_level: str | None = None
    """Nivel de riesgo efectivo de la acción."""

    policy_decision: str | None = None
    """Decisión del policy engine: 'allow', 'deny', 'require_approval'."""

    success: bool | None = None
    """True si la ejecución fue exitosa."""

    duration_ms: float | None = None
    """Duración de la ejecución en milisegundos."""

    sandbox_used: bool = False
    """True si se usó el proceso sandbox."""

    approval_id: str | None = None
    """ID de la aprobación si fue requerida."""

    # --- Resúmenes sanitizados ---
    input_summary: str | None = None
    """Resumen sanitizado del input (sin paths completos sensibles)."""

    output_summary: str | None = None
    """Resumen sanitizado del output."""

    error_code: str | None = None
    """Código de error ForgeError si la ejecución falló."""

    # --- Metadata de seguridad ---
    is_security_event: bool = False
    """True si esta entrada es un evento de seguridad (violación, bloqueo)."""

    policy_name: str | None = None
    """Nombre de la policy que tomó la decisión."""

    extra_context: dict[str, Any] = field(default_factory=dict)
    """
    Contexto adicional específico del event_type.
    No se indexa — solo para debugging y análisis.
    """

    def to_display_dict(self) -> dict[str, Any]:
        """Serializa la entrada para el debug CLI y la UI de admin."""
        return {
            "entry_id": self.entry_id,
            "occurred_at": self.occurred_at.isoformat(),
            "session_id": self.session_id,
            "event_type": self.event_type,
            "tool_id": self.tool_id,
            "risk_level": self.risk_level,
            "policy_decision": self.policy_decision,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "sandbox_used": self.sandbox_used,
            "approval_id": self.approval_id,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "error_code": self.error_code,
            "is_security_event": self.is_security_event,
        }


# =============================================================================
# SESSION STORE PORT
# =============================================================================


class SessionStorePort(abc.ABC):
    """
    Puerto de persistencia de sesiones y turns de conversación.

    Gestiona todo el ciclo de vida de persistencia de sesiones:
    crear, cargar, actualizar, listar y archivar sesiones; así como
    los turns individuales de la conversación.

    Contratos invariantes:
      1. save_session() es idempotente — salvar la misma sesión dos veces
         produce el mismo estado en el storage.
      2. Los turns se añaden con append_turn() — nunca se modifican.
      3. load_session() retorna None si no existe (no lanza KeyError).
      4. Las operaciones de listado están siempre paginadas.
      5. delete_session() elimina en cascada los turns asociados.
      6. Los errores de I/O se convierten en ForgeStorageError.
    """

    @abc.abstractmethod
    async def save_session(self, session: Session) -> None:
        """
        Persiste el estado del aggregate Session.

        Si la sesión no existe, la crea. Si existe, actualiza todos los
        campos mutables (status, last_activity, token counts, etc.).
        No persiste los turns — usa append_turn() para eso.

        Args:
            session: Aggregate Session a persistir.

        Raises:
            ForgeStorageError: Si la operación de I/O falla.
        """

    @abc.abstractmethod
    async def load_session(self, session_id: SessionId) -> Session | None:
        """
        Carga un aggregate Session completo desde el storage.

        Incluye las TurnReferences, TaskReferences y ArtifactReferences
        de la sesión. NO incluye el contenido completo de turns y artifacts
        — solo las referencias ligeras.

        Args:
            session_id: ID de la sesión a cargar.

        Returns:
            Aggregate Session restaurado, o None si no existe.

        Raises:
            ForgeStorageError: Si la operación de I/O falla.
        """

    @abc.abstractmethod
    async def append_turn(
        self,
        session_id: SessionId,
        turn_ref: TurnReference,
        turn_content: dict[str, Any],
    ) -> None:
        """
        Añade un nuevo turn al historial de la sesión.

        La persistencia de turns es append-only — los turns no se modifican
        después de guardarse. El turn_content incluye el contenido completo
        (mensajes, tool calls, resultados) para recuperación futura.

        Args:
            session_id:   ID de la sesión propietaria.
            turn_ref:     Referencia ligera del turn (para el aggregate).
            turn_content: Contenido completo del turn serializado.

        Raises:
            ForgeStorageError: Si la operación falla o la sesión no existe.
        """

    @abc.abstractmethod
    async def load_turns(
        self,
        session_id: SessionId,
        *,
        limit: int = 50,
        offset: int = 0,
        order: str = "asc",
    ) -> list[dict[str, Any]]:
        """
        Carga el contenido completo de turns de una sesión.

        Se usa cuando el ContextBuilder necesita el historial completo
        (no solo las referencias) para construir el LLMContext.

        Args:
            session_id: ID de la sesión.
            limit:      Máximo de turns a cargar.
            offset:     Desplazamiento (para cargar turns antiguos).
            order:      'asc' para cronológico, 'desc' para invertido.

        Returns:
            Lista de dicts con el contenido completo de cada turn.

        Raises:
            ForgeStorageError: Si la operación falla.
        """

    @abc.abstractmethod
    async def load_recent_turns(
        self,
        session_id: SessionId,
        *,
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        """
        Carga los turns más recientes de una sesión que caben en el budget.

        Variante eficiente de load_turns que carga solo los turns que caben
        en max_tokens. Evita cargar todo el historial solo para truncarlo.

        Args:
            session_id: ID de la sesión.
            max_tokens: Budget máximo de tokens para los turns cargados.

        Returns:
            Lista de dicts del historial reciente, en orden cronológico.
        """

    @abc.abstractmethod
    async def append_task_reference(
        self,
        session_id: SessionId,
        task_ref: TaskReference,
    ) -> None:
        """
        Registra una referencia de tarea en la sesión.

        Args:
            session_id: ID de la sesión.
            task_ref:   Referencia de la tarea a registrar.

        Raises:
            ForgeStorageError: Si la operación falla.
        """

    @abc.abstractmethod
    async def append_artifact_reference(
        self,
        session_id: SessionId,
        artifact_ref: ArtifactReference,
    ) -> None:
        """
        Registra una referencia de artifact en la sesión.

        Args:
            session_id:    ID de la sesión.
            artifact_ref:  Referencia del artifact.

        Raises:
            ForgeStorageError: Si la operación falla.
        """

    @abc.abstractmethod
    async def update_compaction_state(
        self,
        session_id: SessionId,
        *,
        compaction_summary: str,
        compacted_turn_ids: list[TurnId],
        freed_tokens: int,
    ) -> None:
        """
        Actualiza el estado de compactación de una sesión.

        Se llama después de que el MemoryManager genera el resumen de
        compactación para persistir el nuevo estado del historial.

        Args:
            session_id:          ID de la sesión.
            compaction_summary:  Resumen generado.
            compacted_turn_ids:  IDs de turns compactados.
            freed_tokens:        Tokens liberados por la compactación.

        Raises:
            ForgeStorageError: Si la operación falla.
        """

    @abc.abstractmethod
    async def list_sessions(
        self,
        query: SessionQuery,
    ) -> PaginatedResult[SessionSummary]:
        """
        Lista sesiones según los criterios del query.

        Retorna SessionSummary (no los aggregates completos) para listados
        eficientes sin cargar historial ni artifacts.

        Args:
            query: Criterios de búsqueda y paginación.

        Returns:
            PaginatedResult con SessionSummary de las sesiones encontradas.

        Raises:
            ForgeStorageError: Si la operación falla.
        """

    @abc.abstractmethod
    async def get_last_active_session(
        self,
        user_id: UserId,
    ) -> SessionId | None:
        """
        Retorna el ID de la última sesión activa del usuario.

        Usado al arrancar la app para restaurar la sesión anterior.

        Args:
            user_id: ID del usuario.

        Returns:
            SessionId de la última sesión activa, o None si no hay ninguna.

        Raises:
            ForgeStorageError: Si la operación falla.
        """

    @abc.abstractmethod
    async def delete_session(self, session_id: SessionId) -> bool:
        """
        Elimina una sesión y todos sus turns del storage.

        Args:
            session_id: ID de la sesión a eliminar.

        Returns:
            True si existía y fue eliminada. False si no existía.

        Raises:
            ForgeStorageError: Si la operación falla.
        """

    @abc.abstractmethod
    async def count_turns(self, session_id: SessionId) -> int:
        """
        Cuenta el total de turns de una sesión sin cargarlos.

        Args:
            session_id: ID de la sesión.

        Returns:
            Número total de turns. 0 si la sesión no existe.

        Raises:
            ForgeStorageError: Si la operación falla.
        """


# =============================================================================
# ARTIFACT STORE PORT
# =============================================================================


class ArtifactStorePort(abc.ABC):
    """
    Puerto de persistencia de artifacts.

    Gestiona el ciclo de vida completo de los artifacts del agente:
    guardar, cargar, buscar y eliminar.

    En V1, todo el contenido se almacena en SQLite. En V2+, los artifacts
    muy grandes (>1MB) se pueden mover a un filesystem local con solo
    la metadata y referencia en SQLite.

    Contratos invariantes:
      1. save() es idempotente para artifacts con el mismo artifact_id.
      2. load() retorna None si el artifact no existe.
      3. Los artifacts son inmutables — save() nunca actualiza el contenido.
         Para "actualizar" un artifact, se crea uno nuevo con parent_artifact_id.
      4. delete() elimina permanentemente el artifact.
      5. search() aplica los criterios en el orden correcto (filtros primero,
         ordenación después, paginación al final).
    """

    @abc.abstractmethod
    async def save(self, artifact: Artifact) -> None:
        """
        Persiste un artifact en el storage.

        Args:
            artifact: Artifact a persistir.

        Raises:
            ForgeStorageError: Si la operación falla o el artifact ya existe
                               con contenido diferente (inmutabilidad violada).
        """

    @abc.abstractmethod
    async def load(self, artifact_id: ArtifactId) -> Artifact | None:
        """
        Carga un artifact completo por su ID.

        Args:
            artifact_id: ID del artifact a cargar.

        Returns:
            Artifact completo con contenido, o None si no existe.

        Raises:
            ForgeStorageError: Si la operación falla.
        """

    @abc.abstractmethod
    async def load_summary(
        self,
        artifact_id: ArtifactId,
    ) -> ArtifactSummary | None:
        """
        Carga solo el summary de un artifact sin cargar el contenido.

        Más eficiente que load() cuando solo se necesita la metadata.

        Args:
            artifact_id: ID del artifact.

        Returns:
            ArtifactSummary sin contenido, o None si no existe.

        Raises:
            ForgeStorageError: Si la operación falla.
        """

    @abc.abstractmethod
    async def list_by_session(
        self,
        session_id: SessionId,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> PaginatedResult[ArtifactSummary]:
        """
        Lista los artifacts de una sesión específica.

        Args:
            session_id: ID de la sesión.
            limit:      Máximo de artifacts a retornar.
            offset:     Desplazamiento para paginación.

        Returns:
            PaginatedResult con ArtifactSummary ordenados por created_at desc.

        Raises:
            ForgeStorageError: Si la operación falla.
        """

    @abc.abstractmethod
    async def search(
        self,
        query: ArtifactQuery,
    ) -> PaginatedResult[ArtifactSummary]:
        """
        Busca artifacts según los criterios del query.

        Args:
            query: Criterios de búsqueda y paginación.

        Returns:
            PaginatedResult con ArtifactSummary que cumplen los criterios.

        Raises:
            ForgeStorageError: Si la operación falla.
        """

    @abc.abstractmethod
    async def delete(self, artifact_id: ArtifactId) -> bool:
        """
        Elimina un artifact del storage.

        Args:
            artifact_id: ID del artifact a eliminar.

        Returns:
            True si existía y fue eliminado. False si no existía.

        Raises:
            ForgeStorageError: Si la operación falla.
        """

    @abc.abstractmethod
    async def delete_by_session(self, session_id: SessionId) -> int:
        """
        Elimina todos los artifacts de una sesión.

        Usado cuando se elimina una sesión completa.

        Args:
            session_id: ID de la sesión cuyos artifacts se eliminan.

        Returns:
            Número de artifacts eliminados.

        Raises:
            ForgeStorageError: Si la operación falla.
        """

    @abc.abstractmethod
    async def count_by_session(self, session_id: SessionId) -> int:
        """
        Cuenta los artifacts de una sesión sin cargarlos.

        Args:
            session_id: ID de la sesión.

        Returns:
            Número de artifacts de la sesión.

        Raises:
            ForgeStorageError: Si la operación falla.
        """


# =============================================================================
# AUDIT LOG PORT
# =============================================================================


class AuditLogPort(abc.ABC):
    """
    Puerto del audit log — registro inmutable de acciones del agente.

    El audit log es el registro histórico de todo lo que el agente ha hecho:
    qué tools ejecutó, con qué riesgos, qué policies se aplicaron, qué
    aprobaciones se solicitaron y cómo se resolvieron.

    Es estrictamente append-only — las entradas nunca se modifican ni eliminan.
    La retención se gestiona con archivado (no borrado) después del período
    configurado en StorageConfig.audit_retention_days.

    Contratos invariantes:
      1. record() es la única operación de escritura — no hay UPDATE ni DELETE.
      2. query() retorna entradas en orden cronológico descendente por defecto.
      3. Las entradas son inmutables una vez registradas.
      4. export() genera un archivo con las entradas en el formato especificado.
      5. Los datos sensibles deben sanitizarse ANTES de llamar a record() —
         el audit log no hace sanitización adicional.
    """

    @abc.abstractmethod
    async def record(self, entry: AuditEntry) -> None:
        """
        Registra una nueva entrada en el audit log.

        Esta es la única operación de escritura del audit log.
        La entrada es inmutable una vez registrada — no se puede modificar.

        Args:
            entry: Entrada del audit log a registrar.

        Raises:
            ForgeStorageError: Si la operación de I/O falla.
                               El caller debe decidir cómo manejar el fallo
                               (generalmente: log del error pero continuar).
        """

    @abc.abstractmethod
    async def record_tool_execution(
        self,
        *,
        session_id: str,
        tool_id: str,
        risk_level: str,
        policy_decision: str,
        success: bool,
        duration_ms: float,
        sandbox_used: bool = False,
        approval_id: str | None = None,
        input_summary: str | None = None,
        output_summary: str | None = None,
        error_code: str | None = None,
        policy_name: str | None = None,
    ) -> None:
        """
        Atajo para registrar una ejecución de tool en el audit log.

        Método de conveniencia que construye el AuditEntry correcto
        para el event_type 'tool_execution'.

        Args:
            session_id:      ID de la sesión.
            tool_id:         ID de la tool ejecutada.
            risk_level:      Nivel de riesgo ('none', 'low', etc.).
            policy_decision: Decisión del policy engine.
            success:         True si la ejecución fue exitosa.
            duration_ms:     Duración en milisegundos.
            sandbox_used:    Si se usó el proceso sandbox.
            approval_id:     ID de aprobación si fue requerida.
            input_summary:   Resumen sanitizado del input.
            output_summary:  Resumen sanitizado del output.
            error_code:      Código de error si falló.
            policy_name:     Nombre de la policy que decidió.

        Raises:
            ForgeStorageError: Si la operación de I/O falla.
        """

    @abc.abstractmethod
    async def record_security_event(
        self,
        *,
        session_id: str,
        event_type: str,
        severity: str,
        details: dict[str, Any],
        tool_id: str | None = None,
    ) -> None:
        """
        Registra un evento de seguridad en el audit log.

        Los eventos de seguridad (path traversal, sandbox violation, etc.)
        se registran siempre con is_security_event=True para facilitar
        su detección en queries de seguridad.

        Args:
            session_id:  ID de la sesión.
            event_type:  Tipo de evento de seguridad.
            severity:    Severidad ('low', 'medium', 'high', 'critical').
            details:     Detalles adicionales del evento (sanitizados).
            tool_id:     Tool involucrada, si aplica.

        Raises:
            ForgeStorageError: Si la operación falla.
        """

    @abc.abstractmethod
    async def query(
        self,
        query: AuditQuery,
    ) -> PaginatedResult[AuditEntry]:
        """
        Consulta el audit log según criterios específicos.

        Args:
            query: Criterios de búsqueda y paginación.

        Returns:
            PaginatedResult con AuditEntry que cumplen los criterios.

        Raises:
            ForgeStorageError: Si la operación falla.
        """

    @abc.abstractmethod
    async def export(
        self,
        *,
        session_id: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        format: str = "json",
    ) -> bytes:
        """
        Exporta entradas del audit log a un formato serializado.

        Útil para el debug CLI y para que el usuario pueda exportar
        su historial de acciones.

        Args:
            session_id: Filtrar por sesión. None exporta todas.
            date_from:  Exportar desde esta fecha.
            date_to:    Exportar hasta esta fecha.
            format:     Formato de exportación: 'json' o 'csv'.

        Returns:
            Bytes del archivo exportado en el formato especificado.

        Raises:
            ForgeStorageError: Si la operación falla.
            ValueError: Si el formato no es soportado.
        """

    @abc.abstractmethod
    async def count_security_events(
        self,
        session_id: str | None = None,
        *,
        last_hours: int = 24,
    ) -> int:
        """
        Cuenta eventos de seguridad recientes.

        Usado por el PolicyEngine para detectar patrones de abuso
        (rate limiting basado en historial de violaciones).

        Args:
            session_id:  Filtrar por sesión. None cuenta en todas.
            last_hours:  Ventana de tiempo en horas.

        Returns:
            Número de eventos de seguridad en la ventana especificada.

        Raises:
            ForgeStorageError: Si la operación falla.
        """


# =============================================================================
# USER PROFILE STORE PORT
# =============================================================================


class UserProfileStorePort(abc.ABC):
    """
    Puerto de persistencia del perfil de usuario.

    Gestiona la persistencia del UserProfile, incluyendo sus preferencias,
    permisos, y estadísticas de uso. En V1 hay exactamente un perfil
    por instalación (single-user). El puerto está preparado para V3 multi-usuario.

    Contratos invariantes:
      1. save() crea el perfil si no existe, o lo actualiza si existe.
      2. load() retorna None si no existe ningún perfil.
      3. Los permisos se serializan por separado del resto del perfil
         para facilitar su actualización granular (ej: first_use approvals).
      4. delete() elimina el perfil y todos sus datos asociados (irreversible).
    """

    @abc.abstractmethod
    async def save(self, profile: UserProfile) -> None:
        """
        Persiste el UserProfile completo.

        Si no existe, lo crea. Si existe, actualiza todos los campos.
        El PermissionSet se serializa y persiste junto con las preferencias.

        Args:
            profile: UserProfile a persistir.

        Raises:
            ForgeStorageError: Si la operación falla.
        """

    @abc.abstractmethod
    async def load(self, user_id: UserId) -> UserProfile | None:
        """
        Carga el UserProfile de un usuario específico.

        Args:
            user_id: ID del usuario.

        Returns:
            UserProfile restaurado, o None si no existe.

        Raises:
            ForgeStorageError: Si la operación falla.
        """

    @abc.abstractmethod
    async def load_default(self) -> UserProfile | None:
        """
        Carga el perfil por defecto (single-user V1).

        En V1, hay un solo usuario por instalación. Este método carga
        ese perfil sin necesitar el UserId.

        Returns:
            UserProfile del usuario por defecto, o None si no hay ninguno.

        Raises:
            ForgeStorageError: Si la operación falla.
        """

    @abc.abstractmethod
    async def save_permissions(
        self,
        user_id: UserId,
        permissions_dict: dict[str, Any],
    ) -> None:
        """
        Actualiza solo el PermissionSet del usuario.

        Operación granular para persistir cambios de permisos sin
        re-serializar todo el perfil. Se llama frecuentemente cuando
        el usuario aprueba tools (first_use approvals).

        Args:
            user_id:          ID del usuario.
            permissions_dict: PermissionSet serializado como dict.

        Raises:
            ForgeStorageError: Si la operación falla.
        """

    @abc.abstractmethod
    async def increment_session_count(self, user_id: UserId) -> None:
        """
        Incrementa el contador de sesiones del usuario.

        Operación atómica — no requiere cargar el perfil completo.

        Args:
            user_id: ID del usuario.

        Raises:
            ForgeStorageError: Si la operación falla.
        """

    @abc.abstractmethod
    async def add_turns_to_total(
        self,
        user_id: UserId,
        turn_count: int,
    ) -> None:
        """
        Añade turns al contador total del usuario.

        Operación atómica — no requiere cargar el perfil completo.

        Args:
            user_id:    ID del usuario.
            turn_count: Número de turns a añadir al total.

        Raises:
            ForgeStorageError: Si la operación falla.
        """

    @abc.abstractmethod
    async def exists(self, user_id: UserId) -> bool:
        """
        Verifica si existe un perfil para el usuario especificado.

        Args:
            user_id: ID del usuario.

        Returns:
            True si existe un perfil para ese usuario.

        Raises:
            ForgeStorageError: Si la operación falla.
        """

    @abc.abstractmethod
    async def delete(self, user_id: UserId) -> bool:
        """
        Elimina el perfil del usuario y todos sus datos asociados.

        Operación IRREVERSIBLE. Debe usarse solo para el flujo de
        "eliminar mi cuenta" (V3) o para resetear la instalación.

        Args:
            user_id: ID del usuario a eliminar.

        Returns:
            True si existía y fue eliminado. False si no existía.

        Raises:
            ForgeStorageError: Si la operación falla.
        """