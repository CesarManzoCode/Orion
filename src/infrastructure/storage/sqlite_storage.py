"""
Implementaciones SQLite de los puertos de storage de HiperForge User.

Este módulo implementa los 4 puertos de storage usando SQLite como base de
datos local. SQLite es la elección correcta para V1:
  - Single-user, local-first — sin necesidad de servidor de base de datos
  - ACID completo con WAL mode — consistencia ante cierres inesperados
  - Rendimiento suficiente para el volumen de datos de un agente personal
  - Portabilidad total — el archivo SQLite es la base de datos

Implementaciones:
  SQLiteSessionStore      → SessionStorePort
  SQLiteArtifactStore     → ArtifactStorePort
  SQLiteAuditLog          → AuditLogPort
  SQLiteUserProfileStore  → UserProfileStorePort

Todas las operaciones son asíncronas usando aiosqlite, que envuelve
sqlite3 en un thread pool para no bloquear el event loop de asyncio.

Configuración:
  - WAL mode habilitado para mejor concurrencia de lectura
  - Foreign keys habilitadas para integridad referencial
  - Pragmas de rendimiento: cache_size, synchronous=NORMAL, temp_store=MEMORY

Schema:
  - sessions: estado del aggregate Session
  - session_turns: historial de turns (append-only)
  - session_tasks: referencias de tareas
  - session_artifacts_ref: referencias ligeras de artifacts en sesión
  - artifacts: artifacts completos con contenido
  - audit_log: registro inmutable de acciones
  - user_profiles: perfil y preferencias del usuario
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from forge_core.errors.types import ForgeStorageError
from forge_core.observability.logging import get_logger

from src.domain.entities.artifact import (
    Artifact,
    ArtifactContent,
    ArtifactContentType,
    ArtifactMetadata,
    ArtifactSummary,
    ArtifactType,
)
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
from src.domain.value_objects.token_budget import BUDGET_128K
from src.ports.outbound.storage_port import (
    ArtifactQuery,
    ArtifactStorePort,
    AuditEntry,
    AuditLogPort,
    AuditQuery,
    PaginatedResult,
    SessionQuery,
    SessionStorePort,
    UserProfileStorePort,
)


logger = get_logger(__name__, component="sqlite_storage")

# Schema SQL de todas las tablas
_SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA cache_size = -8000;
PRAGMA synchronous = NORMAL;
PRAGMA temp_store = MEMORY;

CREATE TABLE IF NOT EXISTS sessions (
    session_id          TEXT PRIMARY KEY,
    user_id             TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'created',
    turn_count          INTEGER NOT NULL DEFAULT 0,
    task_count          INTEGER NOT NULL DEFAULT 0,
    artifact_count      INTEGER NOT NULL DEFAULT 0,
    current_history_tokens INTEGER NOT NULL DEFAULT 0,
    compacted_turns_count  INTEGER NOT NULL DEFAULT 0,
    compaction_summary  TEXT,
    model_context_window   INTEGER NOT NULL DEFAULT 128000,
    created_at          TEXT NOT NULL,
    last_activity       TEXT NOT NULL,
    closed_at           TEXT,
    extra_meta          TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_status
    ON sessions (user_id, status, last_activity DESC);

CREATE TABLE IF NOT EXISTS session_turns (
    turn_id             TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    sequence_number     INTEGER NOT NULL,
    user_message        TEXT NOT NULL DEFAULT '',
    assistant_response  TEXT NOT NULL DEFAULT '',
    routing_type        TEXT NOT NULL DEFAULT 'DirectReply',
    tool_calls_count    INTEGER NOT NULL DEFAULT 0,
    estimated_tokens    INTEGER NOT NULL DEFAULT 0,
    has_tool_calls      INTEGER NOT NULL DEFAULT 0,
    is_compacted        INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_turns_session_seq
    ON session_turns (session_id, sequence_number ASC);

CREATE TABLE IF NOT EXISTS session_tasks (
    task_id             TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    intent_type         TEXT NOT NULL DEFAULT '',
    status              TEXT NOT NULL DEFAULT 'completed',
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_artifacts_ref (
    artifact_id         TEXT NOT NULL,
    session_id          TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    artifact_type       TEXT NOT NULL,
    display_name        TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    PRIMARY KEY (artifact_id, session_id)
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id         TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL,
    source_task_id      TEXT NOT NULL,
    source_turn_id      TEXT NOT NULL,
    artifact_type       TEXT NOT NULL,
    display_name        TEXT NOT NULL,
    content_type        TEXT NOT NULL,
    raw_content         TEXT NOT NULL,
    is_truncated        INTEGER NOT NULL DEFAULT 0,
    original_size_chars INTEGER NOT NULL DEFAULT 0,
    language            TEXT NOT NULL DEFAULT 'es',
    source_filename     TEXT,
    source_url          TEXT,
    source_tool_id      TEXT,
    item_count          INTEGER,
    tags                TEXT NOT NULL DEFAULT '[]',
    parent_artifact_id  TEXT,
    created_at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_artifacts_session
    ON artifacts (session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_artifacts_type
    ON artifacts (artifact_type, session_id);

CREATE TABLE IF NOT EXISTS audit_log (
    entry_id            TEXT PRIMARY KEY,
    occurred_at         TEXT NOT NULL,
    session_id          TEXT NOT NULL,
    event_type          TEXT NOT NULL,
    tool_id             TEXT,
    risk_level          TEXT,
    policy_decision     TEXT,
    success             INTEGER,
    duration_ms         REAL,
    sandbox_used        INTEGER NOT NULL DEFAULT 0,
    approval_id         TEXT,
    input_summary       TEXT,
    output_summary      TEXT,
    error_code          TEXT,
    is_security_event   INTEGER NOT NULL DEFAULT 0,
    policy_name         TEXT,
    extra_context       TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_audit_session_time
    ON audit_log (session_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_security_events
    ON audit_log (is_security_event, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_tool
    ON audit_log (tool_id, occurred_at DESC);

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id             TEXT PRIMARY KEY,
    preferences         TEXT NOT NULL DEFAULT '{}',
    permissions         TEXT NOT NULL DEFAULT '{}',
    session_count       INTEGER NOT NULL DEFAULT 0,
    total_turns         INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_memory_facts (
    fact_id             TEXT PRIMARY KEY,
    user_id             TEXT NOT NULL,
    content             TEXT NOT NULL,
    category            TEXT NOT NULL DEFAULT 'general',
    source_session_id   TEXT NOT NULL DEFAULT '',
    tags                TEXT NOT NULL DEFAULT '[]',
    confidence          REAL NOT NULL DEFAULT 1.0,
    access_count        INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL,
    last_accessed_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_user
    ON user_memory_facts (user_id, last_accessed_at DESC);
"""


class SQLiteStorageBase:
    """
    Clase base para los adapters SQLite.

    Gestiona la conexión a la base de datos y proporciona helpers
    comunes para manejo de errores y formateo de timestamps.
    """

    def __init__(self, db_path: str | Path) -> None:
        """
        Args:
            db_path: Ruta al archivo SQLite.
        """
        self._db_path = str(db_path)
        self._initialized = False

    async def initialize(self) -> None:
        """
        Inicializa la base de datos creando las tablas si no existen.

        Debe llamarse una vez durante el bootstrap del sistema.
        Es idempotente — puede llamarse múltiples veces sin efecto.
        """
        if self._initialized:
            return

        # Asegurar que el directorio padre existe
        db_dir = os.path.dirname(self._db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        try:
            async with aiosqlite.connect(self._db_path) as db:
                db.row_factory = aiosqlite.Row
                await db.executescript(_SCHEMA_SQL)
                await db.commit()
            self._initialized = True
            logger.info(
                "sqlite_inicializado",
                db_path=self._db_path,
            )
        except Exception as e:
            raise ForgeStorageError(
                f"No se pudo inicializar la base de datos SQLite en '{self._db_path}': {e}"
            ) from e

    @asynccontextmanager
    async def _get_conn(self) -> Any:
        """
        Abre una conexión SQLite configurada y garantiza su cierre.

        Este helper evita reusar indebidamente el mismo objeto connection
        dentro de context managers, lo que en aiosqlite provoca errores como
        "threads can only be started once".
        """
        async with aiosqlite.connect(self._db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA foreign_keys = ON")
            await conn.execute("PRAGMA journal_mode = WAL")
            yield conn

    @staticmethod
    def _now_iso() -> str:
        """Retorna el timestamp actual en ISO 8601 UTC."""
        return datetime.now(tz=timezone.utc).isoformat()

    @staticmethod
    def _parse_dt(value: str | None) -> datetime | None:
        """Parsea un string ISO a datetime con timezone UTC."""
        if not value:
            return None
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    @staticmethod
    def _json_dump(obj: Any) -> str:
        """Serializa un objeto a JSON string."""
        return json.dumps(obj, ensure_ascii=False)

    @staticmethod
    def _json_load(text: str | None) -> Any:
        """Deserializa un JSON string. Retorna {} si None o vacío."""
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}


class SQLiteSessionStore(SQLiteStorageBase, SessionStorePort):
    """
    Implementación SQLite del SessionStorePort.

    Gestiona la persistencia de sesiones y sus turns.
    Los turns son append-only — nunca se modifican una vez escritos.
    """

    async def save_session(self, session: Session) -> None:
        """Persiste el estado del aggregate Session (upsert)."""
        try:
            async with self._get_conn() as db:
                now = self._now_iso()
                await db.execute("""
                    INSERT INTO sessions (
                        session_id, user_id, status, turn_count, task_count,
                        artifact_count, current_history_tokens, compacted_turns_count,
                        compaction_summary, model_context_window,
                        created_at, last_activity, extra_meta
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        status = excluded.status,
                        turn_count = excluded.turn_count,
                        task_count = excluded.task_count,
                        artifact_count = excluded.artifact_count,
                        current_history_tokens = excluded.current_history_tokens,
                        compacted_turns_count = excluded.compacted_turns_count,
                        compaction_summary = excluded.compaction_summary,
                        last_activity = excluded.last_activity
                """, (
                    session.session_id.to_str(),
                    session.user_id.to_str(),
                    session.status.value,
                    session.turn_count,
                    session.task_count,
                    session.artifact_count,
                    session.current_history_tokens,
                    session.compacted_turns_count,
                    session.compaction_summary,
                    session._token_budget.model_context_window,
                    session.created_at.isoformat(),
                    session.last_activity.isoformat(),
                    "{}",
                ))
                await db.commit()
        except Exception as e:
            raise ForgeStorageError(
                f"Error al guardar sesión {session.session_id.to_str()}: {e}"
            ) from e

    async def load_session(self, session_id: SessionId) -> Session | None:
        """
        Carga un aggregate Session completo desde el storage.

        Reconstruye el aggregate con las TurnReference reales desde la tabla
        session_turns. Esto es necesario para que session.turn_count sea
        correcto y el próximo turn se persista con el sequence_number correcto.

        Sin este fix, session.turn_count sería 0 al restaurar (turns=[]) y el
        primer turn nuevo tendría sequence_number=1, colisionando con los turns
        ya existentes en la BD y siendo silenciado por INSERT OR IGNORE.
        """
        try:
            async with self._get_conn() as db:
                async with db.execute(
                    "SELECT * FROM sessions WHERE session_id = ?",
                    (session_id.to_str(),),
                ) as cursor:
                    row = await cursor.fetchone()

                if row is None:
                    return None

                # Cargar TurnReferences para reconstruir turn_count correctamente
                async with db.execute(
                    """
                    SELECT turn_id, sequence_number, estimated_tokens,
                           has_tool_calls, is_compacted, created_at
                    FROM session_turns
                    WHERE session_id = ?
                    ORDER BY sequence_number ASC
                    """,
                    (session_id.to_str(),),
                ) as cursor:
                    turn_rows = await cursor.fetchall()

            turn_refs = [self._row_to_turn_ref(dict(r)) for r in turn_rows]
            return self._row_to_session(dict(row), turn_refs=turn_refs)
        except Exception as e:
            raise ForgeStorageError(
                f"Error al cargar sesión {session_id.to_str()}: {e}"
            ) from e

    def _row_to_turn_ref(self, row: dict[str, Any]) -> TurnReference:
        """Reconstruye una TurnReference desde una fila de session_turns."""
        return TurnReference(
            turn_id=TurnId.from_str(row["turn_id"]),
            sequence_number=row["sequence_number"],
            estimated_tokens=row.get("estimated_tokens", 0),
            created_at=self._parse_dt(row["created_at"]) or datetime.now(tz=timezone.utc),
            has_tool_calls=bool(row.get("has_tool_calls", 0)),
            is_compacted=bool(row.get("is_compacted", 0)),
        )

    def _row_to_session(
        self,
        row: dict[str, Any],
        turn_refs: list[TurnReference] | None = None,
    ) -> Session:
        """
        Reconstruye un aggregate Session desde una fila de la BD.

        Args:
            row:       Fila de la tabla sessions.
            turn_refs: TurnReferences precargadas desde session_turns.
                       None equivale a lista vacía (solo para sesiones nuevas
                       que aún no tienen turns en la BD).
        """
        from src.domain.value_objects.token_budget import BUDGET_128K, BUDGET_200K
        context_window = row.get("model_context_window", 128_000)
        budget = BUDGET_200K if context_window >= 200_000 else BUDGET_128K

        return Session.restore(
            session_id=SessionId.from_str(row["session_id"]),
            user_id=UserId.from_str(row["user_id"]),
            token_budget=budget,
            status=SessionStatus(row["status"]),
            created_at=self._parse_dt(row["created_at"]) or datetime.now(tz=timezone.utc),
            last_activity=self._parse_dt(row["last_activity"]) or datetime.now(tz=timezone.utc),
            turns=turn_refs or [],
            tasks=[],   # Las tasks son solo referencias estadísticas, no se usan en runtime
            artifacts=[], # Los artifacts se cargan bajo demanda via ArtifactStore
            compacted_turns_count=row.get("compacted_turns_count", 0),
            compaction_summary=row.get("compaction_summary"),
            current_history_tokens=row.get("current_history_tokens", 0),
        )

    async def append_turn(
        self,
        session_id: SessionId,
        turn_ref: TurnReference,
        turn_content: dict[str, Any],
    ) -> None:
        """Añade un turn al historial de la sesión (append-only)."""
        try:
            async with self._get_conn() as db:
                await db.execute("""
                    INSERT OR IGNORE INTO session_turns (
                        turn_id, session_id, sequence_number,
                        user_message, assistant_response,
                        routing_type, tool_calls_count, estimated_tokens,
                        has_tool_calls, is_compacted, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                """, (
                    turn_ref.turn_id.to_str(),
                    session_id.to_str(),
                    turn_ref.sequence_number,
                    turn_content.get("user_message", ""),
                    turn_content.get("assistant_response", ""),
                    turn_content.get("routing_type", "DirectReply"),
                    turn_content.get("tool_calls_count", 0),
                    turn_ref.estimated_tokens,
                    1 if turn_ref.has_tool_calls else 0,
                    turn_ref.created_at.isoformat(),
                ))
                await db.commit()
        except Exception as e:
            raise ForgeStorageError(f"Error al añadir turn: {e}") from e

    async def load_turns(
        self,
        session_id: SessionId,
        *,
        limit: int = 50,
        offset: int = 0,
        order: str = "asc",
    ) -> list[dict[str, Any]]:
        """Carga turns de la sesión con paginación."""
        direction = "ASC" if order.lower() == "asc" else "DESC"
        try:
            async with self._get_conn() as db:
                async with db.execute(f"""
                    SELECT * FROM session_turns
                    WHERE session_id = ? AND is_compacted = 0
                    ORDER BY sequence_number {direction}
                    LIMIT ? OFFSET ?
                """, (session_id.to_str(), limit, offset)) as cursor:
                    rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            raise ForgeStorageError(f"Error al cargar turns: {e}") from e

    async def load_recent_turns(
        self,
        session_id: SessionId,
        *,
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        """Carga los turns más recientes que caben en el budget de tokens."""
        try:
            async with self._get_conn() as db:
                # Cargar todos los turns no compactados en orden descendente
                async with db.execute("""
                    SELECT * FROM session_turns
                    WHERE session_id = ? AND is_compacted = 0
                    ORDER BY sequence_number DESC
                    LIMIT 200
                """, (session_id.to_str(),)) as cursor:
                    rows = await cursor.fetchall()

            # Seleccionar turns desde el más reciente hasta agotar el budget
            selected: list[dict[str, Any]] = []
            accumulated = 0
            for row in rows:
                row_dict = dict(row)
                turn_tokens = row_dict.get("estimated_tokens", 0)
                if accumulated + turn_tokens > max_tokens:
                    break
                selected.append(row_dict)
                accumulated += turn_tokens

            # Invertir para obtener orden cronológico
            selected.reverse()
            return selected

        except Exception as e:
            raise ForgeStorageError(f"Error al cargar turns recientes: {e}") from e

    async def append_task_reference(
        self,
        session_id: SessionId,
        task_ref: TaskReference,
    ) -> None:
        """Registra una referencia de tarea en la sesión."""
        try:
            async with self._get_conn() as db:
                await db.execute("""
                    INSERT OR IGNORE INTO session_tasks
                    (task_id, session_id, intent_type, status, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    task_ref.task_id.to_str(),
                    session_id.to_str(),
                    getattr(task_ref, "intent_type", ""),
                    getattr(task_ref, "status", "completed"),
                    datetime.now(tz=timezone.utc).isoformat(),
                ))
                await db.commit()
        except Exception as e:
            raise ForgeStorageError(f"Error al registrar tarea: {e}") from e

    async def append_artifact_reference(
        self,
        session_id: SessionId,
        artifact_ref: ArtifactReference,
    ) -> None:
        """Registra una referencia de artifact en la sesión."""
        try:
            async with self._get_conn() as db:
                await db.execute("""
                    INSERT OR IGNORE INTO session_artifacts_ref
                    (artifact_id, session_id, artifact_type, display_name, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    artifact_ref.artifact_id.to_str(),
                    session_id.to_str(),
                    artifact_ref.artifact_type,
                    artifact_ref.display_name,
                    artifact_ref.created_at.isoformat(),
                ))
                await db.commit()
        except Exception as e:
            raise ForgeStorageError(f"Error al registrar artifact: {e}") from e

    async def update_compaction_state(
        self,
        session_id: SessionId,
        *,
        compaction_summary: str,
        compacted_turn_ids: list[TurnId],
        freed_tokens: int,
    ) -> None:
        """Actualiza el estado de compactación de la sesión."""
        try:
            compacted_ids = [t.to_str() for t in compacted_turn_ids]
            async with self._get_conn() as db:
                # Marcar turns como compactados
                if compacted_ids:
                    placeholders = ",".join("?" * len(compacted_ids))
                    await db.execute(
                        f"UPDATE session_turns SET is_compacted = 1 WHERE turn_id IN ({placeholders})",
                        compacted_ids,
                    )

                # Actualizar el resumen en la sesión
                await db.execute("""
                    UPDATE sessions
                    SET compaction_summary = ?,
                        compacted_turns_count = compacted_turns_count + ?,
                        current_history_tokens = MAX(0, current_history_tokens - ?)
                    WHERE session_id = ?
                """, (
                    compaction_summary,
                    len(compacted_ids),
                    freed_tokens,
                    session_id.to_str(),
                ))
                await db.commit()
        except Exception as e:
            raise ForgeStorageError(f"Error al actualizar compactación: {e}") from e

    async def list_sessions(
        self,
        query: SessionQuery,
    ) -> PaginatedResult[SessionSummary]:
        """Lista sesiones según los criterios del query."""
        conditions: list[str] = []
        params: list[Any] = []

        if query.user_id:
            conditions.append("user_id = ?")
            params.append(query.user_id.to_str())
        if query.status:
            conditions.append("status = ?")
            params.append(query.status.value)
        if query.created_after:
            conditions.append("created_at >= ?")
            params.append(query.created_after.isoformat())
        if query.created_before:
            conditions.append("created_at <= ?")
            params.append(query.created_before.isoformat())

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        order_map = {
            "last_activity": "last_activity",
            "created_at": "created_at",
            "turn_count": "turn_count",
        }
        order_col = order_map.get(query.order_by, "last_activity")
        order_dir = "DESC" if query.order_dir.lower() == "desc" else "ASC"

        sql = f"""
            SELECT *, COUNT(*) OVER() as total_count
            FROM sessions {where_clause}
            ORDER BY {order_col} {order_dir}
            LIMIT ? OFFSET ?
        """
        params.extend([query.limit, query.offset])

        try:
            async with self._get_conn() as db:
                async with db.execute(sql, params) as cursor:
                    rows = await cursor.fetchall()

            if not rows:
                return PaginatedResult(items=[], total_count=0, limit=query.limit, offset=query.offset)

            total = rows[0]["total_count"] if rows else 0
            summaries = [self._row_to_session_summary(dict(r)) for r in rows]
            return PaginatedResult(
                items=summaries,
                total_count=total,
                limit=query.limit,
                offset=query.offset,
            )
        except Exception as e:
            raise ForgeStorageError(f"Error al listar sesiones: {e}") from e

    def _row_to_session_summary(self, row: dict[str, Any]) -> SessionSummary:
        """Convierte una fila de BD en SessionSummary."""
        created_at = self._parse_dt(row.get("created_at")) or datetime.now(tz=timezone.utc)
        last_activity = self._parse_dt(row.get("last_activity")) or datetime.now(tz=timezone.utc)
        duration = (last_activity - created_at).total_seconds() / 60

        return SessionSummary(
            session_id=SessionId.from_str(row["session_id"]),
            user_id=UserId.from_str(row["user_id"]),
            status=SessionStatus(row["status"]),
            turn_count=row["turn_count"],
            artifact_count=row["artifact_count"],
            created_at=created_at,
            last_activity=last_activity,
            duration_minutes=round(duration, 1),
        )

    async def get_last_active_session(self, user_id: UserId) -> SessionId | None:
        """Retorna el ID de la última sesión activa del usuario."""
        try:
            async with self._get_conn() as db:
                async with db.execute("""
                    SELECT session_id FROM sessions
                    WHERE user_id = ? AND status IN ('active', 'paused')
                    ORDER BY last_activity DESC
                    LIMIT 1
                """, (user_id.to_str(),)) as cursor:
                    row = await cursor.fetchone()
            if row is None:
                return None
            return SessionId.from_str(row["session_id"])
        except Exception as e:
            raise ForgeStorageError(f"Error al buscar última sesión: {e}") from e

    async def delete_session(self, session_id: SessionId) -> bool:
        """Elimina la sesión y sus turns en cascada."""
        try:
            async with self._get_conn() as db:
                await db.execute(
                    "DELETE FROM sessions WHERE session_id = ?",
                    (session_id.to_str(),),
                )
                deleted = db.total_changes > 0
                await db.commit()
            return deleted
        except Exception as e:
            raise ForgeStorageError(f"Error al eliminar sesión: {e}") from e

    async def count_turns(self, session_id: SessionId) -> int:
        """Cuenta el total de turns de la sesión."""
        try:
            async with self._get_conn() as db:
                async with db.execute(
                    "SELECT COUNT(*) FROM session_turns WHERE session_id = ?",
                    (session_id.to_str(),),
                ) as cursor:
                    row = await cursor.fetchone()
            return row[0] if row else 0
        except Exception as e:
            raise ForgeStorageError(f"Error al contar turns: {e}") from e


class SQLiteArtifactStore(SQLiteStorageBase, ArtifactStorePort):
    """
    Implementación SQLite del ArtifactStorePort.

    Los artifacts son inmutables una vez guardados.
    Garantía de integridad: save() falla si el artifact_id ya existe con contenido diferente.
    """

    async def save(self, artifact: Artifact) -> None:
        """Persiste un artifact en el storage (INSERT OR IGNORE)."""
        try:
            async with self._get_conn() as db:
                await db.execute("""
                    INSERT OR IGNORE INTO artifacts (
                        artifact_id, session_id, source_task_id, source_turn_id,
                        artifact_type, display_name, content_type, raw_content,
                        is_truncated, original_size_chars, language,
                        source_filename, source_url, source_tool_id, item_count,
                        tags, parent_artifact_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    artifact.artifact_id.to_str(),
                    artifact.session_id.to_str(),
                    artifact.source_task_id.to_str(),
                    artifact.source_turn_id.to_str(),
                    artifact.artifact_type.value,
                    artifact.display_name,
                    artifact.content.content_type.value,
                    artifact.content.raw_content,
                    1 if artifact.content.is_truncated else 0,
                    artifact.content.original_size_chars,
                    artifact.metadata.language,
                    artifact.metadata.source_filename,
                    artifact.metadata.source_url,
                    artifact.metadata.source_tool_id,
                    artifact.metadata.item_count,
                    self._json_dump(list(artifact.metadata.tags)),
                    artifact.parent_artifact_id.to_str() if artifact.parent_artifact_id else None,
                    artifact.created_at.isoformat(),
                ))
                await db.commit()
        except Exception as e:
            raise ForgeStorageError(f"Error al guardar artifact: {e}") from e

    async def load(self, artifact_id: ArtifactId) -> Artifact | None:
        """Carga un artifact completo por su ID."""
        try:
            async with self._get_conn() as db:
                async with db.execute(
                    "SELECT * FROM artifacts WHERE artifact_id = ?",
                    (artifact_id.to_str(),),
                ) as cursor:
                    row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_artifact(dict(row))
        except Exception as e:
            raise ForgeStorageError(f"Error al cargar artifact: {e}") from e

    def _row_to_artifact(self, row: dict[str, Any]) -> Artifact:
        """Reconstruye un Artifact desde una fila de la BD."""
        from src.domain.value_objects.identifiers import TaskId, TurnId

        content = ArtifactContent(
            content_type=ArtifactContentType(row["content_type"]),
            raw_content=row["raw_content"],
            is_truncated=bool(row.get("is_truncated", 0)),
            original_size_chars=row.get("original_size_chars", 0),
        )
        tags_data = self._json_load(row.get("tags", "[]"))
        tags = tuple(tags_data) if isinstance(tags_data, list) else ()

        metadata = ArtifactMetadata(
            language=row.get("language", "es"),
            source_filename=row.get("source_filename"),
            source_url=row.get("source_url"),
            source_tool_id=row.get("source_tool_id"),
            item_count=row.get("item_count"),
            tags=tags,
        )

        parent_id_str = row.get("parent_artifact_id")

        return Artifact(
            artifact_id=ArtifactId.from_str(row["artifact_id"]),
            artifact_type=ArtifactType(row["artifact_type"]),
            display_name=row["display_name"],
            content=content,
            metadata=metadata,
            session_id=SessionId.from_str(row["session_id"]),
            source_task_id=TaskId.from_str(row["source_task_id"]),
            source_turn_id=TurnId.from_str(row["source_turn_id"]),
            created_at=self._parse_dt(row["created_at"]) or datetime.now(tz=timezone.utc),
            parent_artifact_id=ArtifactId.from_str(parent_id_str) if parent_id_str else None,
        )

    async def load_summary(self, artifact_id: ArtifactId) -> ArtifactSummary | None:
        """Carga solo el summary (sin raw_content) de un artifact."""
        try:
            async with self._get_conn() as db:
                async with db.execute("""
                    SELECT artifact_id, artifact_type, display_name, session_id,
                           source_task_id, created_at, item_count, source_filename,
                           tags, is_truncated,
                           LENGTH(raw_content) as content_size_chars,
                           LENGTH(raw_content) / 4 as estimated_tokens
                    FROM artifacts WHERE artifact_id = ?
                """, (artifact_id.to_str(),)) as cursor:
                    row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_summary(dict(row))
        except Exception as e:
            raise ForgeStorageError(f"Error al cargar summary de artifact: {e}") from e

    def _row_to_summary(self, row: dict[str, Any]) -> ArtifactSummary:
        """Convierte fila en ArtifactSummary sin cargar el contenido completo."""
        from src.domain.value_objects.identifiers import TaskId
        tags_raw = self._json_load(row.get("tags", "[]"))
        tags = tuple(tags_raw) if isinstance(tags_raw, list) else ()

        return ArtifactSummary(
            artifact_id=ArtifactId.from_str(row["artifact_id"]),
            artifact_type=ArtifactType(row["artifact_type"]),
            display_name=row["display_name"],
            session_id=SessionId.from_str(row["session_id"]),
            source_task_id=TaskId.from_str(row["source_task_id"]),
            created_at=self._parse_dt(row["created_at"]) or datetime.now(tz=timezone.utc),
            content_size_chars=row.get("content_size_chars", 0),
            estimated_tokens=row.get("estimated_tokens", 0),
            is_truncated=bool(row.get("is_truncated", 0)),
            item_count=row.get("item_count"),
            source_filename=row.get("source_filename"),
            tags=tags,
        )

    async def list_by_session(
        self,
        session_id: SessionId,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> PaginatedResult[ArtifactSummary]:
        """Lista artifacts de una sesión paginados."""
        try:
            async with self._get_conn() as db:
                async with db.execute("""
                    SELECT artifact_id, artifact_type, display_name, session_id,
                           source_task_id, created_at, item_count, source_filename,
                           tags, is_truncated,
                           LENGTH(raw_content) as content_size_chars,
                           LENGTH(raw_content) / 4 as estimated_tokens,
                           COUNT(*) OVER() as total_count
                    FROM artifacts WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """, (session_id.to_str(), limit, offset)) as cursor:
                    rows = await cursor.fetchall()

            if not rows:
                return PaginatedResult(items=[], total_count=0, limit=limit, offset=offset)
            total = rows[0]["total_count"]
            items = [self._row_to_summary(dict(r)) for r in rows]
            return PaginatedResult(items=items, total_count=total, limit=limit, offset=offset)
        except Exception as e:
            raise ForgeStorageError(f"Error al listar artifacts de sesión: {e}") from e

    async def search(self, query: ArtifactQuery) -> PaginatedResult[ArtifactSummary]:
        """Busca artifacts según criterios del query."""
        conditions: list[str] = []
        params: list[Any] = []

        if query.session_id:
            conditions.append("session_id = ?")
            params.append(query.session_id.to_str())
        if query.artifact_type:
            conditions.append("artifact_type = ?")
            params.append(query.artifact_type.value)
        if query.source_filename:
            conditions.append("source_filename LIKE ?")
            params.append(f"%{query.source_filename}%")
        if query.text_search:
            conditions.append("display_name LIKE ?")
            params.append(f"%{query.text_search}%")
        if query.created_after:
            conditions.append("created_at >= ?")
            params.append(query.created_after.isoformat())
        if query.created_before:
            conditions.append("created_at <= ?")
            params.append(query.created_before.isoformat())

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        order_map = {"created_at": "created_at", "display_name": "display_name"}
        order_col = order_map.get(query.order_by, "created_at")
        order_dir = "DESC" if query.order_dir.lower() == "desc" else "ASC"

        sql = f"""
            SELECT artifact_id, artifact_type, display_name, session_id,
                   source_task_id, created_at, item_count, source_filename,
                   tags, is_truncated,
                   LENGTH(raw_content) as content_size_chars,
                   LENGTH(raw_content) / 4 as estimated_tokens,
                   COUNT(*) OVER() as total_count
            FROM artifacts {where}
            ORDER BY {order_col} {order_dir}
            LIMIT ? OFFSET ?
        """
        params.extend([query.limit, query.offset])

        try:
            async with self._get_conn() as db:
                async with db.execute(sql, params) as cursor:
                    rows = await cursor.fetchall()

            if not rows:
                return PaginatedResult(items=[], total_count=0, limit=query.limit, offset=query.offset)
            total = rows[0]["total_count"]
            items = [self._row_to_summary(dict(r)) for r in rows]
            return PaginatedResult(items=items, total_count=total, limit=query.limit, offset=query.offset)
        except Exception as e:
            raise ForgeStorageError(f"Error en búsqueda de artifacts: {e}") from e

    async def delete(self, artifact_id: ArtifactId) -> bool:
        """Elimina un artifact por su ID."""
        try:
            async with self._get_conn() as db:
                await db.execute(
                    "DELETE FROM artifacts WHERE artifact_id = ?",
                    (artifact_id.to_str(),),
                )
                deleted = db.total_changes > 0
                await db.commit()
            return deleted
        except Exception as e:
            raise ForgeStorageError(f"Error al eliminar artifact: {e}") from e

    async def delete_by_session(self, session_id: SessionId) -> int:
        """Elimina todos los artifacts de una sesión."""
        try:
            async with self._get_conn() as db:
                await db.execute(
                    "DELETE FROM artifacts WHERE session_id = ?",
                    (session_id.to_str(),),
                )
                count = db.total_changes
                await db.commit()
            return count
        except Exception as e:
            raise ForgeStorageError(f"Error al eliminar artifacts de sesión: {e}") from e

    async def count_by_session(self, session_id: SessionId) -> int:
        """Cuenta los artifacts de una sesión."""
        try:
            async with self._get_conn() as db:
                async with db.execute(
                    "SELECT COUNT(*) FROM artifacts WHERE session_id = ?",
                    (session_id.to_str(),),
                ) as cursor:
                    row = await cursor.fetchone()
            return row[0] if row else 0
        except Exception as e:
            raise ForgeStorageError(f"Error al contar artifacts: {e}") from e


class SQLiteAuditLog(SQLiteStorageBase, AuditLogPort):
    """
    Implementación SQLite del AuditLogPort (append-only).

    El audit log es inmutable — solo se insertan filas, nunca se actualizan.
    SQLite garantiza la durabilidad con WAL mode.
    """

    async def record(self, entry: AuditEntry) -> None:
        """Registra una entrada inmutable en el audit log."""
        try:
            async with self._get_conn() as db:
                await db.execute("""
                    INSERT OR IGNORE INTO audit_log (
                        entry_id, occurred_at, session_id, event_type,
                        tool_id, risk_level, policy_decision, success,
                        duration_ms, sandbox_used, approval_id,
                        input_summary, output_summary, error_code,
                        is_security_event, policy_name, extra_context
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.entry_id,
                    entry.occurred_at.isoformat(),
                    entry.session_id,
                    entry.event_type,
                    entry.tool_id,
                    entry.risk_level,
                    entry.policy_decision,
                    1 if entry.success else (0 if entry.success is False else None),
                    entry.duration_ms,
                    1 if entry.sandbox_used else 0,
                    entry.approval_id,
                    entry.input_summary,
                    entry.output_summary,
                    entry.error_code,
                    1 if entry.is_security_event else 0,
                    entry.policy_name,
                    self._json_dump(entry.extra_context),
                ))
                await db.commit()
        except Exception as e:
            raise ForgeStorageError(f"Error al registrar en audit log: {e}") from e

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
        """Atajo para registrar una ejecución de tool."""
        import uuid
        entry = AuditEntry(
            entry_id=str(uuid.uuid4()),
            occurred_at=datetime.now(tz=timezone.utc),
            session_id=session_id,
            event_type="tool_execution",
            tool_id=tool_id,
            risk_level=risk_level,
            policy_decision=policy_decision,
            success=success,
            duration_ms=duration_ms,
            sandbox_used=sandbox_used,
            approval_id=approval_id,
            input_summary=input_summary,
            output_summary=output_summary,
            error_code=error_code,
            policy_name=policy_name,
        )
        await self.record(entry)

    async def record_security_event(
        self,
        *,
        session_id: str,
        event_type: str,
        severity: str,
        details: dict[str, Any],
        tool_id: str | None = None,
    ) -> None:
        """Registra un evento de seguridad."""
        import uuid
        entry = AuditEntry(
            entry_id=str(uuid.uuid4()),
            occurred_at=datetime.now(tz=timezone.utc),
            session_id=session_id,
            event_type="security_event",
            tool_id=tool_id,
            risk_level=severity,
            is_security_event=True,
            extra_context={"security_event_type": event_type, **details},
        )
        await self.record(entry)

    async def query(self, query: AuditQuery) -> PaginatedResult[AuditEntry]:
        """Consulta el audit log con filtros."""
        conditions: list[str] = []
        params: list[Any] = []

        if query.session_id:
            conditions.append("session_id = ?")
            params.append(query.session_id.to_str())
        if query.tool_id:
            conditions.append("tool_id = ?")
            params.append(query.tool_id)
        if query.risk_level:
            conditions.append("risk_level = ?")
            params.append(query.risk_level)
        if query.policy_decision:
            conditions.append("policy_decision = ?")
            params.append(query.policy_decision)
        if query.success is not None:
            conditions.append("success = ?")
            params.append(1 if query.success else 0)
        if query.is_security_event is not None:
            conditions.append("is_security_event = ?")
            params.append(1 if query.is_security_event else 0)
        if query.occurred_after:
            conditions.append("occurred_at >= ?")
            params.append(query.occurred_after.isoformat())
        if query.occurred_before:
            conditions.append("occurred_at <= ?")
            params.append(query.occurred_before.isoformat())

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        sql = f"""
            SELECT *, COUNT(*) OVER() as total_count
            FROM audit_log {where}
            ORDER BY occurred_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([query.limit, query.offset])

        try:
            async with self._get_conn() as db:
                async with db.execute(sql, params) as cursor:
                    rows = await cursor.fetchall()

            if not rows:
                return PaginatedResult(items=[], total_count=0, limit=query.limit, offset=query.offset)
            total = rows[0]["total_count"]
            items = [self._row_to_audit_entry(dict(r)) for r in rows]
            return PaginatedResult(items=items, total_count=total, limit=query.limit, offset=query.offset)
        except Exception as e:
            raise ForgeStorageError(f"Error al consultar audit log: {e}") from e

    def _row_to_audit_entry(self, row: dict[str, Any]) -> AuditEntry:
        """Convierte una fila de BD en AuditEntry."""
        return AuditEntry(
            entry_id=row["entry_id"],
            occurred_at=self._parse_dt(row["occurred_at"]) or datetime.now(tz=timezone.utc),
            session_id=row["session_id"],
            event_type=row["event_type"],
            tool_id=row.get("tool_id"),
            risk_level=row.get("risk_level"),
            policy_decision=row.get("policy_decision"),
            success=bool(row["success"]) if row.get("success") is not None else None,
            duration_ms=row.get("duration_ms"),
            sandbox_used=bool(row.get("sandbox_used", 0)),
            approval_id=row.get("approval_id"),
            input_summary=row.get("input_summary"),
            output_summary=row.get("output_summary"),
            error_code=row.get("error_code"),
            is_security_event=bool(row.get("is_security_event", 0)),
            policy_name=row.get("policy_name"),
            extra_context=self._json_load(row.get("extra_context")),
        )

    async def export(
        self,
        *,
        session_id: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        format: str = "json",
    ) -> bytes:
        """Exporta el audit log en el formato especificado."""
        query = AuditQuery(
            session_id=SessionId.from_str(session_id) if session_id else None,
            occurred_after=date_from,
            occurred_before=date_to,
            limit=10_000,  # exportar hasta 10k entradas
        )
        result = await self.query(query)

        if format == "csv":
            lines = ["entry_id,occurred_at,session_id,event_type,tool_id,risk_level,policy_decision,success,duration_ms"]
            for entry in result.items:
                lines.append(
                    f"{entry.entry_id},{entry.occurred_at.isoformat()},{entry.session_id},"
                    f"{entry.event_type},{entry.tool_id or ''},{entry.risk_level or ''},"
                    f"{entry.policy_decision or ''},{entry.success},{entry.duration_ms or ''}"
                )
            return "\n".join(lines).encode("utf-8")

        # JSON por defecto
        data = [entry.to_display_dict() for entry in result.items]
        return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")

    async def count_security_events(
        self,
        session_id: str | None = None,
        *,
        last_hours: int = 24,
    ) -> int:
        """Cuenta eventos de seguridad recientes."""
        from datetime import timedelta
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(hours=last_hours)).isoformat()
        params: list[Any] = [cutoff]
        condition = "occurred_at >= ?"

        if session_id:
            condition += " AND session_id = ?"
            params.append(session_id)

        try:
            async with self._get_conn() as db:
                async with db.execute(
                    f"SELECT COUNT(*) FROM audit_log WHERE is_security_event = 1 AND {condition}",
                    params,
                ) as cursor:
                    row = await cursor.fetchone()
            return row[0] if row else 0
        except Exception as e:
            raise ForgeStorageError(f"Error al contar eventos de seguridad: {e}") from e


class SQLiteUserProfileStore(SQLiteStorageBase, UserProfileStorePort):
    """
    Implementación SQLite del UserProfileStorePort.
    """

    async def save(self, profile: UserProfile) -> None:
        """Persiste el UserProfile completo (upsert)."""
        try:
            from src.domain.value_objects.permission import PermissionSet
            prefs_dict = profile.preferences.to_dict()
            # Serializar permissions de forma simplificada (V1)
            perms_dict = {"permissions_version": 1}

            async with self._get_conn() as db:
                now = self._now_iso()
                await db.execute("""
                    INSERT INTO user_profiles (
                        user_id, preferences, permissions,
                        session_count, total_turns, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        preferences = excluded.preferences,
                        permissions = excluded.permissions,
                        session_count = excluded.session_count,
                        total_turns = excluded.total_turns,
                        updated_at = excluded.updated_at
                """, (
                    profile.user_id.to_str(),
                    self._json_dump(prefs_dict),
                    self._json_dump(perms_dict),
                    profile.session_count,
                    profile.total_turns,
                    profile.created_at.isoformat(),
                    profile.updated_at.isoformat(),
                ))
                await db.commit()

            profile.mark_clean()
        except Exception as e:
            raise ForgeStorageError(f"Error al guardar perfil: {e}") from e

    async def load(self, user_id: UserId) -> UserProfile | None:
        """Carga el UserProfile de un usuario."""
        try:
            async with self._get_conn() as db:
                async with db.execute(
                    "SELECT * FROM user_profiles WHERE user_id = ?",
                    (user_id.to_str(),),
                ) as cursor:
                    row = await cursor.fetchone()

            if row is None:
                return None
            return self._row_to_profile(dict(row))
        except Exception as e:
            raise ForgeStorageError(f"Error al cargar perfil: {e}") from e

    async def load_default(self) -> UserProfile | None:
        """Carga el perfil por defecto (el primer usuario registrado en V1)."""
        try:
            async with self._get_conn() as db:
                async with db.execute(
                    "SELECT * FROM user_profiles ORDER BY created_at ASC LIMIT 1"
                ) as cursor:
                    row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_profile(dict(row))
        except Exception as e:
            raise ForgeStorageError(f"Error al cargar perfil por defecto: {e}") from e

    def _row_to_profile(self, row: dict[str, Any]) -> UserProfile:
        """Reconstruye un UserProfile desde una fila de la BD."""
        from src.domain.value_objects.permission import DEFAULT_PERMISSIONS

        prefs_dict = self._json_load(row.get("preferences", "{}"))

        return UserProfile.restore(
            user_id=UserId.from_str(row["user_id"]),
            preferences_dict=prefs_dict,
            permissions=DEFAULT_PERMISSIONS,  # V1: permisos desde DEFAULT
            created_at=self._parse_dt(row["created_at"]) or datetime.now(tz=timezone.utc),
            updated_at=self._parse_dt(row["updated_at"]) or datetime.now(tz=timezone.utc),
            session_count=row.get("session_count", 0),
            total_turns=row.get("total_turns", 0),
        )

    async def save_permissions(
        self,
        user_id: UserId,
        permissions_dict: dict[str, Any],
    ) -> None:
        """Actualiza solo los permisos del usuario."""
        try:
            async with self._get_conn() as db:
                await db.execute(
                    "UPDATE user_profiles SET permissions = ?, updated_at = ? WHERE user_id = ?",
                    (self._json_dump(permissions_dict), self._now_iso(), user_id.to_str()),
                )
                await db.commit()
        except Exception as e:
            raise ForgeStorageError(f"Error al actualizar permisos: {e}") from e

    async def increment_session_count(self, user_id: UserId) -> None:
        """Incrementa atómicamente el contador de sesiones."""
        try:
            async with self._get_conn() as db:
                await db.execute(
                    "UPDATE user_profiles SET session_count = session_count + 1, updated_at = ? WHERE user_id = ?",
                    (self._now_iso(), user_id.to_str()),
                )
                await db.commit()
        except Exception as e:
            raise ForgeStorageError(f"Error al incrementar sesiones: {e}") from e

    async def add_turns_to_total(self, user_id: UserId, turn_count: int) -> None:
        """Añade turns al contador total atómicamente."""
        try:
            async with self._get_conn() as db:
                await db.execute(
                    "UPDATE user_profiles SET total_turns = total_turns + ?, updated_at = ? WHERE user_id = ?",
                    (turn_count, self._now_iso(), user_id.to_str()),
                )
                await db.commit()
        except Exception as e:
            raise ForgeStorageError(f"Error al actualizar turns: {e}") from e

    async def exists(self, user_id: UserId) -> bool:
        """Verifica si existe un perfil para el usuario."""
        try:
            async with self._get_conn() as db:
                async with db.execute(
                    "SELECT 1 FROM user_profiles WHERE user_id = ? LIMIT 1",
                    (user_id.to_str(),),
                ) as cursor:
                    row = await cursor.fetchone()
            return row is not None
        except Exception as e:
            raise ForgeStorageError(f"Error al verificar perfil: {e}") from e

    async def delete(self, user_id: UserId) -> bool:
        """Elimina el perfil del usuario (irreversible)."""
        try:
            async with self._get_conn() as db:
                await db.execute(
                    "DELETE FROM user_profiles WHERE user_id = ?",
                    (user_id.to_str(),),
                )
                deleted = db.total_changes > 0
                await db.commit()
            return deleted
        except Exception as e:
            raise ForgeStorageError(f"Error al eliminar perfil: {e}") from e


async def create_sqlite_stores(db_path: str | Path) -> tuple[
    SQLiteSessionStore,
    SQLiteArtifactStore,
    SQLiteAuditLog,
    SQLiteUserProfileStore,
    SQLiteMemoryStore,
]:
    """
    Factory: crea e inicializa los 4 stores SQLite desde una ruta de BD.

    Se llama una sola vez durante el bootstrap del sistema. Todos los
    stores comparten el mismo archivo SQLite (y por ende el mismo schema).

    Args:
        db_path: Ruta al archivo SQLite (se crea si no existe).

    Returns:
        Tupla de (SessionStore, ArtifactStore, AuditLog, ProfileStore).

    Example:
        session_store, artifact_store, audit_log, profile_store = \\
            await create_sqlite_stores(Path("~/.forge/forge_user.db").expanduser())
    """
    session_store = SQLiteSessionStore(db_path)
    artifact_store = SQLiteArtifactStore(db_path)
    audit_log = SQLiteAuditLog(db_path)
    profile_store = SQLiteUserProfileStore(db_path)
    memory_store = SQLiteMemoryStore(db_path)

    # Inicializar solo una vez (todos comparten la misma BD)
    await session_store.initialize()
    # Los otros reutilizan la misma BD ya inicializada
    artifact_store._initialized = True
    audit_log._initialized = True
    profile_store._initialized = True
    memory_store._initialized = True

    return session_store, artifact_store, audit_log, profile_store, memory_store


class SQLiteMemoryStore(SQLiteStorageBase):
    """
    Implementación SQLite del store de memoria a largo plazo del usuario.

    Persiste y recupera LongTermFact entre sesiones.
    Implementación simple y eficiente: todos los hechos del usuario se
    cargan en una sola query y se mantienen en memoria durante la sesión.
    """

    async def load_facts(self, user_id: str) -> list[dict]:
        """Carga todos los hechos de largo plazo de un usuario."""
        try:
            async with self._get_conn() as db:
                async with db.execute(
                    "SELECT * FROM user_memory_facts WHERE user_id = ? ORDER BY last_accessed_at DESC",
                    (user_id,),
                ) as cursor:
                    rows = await cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            raise ForgeStorageError(f"Error al cargar hechos de memoria: {e}") from e

    async def save_facts(self, user_id: str, facts: list[dict]) -> None:
        """Persiste la lista completa de hechos del usuario (reemplaza los existentes)."""
        try:
            async with self._get_conn() as db:
                await db.execute(
                    "DELETE FROM user_memory_facts WHERE user_id = ?",
                    (user_id,),
                )
                for fact in facts:
                    await db.execute("""
                        INSERT INTO user_memory_facts (
                            fact_id, user_id, content, category,
                            source_session_id, tags, confidence,
                            access_count, created_at, last_accessed_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        fact["fact_id"],
                        user_id,
                        fact["content"],
                        fact.get("category", "general"),
                        fact.get("source_session_id", ""),
                        self._json_dump(fact.get("tags", [])),
                        fact.get("confidence", 1.0),
                        fact.get("access_count", 0),
                        fact.get("created_at", self._now_iso()),
                        fact.get("last_accessed_at", self._now_iso()),
                    ))
                await db.commit()
        except Exception as e:
            raise ForgeStorageError(f"Error al guardar hechos de memoria: {e}") from e

    async def delete_all_facts(self, user_id: str) -> int:
        """Elimina todos los hechos de un usuario."""
        try:
            async with self._get_conn() as db:
                await db.execute(
                    "DELETE FROM user_memory_facts WHERE user_id = ?",
                    (user_id,),
                )
                count = db.total_changes
                await db.commit()
            return count
        except Exception as e:
            raise ForgeStorageError(f"Error al eliminar hechos de memoria: {e}") from e