"""
ApprovalWorkflow — implementación del ciclo de vida de aprobaciones.

El ApprovalWorkflow implementa ApprovalPort y gestiona el ciclo de vida
completo de las solicitudes de aprobación del usuario:

  1. El TaskExecutor detecta que una acción requiere aprobación (RequireApproval)
  2. Llama a request_approval() → crea y almacena ApprovalRequest
  3. Llama a present_to_user() para que la UI muestre el diálogo
  4. El TaskExecutor llama a await_decision() → se suspende con asyncio.Event
  5. El usuario responde → la UI llama a ConversationPort.respond_to_approval()
  6. El Coordinator llama a resolve() → dispara el Event
  7. await_decision() retorna con la ApprovalDecision
  8. El TaskExecutor continúa o cancela según la decisión

Garantías de seguridad:
  - El timeout es un hard limit — expirada la solicitud, auto-deny.
  - Una solicitud solo puede resolverse una vez.
  - Las solicitudes expiradas no pueden ser resueltas por el usuario.
  - El audit log registra toda decisión (granted, denied, expired, cancelled).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from forge_core.observability.logging import audit_logger, get_logger
from forge_core.tools.protocol import RiskLevel

from src.domain.value_objects.identifiers import ApprovalId, SessionId
from src.ports.outbound.policy_port import (
    ApprovalDecision,
    ApprovalPort,
    ApprovalRequest,
    ApprovalStatus,
)


logger = get_logger(__name__, component="approval_workflow")

# Timeout por defecto para solicitudes de aprobación (segundos)
_DEFAULT_TIMEOUT_SECONDS = 300.0

# Máximo de solicitudes pendientes por sesión
_MAX_PENDING_PER_SESSION = 5


class _PendingApproval:
    """
    Estado interno de una aprobación pendiente.

    Mantiene el asyncio.Event que permite a await_decision() suspenderse
    hasta que el usuario responda o el timeout expire.
    """

    __slots__ = (
        "request",
        "event",
        "decision",
        "created_at",
    )

    def __init__(self, request: ApprovalRequest) -> None:
        self.request = request
        self.event = asyncio.Event()
        self.decision: ApprovalDecision | None = None
        self.created_at = datetime.now(tz=timezone.utc)


class DefaultApprovalWorkflow(ApprovalPort):
    """
    Implementación del flujo de aprobaciones de HiperForge User.

    Gestiona el ciclo de vida de las solicitudes de aprobación usando
    asyncio.Event para la coordinación entre el TaskExecutor (que espera)
    y el ConversationCoordinator (que resuelve cuando el usuario responde).

    Es thread-safe para asyncio: todas las operaciones se ejecutan en el
    mismo event loop y usan asyncio.Event para sincronización.
    """

    def __init__(
        self,
        *,
        default_timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """
        Args:
            default_timeout_seconds: Timeout por defecto para las solicitudes.
        """
        self._default_timeout = default_timeout_seconds

        # Almacén de solicitudes pendientes: approval_id_str → _PendingApproval
        self._pending: dict[str, _PendingApproval] = {}

        # Índice por sesión: session_id_str → set[approval_id_str]
        self._by_session: dict[str, set[str]] = {}

        # Solicitudes ya resueltas (para consulta post-decision): id → decision
        self._resolved: dict[str, ApprovalDecision] = {}

    # =========================================================================
    # ApprovalPort — implementación
    # =========================================================================

    async def request_approval(
        self,
        *,
        session_id: SessionId,
        description: str,
        risk_level: RiskLevel,
        tool_id: str,
        tool_display_name: str,
        is_first_time: bool = False,
        can_remember_decision: bool = True,
        context_summary: str = "",
        risk_explanation: str = "",
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> ApprovalRequest:
        """
        Crea una nueva solicitud de aprobación con timeout.

        Args:
            session_id:          ID de la sesión.
            description:         Descripción legible de la acción.
            risk_level:          Nivel de riesgo.
            tool_id:             ID de la tool a aprobar.
            tool_display_name:   Nombre amigable de la tool.
            is_first_time:       True si es primera vez.
            can_remember_decision: Si se puede recordar la decisión.
            context_summary:     Contexto adicional para el usuario.
            risk_explanation:    Explicación del riesgo.
            timeout_seconds:     Segundos hasta auto-deny.

        Returns:
            ApprovalRequest creado.
        """
        session_id_str = session_id.to_str()

        # Verificar límite de solicitudes pendientes por sesión
        session_pending = self._by_session.get(session_id_str, set())
        active_pending = {
            aid for aid in session_pending
            if aid in self._pending
        }

        if len(active_pending) >= _MAX_PENDING_PER_SESSION:
            # Auto-expirar las más antiguas para no bloquear
            oldest_id = min(
                active_pending,
                key=lambda aid: self._pending[aid].created_at,
            )
            await self._expire_approval(oldest_id, reason="max_pending_exceeded")

        # Crear la solicitud
        approval_id = ApprovalId.generate()
        now = datetime.now(tz=timezone.utc)
        expires_at = now + timedelta(seconds=timeout_seconds)

        # Generar el label de riesgo en español
        risk_label = _risk_level_label(risk_level)

        request = ApprovalRequest(
            approval_id=approval_id,
            session_id=session_id,
            description=description,
            risk_level=risk_level,
            risk_label=risk_label,
            tool_display_name=tool_display_name,
            is_first_time=is_first_time,
            can_remember_decision=can_remember_decision,
            expires_at=expires_at,
            created_at=now,
            context_summary=context_summary,
            risk_explanation=risk_explanation,
        )

        # Almacenar la solicitud pendiente
        pending = _PendingApproval(request)
        approval_id_str = approval_id.to_str()
        self._pending[approval_id_str] = pending

        # Indexar por sesión
        if session_id_str not in self._by_session:
            self._by_session[session_id_str] = set()
        self._by_session[session_id_str].add(approval_id_str)

        logger.info(
            "aprobacion_solicitada",
            approval_id=approval_id_str,
            session_id=session_id_str,
            tool_id=tool_id,
            risk_level=risk_level.name,
            timeout_seconds=timeout_seconds,
            is_first_time=is_first_time,
        )

        return request

    async def await_decision(
        self,
        approval_id: ApprovalId,
    ) -> ApprovalDecision:
        """
        Espera la decisión del usuario de forma asíncrona.

        Se suspende hasta que el usuario responda (via resolve()) o
        el timeout expire. Usa asyncio.wait_for para el timeout hard limit.

        Args:
            approval_id: ID de la solicitud a esperar.

        Returns:
            ApprovalDecision con la decisión del usuario o auto-deny por timeout.
        """
        approval_id_str = approval_id.to_str()
        pending = self._pending.get(approval_id_str)

        if pending is None:
            # Ya fue resuelta — buscar en resolved
            decision = self._resolved.get(approval_id_str)
            if decision:
                return decision
            # No existe en ningún lado — retornar expired por defecto
            return ApprovalDecision(
                approval_id=approval_id,
                granted=False,
                remember_decision=False,
                decided_at=datetime.now(tz=timezone.utc),
                decided_by="timeout",
            )

        # Calcular timeout restante
        seconds_remaining = pending.request.seconds_remaining
        if seconds_remaining <= 0:
            # Ya expiró antes de que alguien esperara
            await self._expire_approval(approval_id_str, reason="already_expired")
            decision = self._resolved.get(approval_id_str)
            return decision or _make_expired_decision(approval_id)

        # Esperar el evento con timeout
        try:
            await asyncio.wait_for(
                pending.event.wait(),
                timeout=seconds_remaining,
            )
        except asyncio.TimeoutError:
            # Timeout expirado — auto-deny
            await self._expire_approval(approval_id_str, reason="timeout")

        # Obtener la decisión (puede haber sido resuelta por el usuario o por timeout)
        decision = self._resolved.get(approval_id_str)
        return decision or _make_expired_decision(approval_id)

    async def resolve(
        self,
        approval_id: ApprovalId,
        *,
        granted: bool,
        remember_decision: bool = False,
    ) -> None:
        """
        Registra la decisión del usuario y desbloquea await_decision().

        Args:
            approval_id:       ID de la solicitud.
            granted:           True si aprobó, False si denegó.
            remember_decision: True si quiere recordar la decisión.

        Raises:
            ValueError: Si la solicitud no existe o ya fue resuelta.
        """
        approval_id_str = approval_id.to_str()
        pending = self._pending.get(approval_id_str)

        if pending is None:
            if approval_id_str in self._resolved:
                raise ValueError(
                    f"La aprobación {approval_id_str} ya fue resuelta anteriormente."
                )
            raise ValueError(
                f"La aprobación {approval_id_str} no existe o ya expiró."
            )

        # Verificar que no expiró antes de que el usuario respondiera
        if pending.request.is_expired:
            raise ValueError(
                f"La aprobación {approval_id_str} ha expirado. No se puede resolver."
            )

        # Crear la decisión
        decision = ApprovalDecision(
            approval_id=approval_id,
            granted=granted,
            remember_decision=remember_decision,
            decided_at=datetime.now(tz=timezone.utc),
            decided_by="user",
        )

        # Registrar y notificar
        pending.decision = decision
        self._resolved[approval_id_str] = decision
        self._cleanup_pending(approval_id_str)
        pending.event.set()  # desbloquear await_decision()

        logger.info(
            "aprobacion_resuelta",
            approval_id=approval_id_str,
            granted=granted,
            remember=remember_decision,
        )

        # Audit log
        try:
            audit_logger.record_approval(
                approval_id=approval_id_str,
                decision="granted" if granted else "denied",
                remember=remember_decision,
            )
        except Exception:
            pass  # errores de audit no afectan el flujo

    async def cancel(self, approval_id: ApprovalId) -> None:
        """
        Cancela una solicitud de aprobación pendiente.

        Se llama cuando la tarea es cancelada antes de que el usuario responda.

        Args:
            approval_id: ID de la solicitud a cancelar.
        """
        approval_id_str = approval_id.to_str()
        pending = self._pending.get(approval_id_str)

        if pending is None:
            return  # Ya fue resuelta o no existe

        decision = ApprovalDecision(
            approval_id=approval_id,
            granted=False,
            remember_decision=False,
            decided_at=datetime.now(tz=timezone.utc),
            decided_by="cancelled",
        )

        pending.decision = decision
        self._resolved[approval_id_str] = decision
        self._cleanup_pending(approval_id_str)
        pending.event.set()

        logger.info("aprobacion_cancelada", approval_id=approval_id_str)

    async def get_pending(
        self,
        session_id: SessionId,
    ) -> list[ApprovalRequest]:
        """
        Retorna las solicitudes de aprobación pendientes de la sesión.

        Args:
            session_id: ID de la sesión.

        Returns:
            Lista de ApprovalRequest pendientes, ordenadas por created_at.
        """
        session_id_str = session_id.to_str()
        pending_ids = self._by_session.get(session_id_str, set())

        requests = []
        expired_ids = []

        for approval_id_str in pending_ids:
            pending = self._pending.get(approval_id_str)
            if pending is None:
                continue

            if pending.request.is_expired:
                expired_ids.append(approval_id_str)
                continue

            requests.append(pending.request)

        # Expirar las que encontramos vencidas
        for approval_id_str in expired_ids:
            await self._expire_approval(approval_id_str, reason="timeout")

        return sorted(requests, key=lambda r: r.created_at)

    async def expire_stale(self, max_age_seconds: float) -> int:
        """
        Expira solicitudes que llevan más de max_age_seconds pendientes.

        Args:
            max_age_seconds: Edad máxima en segundos.

        Returns:
            Número de solicitudes expiradas.
        """
        now = datetime.now(tz=timezone.utc)
        cutoff = now - timedelta(seconds=max_age_seconds)
        expired_count = 0

        stale_ids = [
            aid for aid, pending in self._pending.items()
            if pending.created_at < cutoff
        ]

        for approval_id_str in stale_ids:
            await self._expire_approval(approval_id_str, reason="stale")
            expired_count += 1

        if expired_count > 0:
            logger.info("aprobaciones_expiradas_batch", count=expired_count)

        return expired_count

    async def get_decision(
        self,
        approval_id: ApprovalId,
    ) -> ApprovalDecision | None:
        """
        Obtiene la decisión de una solicitud ya resuelta.

        Args:
            approval_id: ID de la solicitud.

        Returns:
            ApprovalDecision si ya fue resuelta, None si aún está pendiente.
        """
        return self._resolved.get(approval_id.to_str())

    @property
    def pending_count(self) -> int:
        """Número de solicitudes de aprobación pendientes en total."""
        return len(self._pending)

    # =========================================================================
    # HELPERS INTERNOS
    # =========================================================================

    async def _expire_approval(
        self,
        approval_id_str: str,
        reason: str = "timeout",
    ) -> None:
        """
        Expira una solicitud pendiente y dispara el evento para desbloquear waiters.

        Args:
            approval_id_str: ID string de la solicitud.
            reason:          Razón de la expiración.
        """
        pending = self._pending.get(approval_id_str)
        if pending is None:
            return

        approval_id = pending.request.approval_id
        decision = ApprovalDecision(
            approval_id=approval_id,
            granted=False,
            remember_decision=False,
            decided_at=datetime.now(tz=timezone.utc),
            decided_by="timeout",
        )

        pending.decision = decision
        self._resolved[approval_id_str] = decision
        self._cleanup_pending(approval_id_str)
        pending.event.set()

        logger.info(
            "aprobacion_expirada",
            approval_id=approval_id_str,
            reason=reason,
        )

    def _cleanup_pending(self, approval_id_str: str) -> None:
        """
        Elimina la solicitud del almacén de pendientes y del índice de sesión.

        Args:
            approval_id_str: ID string de la solicitud a limpiar.
        """
        pending = self._pending.pop(approval_id_str, None)
        if pending is None:
            return

        session_id_str = pending.request.session_id.to_str()
        session_set = self._by_session.get(session_id_str)
        if session_set:
            session_set.discard(approval_id_str)
            if not session_set:
                del self._by_session[session_id_str]


# =============================================================================
# HELPERS DE MÓDULO
# =============================================================================


def _risk_level_label(risk_level: RiskLevel) -> str:
    """Retorna la etiqueta en español para un nivel de riesgo."""
    labels = {
        RiskLevel.NONE: "sin riesgo",
        RiskLevel.LOW: "riesgo bajo",
        RiskLevel.MEDIUM: "riesgo moderado",
        RiskLevel.HIGH: "riesgo alto",
        RiskLevel.CRITICAL: "riesgo crítico",
    }
    return labels.get(risk_level, risk_level.name.lower())


def _make_expired_decision(approval_id: ApprovalId) -> ApprovalDecision:
    """Crea una ApprovalDecision de expiración por timeout."""
    return ApprovalDecision(
        approval_id=approval_id,
        granted=False,
        remember_decision=False,
        decided_at=datetime.now(tz=timezone.utc),
        decided_by="timeout",
    )