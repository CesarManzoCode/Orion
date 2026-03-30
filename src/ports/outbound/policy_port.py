"""
Puertos del PolicyEngine y ApprovalWorkflow de HiperForge User.

Este módulo define los contratos del sistema de seguridad del agente.
El PolicyEngine es el guardián de toda acción — nada se ejecuta sin pasar
por él. El ApprovalWorkflow gestiona el ciclo de vida de las confirmaciones
del usuario cuando una acción requiere su consentimiento explícito.

Filosofía de seguridad:
  "En caso de duda, denegar."
  "Un error en el PolicyEngine nunca resulta en una acción permitida."
  "La seguridad es una restricción arquitectónica, no una feature opcional."

El PolicyEngine implementado aquí va más allá de lo que el documento de
arquitectura describe. Es un sistema de defensa en profundidad con múltiples
capas independientes de protección:

  Capa 0: Circuit Breaker
    — Si el PolicyEngine mismo falla N veces consecutivas, activa modo
      fail-closed total: todas las acciones se deniegan hasta restablecer.

  Capa 1: Blacklist global inmutable (compile-time)
    — Patrones de acciones que NUNCA se permiten bajo ninguna circunstancia.
    — No configurable en runtime. Solo cambia con actualizaciones de código.

  Capa 2: Análisis de riesgo del input (DomainRiskClassifier)
    — Path traversal, null bytes, shell metacaracteres, unicode tricks.
    — Elevación automática de riesgo cuando el input es anómalo.

  Capa 3: Evaluación de policies ordenadas por prioridad
    — CRITICAL_BLOCK → SECURITY → ADMIN → DEFAULT → PERMISSIVE → FALLBACK
    — Primera policy cuya condición sea True gana.
    — Fallback siempre DENY (fail-closed).

  Capa 4: Verificación de permisos del usuario (PermissionSet)
    — La policy puede decir ALLOW pero el PermissionSet dice DENY.
    — El PermissionSet tiene precedencia sobre las policies para restricciones
      específicas de usuario.

  Capa 5: Rate limiting multi-dimensional
    — Por sesión: máx tool calls por minuto.
    — Por tool: máx invocaciones por ventana de tiempo.
    — Por categoría: máx acciones de desktop/filesystem por minuto.
    — Global: máx acciones totales por sesión.

  Capa 6: Análisis de patrones de ataque
    — Detección de prompt injection en los argumentos del LLM.
    — Detección de intentos repetidos de acciones bloqueadas (jailbreak).
    — Detección de escalada de privilegios gradual.
    — Al detectar un patrón: incrementar severidad de la sesión → más restricciones.

  Capa 7: Contexto conversacional
    — ¿Es coherente esta acción con la tarea que el usuario pidió?
    — ¿Hay evidencia de que el LLM fue manipulado (prompt injection externo)?
    — Acciones que no tienen sentido en el contexto se marcan como sospechosas.

  Capa 8: Audit y telemetría
    — Toda decisión se registra con toda su evidencia.
    — Los patrones de denegación se analizan para mejorar las policies.

Jerarquía de tipos:

  SecuritySeverity        — nivel de alerta de seguridad de la sesión
  RateLimitState          — estado del rate limiting por dimensión
  AttackPattern           — patrón de ataque detectado
  SessionSecurityContext  — contexto de seguridad acumulado de la sesión
  PolicyViolation         — violación específica detectada
  PolicyEvaluationContext — contexto completo para la evaluación
  PolicyEvaluationDetail  — detalle de la evaluación para cada capa
  RobustPolicyDecision    — decisión enriquecida con evidencia completa
  PolicyEnginePort        — puerto abstracto del engine robusto
  ApprovalRequest         — solicitud de aprobación al usuario
  ApprovalDecision        — decisión del usuario sobre una aprobación
  ApprovalPort            — puerto abstracto del flujo de aprobaciones
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Any

from forge_core.policy.framework import (
    Allow,
    Deny,
    Policy,
    PolicyDecision,
    PolicyEvaluationResult,
    ProposedAction,
    RequireApproval,
)
from forge_core.tools.protocol import RiskAssessment, RiskLevel

from src.domain.value_objects.identifiers import ApprovalId, SessionId
from src.domain.value_objects.permission import PermissionSet
from src.domain.value_objects.risk import InputRiskAnalysis


# =============================================================================
# NIVEL DE ALERTA DE SEGURIDAD DE LA SESIÓN
# =============================================================================


class SecuritySeverity(Enum):
    """
    Nivel de alerta de seguridad acumulado para una sesión.

    A medida que el sistema detecta comportamientos sospechosos en una sesión,
    eleva su SecuritySeverity. Un nivel más alto implica más restricciones
    automáticas — el PolicyEngine aplica políticas más conservadoras sin
    necesidad de configuración manual.

    La severidad NUNCA baja durante una sesión — solo puede subir.
    Esto previene que un atacante "repare" su reputación dentro de la misma sesión
    para luego intentar acciones bloqueadas de nuevo.
    """

    NORMAL = 0
    """
    Sin anomalías detectadas. El PolicyEngine opera con sus reglas estándar.
    Estado inicial de toda sesión nueva.
    """

    ELEVATED = 1
    """
    Se detectó comportamiento inusual (intentos de acceso a paths sensibles,
    queries sospechosas, patrones de enumeración). El PolicyEngine aplica
    reglas más conservadoras: requiere aprobación para más acciones.
    """

    HIGH = 2
    """
    Se detectaron intentos claros de evasión de seguridad (path traversal,
    null bytes, intentos repetidos de acciones bloqueadas). El PolicyEngine
    requiere aprobación para TODAS las acciones de filesystem y desktop.
    El umbral de rate limiting se reduce a la mitad.
    """

    CRITICAL = 3
    """
    Se detectó un ataque confirmado (prompt injection externo, sandbox escape
    attempt, multiple security violations). El PolicyEngine bloquea TODAS
    las tool executions excepto las de tipo RESEARCH con riesgo NONE.
    La sesión debe ser revisada por el usuario.
    """

    LOCKDOWN = 4
    """
    La sesión está en lockdown total. Solo se permite conversación pura.
    Ninguna tool puede ejecutarse hasta que el usuario reinicie la sesión.
    Se activa después de N violaciones CRITICAL en la misma sesión.
    """

    def is_at_least(self, level: SecuritySeverity) -> bool:
        """True si la severidad actual es igual o mayor a level."""
        return self.value >= level.value

    def escalate(self) -> SecuritySeverity:
        """
        Retorna el siguiente nivel de severidad (escalada).

        Si ya está en el nivel máximo, retorna el mismo nivel.
        """
        levels = list(SecuritySeverity)
        idx = levels.index(self)
        if idx + 1 < len(levels):
            return levels[idx + 1]
        return self

    def label(self) -> str:
        """Label legible del nivel de severidad."""
        labels = {
            SecuritySeverity.NORMAL: "normal",
            SecuritySeverity.ELEVATED: "elevado",
            SecuritySeverity.HIGH: "alto",
            SecuritySeverity.CRITICAL: "crítico",
            SecuritySeverity.LOCKDOWN: "bloqueado",
        }
        return labels[self]


# =============================================================================
# ESTADO DE RATE LIMITING
# =============================================================================


@dataclass
class RateLimitState:
    """
    Estado del rate limiting para una dimensión específica.

    El PolicyEngine mantiene múltiples instancias de RateLimitState:
    una por sesión, una por tool, una por categoría, y una global.
    """

    dimension: str
    """Identificador de la dimensión ('session', 'tool:web_search', 'category:desktop')."""

    window_seconds: int
    """Duración de la ventana de tiempo en segundos."""

    max_calls: int
    """Máximo de llamadas permitidas en la ventana."""

    call_timestamps: list[datetime] = field(default_factory=list)
    """Timestamps de las llamadas dentro de la ventana actual."""

    def record_call(self) -> None:
        """Registra una nueva llamada en la ventana actual."""
        self._prune_expired()
        self.call_timestamps.append(datetime.now(tz=timezone.utc))

    def is_rate_limited(self) -> bool:
        """True si se ha alcanzado o superado el límite de llamadas."""
        self._prune_expired()
        return len(self.call_timestamps) >= self.max_calls

    def calls_remaining(self) -> int:
        """Número de llamadas restantes en la ventana actual."""
        self._prune_expired()
        return max(0, self.max_calls - len(self.call_timestamps))

    def time_until_reset_seconds(self) -> float:
        """Segundos hasta que expire la llamada más antigua de la ventana."""
        self._prune_expired()
        if not self.call_timestamps:
            return 0.0
        oldest = self.call_timestamps[0]
        expiry = oldest + timedelta(seconds=self.window_seconds)
        remaining = (expiry - datetime.now(tz=timezone.utc)).total_seconds()
        return max(0.0, remaining)

    def current_rate(self) -> float:
        """Llamadas por segundo en la ventana actual."""
        self._prune_expired()
        if self.window_seconds <= 0:
            return 0.0
        return len(self.call_timestamps) / self.window_seconds

    def _prune_expired(self) -> None:
        """Elimina timestamps fuera de la ventana de tiempo actual."""
        cutoff = datetime.now(tz=timezone.utc) - timedelta(seconds=self.window_seconds)
        self.call_timestamps = [
            ts for ts in self.call_timestamps if ts > cutoff
        ]

    def get_summary(self) -> dict[str, Any]:
        """Serializa el estado para logging."""
        self._prune_expired()
        return {
            "dimension": self.dimension,
            "calls_in_window": len(self.call_timestamps),
            "max_calls": self.max_calls,
            "calls_remaining": self.calls_remaining(),
            "is_limited": self.is_rate_limited(),
            "window_seconds": self.window_seconds,
        }


# =============================================================================
# PATRÓN DE ATAQUE
# =============================================================================


class AttackPatternType(Enum):
    """Tipos de patrones de ataque que el PolicyEngine puede detectar."""

    PATH_TRAVERSAL_ATTEMPT = auto()
    """Intento de acceder a paths fuera del sandbox usando '../' o equivalentes."""

    NULL_BYTE_INJECTION = auto()
    """Intento de inyectar bytes nulos en paths o argumentos."""

    SHELL_INJECTION_ATTEMPT = auto()
    """Presencia de metacaracteres de shell en argumentos de command-like fields."""

    UNICODE_EVASION = auto()
    """Uso de caracteres Unicode para evadir filtros de seguridad."""

    REPEATED_DENIED_ACTION = auto()
    """
    El mismo tipo de acción ha sido denegado N veces consecutivas.
    Indica un intento de jailbreak o evasión.
    """

    PRIVILEGE_ESCALATION_ATTEMPT = auto()
    """
    Secuencia de acciones que gradualmente intenta escalar privilegios.
    Ej: read file → list directory → read sensitive file → write system file.
    """

    PROMPT_INJECTION_SUSPECTED = auto()
    """
    Los argumentos del LLM contienen instrucciones que parecen haber sido
    inyectadas por contenido externo (página web maliciosa, documento, etc.).
    """

    SANDBOX_ESCAPE_ATTEMPT = auto()
    """
    La tool intenta interactuar con el proceso principal desde el sandbox,
    o el principal intenta interactuar con el sandbox de forma no autorizada.
    """

    RATE_LIMIT_EVASION = auto()
    """
    Patrón de llamadas que parece diseñado para evadir el rate limiting
    (ej: alternar entre tools para superar los límites por tool).
    """

    CREDENTIAL_HARVESTING_ATTEMPT = auto()
    """
    Secuencia de accesos a paths/recursos que parecen diseñados para
    recopilar credenciales del usuario.
    """


@dataclass(frozen=True)
class AttackPattern:
    """
    Patrón de ataque detectado en la sesión.

    El PolicyEngine acumula AttackPatterns durante la sesión. Cada nuevo
    patrón puede escalar la SecuritySeverity de la sesión y activar
    controles adicionales automáticamente.
    """

    pattern_type: AttackPatternType
    """Tipo de patrón detectado."""

    detected_at: datetime
    """Timestamp de detección (UTC)."""

    session_id: str
    """ID de la sesión donde se detectó."""

    tool_id: str | None
    """Tool involucrada en la detección, si aplica."""

    evidence: str
    """
    Evidencia sanitizada del patrón.
    Debe ser suficiente para el análisis post-mortem sin exponer datos sensibles.
    """

    severity_impact: int
    """
    Impacto en la SecuritySeverity (0=ninguno, 1=ELEVATED, 2=HIGH, 3=CRITICAL).
    Se usa para decidir si escalar la severidad de la sesión.
    """

    @property
    def is_critical(self) -> bool:
        """True si el patrón por sí solo justifica elevar a CRITICAL."""
        return self.severity_impact >= 3

    @property
    def escalates_severity(self) -> bool:
        """True si el patrón requiere escalar la severidad de la sesión."""
        return self.severity_impact > 0


# =============================================================================
# CONTEXTO DE SEGURIDAD DE LA SESIÓN
# =============================================================================


@dataclass
class SessionSecurityContext:
    """
    Contexto de seguridad acumulado durante una sesión.

    El PolicyEngine mantiene este contexto por sesión. Evoluciona a medida
    que se detectan comportamientos sospechosos. Es la memoria de seguridad
    de la sesión — permite detectar ataques graduales que no son obvios
    en una sola acción aislada.

    Es mutable — se actualiza en cada evaluación. El PolicyEngine lo
    persiste periódicamente en el SessionStore para sobrevivir reinicios.
    """

    session_id: str
    """ID de la sesión."""

    severity: SecuritySeverity = SecuritySeverity.NORMAL
    """Nivel de alerta de seguridad actual de la sesión."""

    attack_patterns: list[AttackPattern] = field(default_factory=list)
    """Patrones de ataque detectados en esta sesión."""

    denied_action_counts: dict[str, int] = field(default_factory=dict)
    """
    Conteo de denials por tipo de acción (tool_id → count).
    Permite detectar intentos repetidos de acciones bloqueadas.
    """

    security_events_count: int = 0
    """Total de eventos de seguridad registrados en la sesión."""

    last_security_event_at: datetime | None = None
    """Timestamp del último evento de seguridad."""

    # Rate limits activos
    rate_limit_states: dict[str, RateLimitState] = field(default_factory=dict)
    """Estados de rate limiting por dimensión."""

    # Historial de acciones recientes para análisis de contexto
    recent_action_types: list[str] = field(default_factory=list)
    """
    Tipos de las últimas N acciones (tool_ids).
    Ventana deslizante de tamaño MAX_RECENT_ACTIONS.
    Usado para detección de secuencias sospechosas.
    """

    MAX_RECENT_ACTIONS: int = field(default=20, init=False, repr=False)
    """Tamaño de la ventana de historial de acciones recientes."""

    REPEATED_DENIAL_THRESHOLD: int = field(default=3, init=False, repr=False)
    """Número de denials consecutivos de la misma acción para marcar como patrón."""

    LOCKDOWN_THRESHOLD: int = field(default=5, init=False, repr=False)
    """Número de eventos CRITICAL para activar lockdown."""

    def record_action(self, tool_id: str) -> None:
        """
        Registra una acción en el historial reciente.

        Mantiene la ventana deslizante de MAX_RECENT_ACTIONS acciones.

        Args:
            tool_id: ID de la tool que se ejecutó.
        """
        self.recent_action_types.append(tool_id)
        if len(self.recent_action_types) > self.MAX_RECENT_ACTIONS:
            self.recent_action_types = self.recent_action_types[-self.MAX_RECENT_ACTIONS:]

    def record_denial(self, tool_id: str) -> None:
        """
        Registra un denial para una tool específica.

        Si el conteo supera REPEATED_DENIAL_THRESHOLD, genera un AttackPattern.

        Args:
            tool_id: ID de la tool denegada.
        """
        self.denied_action_counts[tool_id] = (
            self.denied_action_counts.get(tool_id, 0) + 1
        )

        # Detectar intentos repetidos
        if self.denied_action_counts[tool_id] >= self.REPEATED_DENIAL_THRESHOLD:
            pattern = AttackPattern(
                pattern_type=AttackPatternType.REPEATED_DENIED_ACTION,
                detected_at=datetime.now(tz=timezone.utc),
                session_id=self.session_id,
                tool_id=tool_id,
                evidence=(
                    f"Acción '{tool_id}' denegada "
                    f"{self.denied_action_counts[tool_id]} veces consecutivas."
                ),
                severity_impact=1,  # eleva a ELEVATED
            )
            self.add_pattern(pattern)

    def add_pattern(self, pattern: AttackPattern) -> None:
        """
        Añade un patrón de ataque detectado y escala la severidad si es necesario.

        Args:
            pattern: Patrón detectado.
        """
        self.attack_patterns.append(pattern)
        self.security_events_count += 1
        self.last_security_event_at = pattern.detected_at

        # Escalar severidad según el impacto del patrón
        if pattern.severity_impact >= 3:
            self.severity = SecuritySeverity.CRITICAL
        elif pattern.severity_impact >= 2 and not self.severity.is_at_least(SecuritySeverity.HIGH):
            self.severity = SecuritySeverity.HIGH
        elif pattern.severity_impact >= 1 and not self.severity.is_at_least(SecuritySeverity.ELEVATED):
            self.severity = SecuritySeverity.ELEVATED

        # Verificar umbral de lockdown
        critical_patterns = sum(
            1 for p in self.attack_patterns
            if p.is_critical
        )
        if critical_patterns >= self.LOCKDOWN_THRESHOLD:
            self.severity = SecuritySeverity.LOCKDOWN

    def get_rate_limit_state(
        self,
        dimension: str,
        window_seconds: int,
        max_calls: int,
    ) -> RateLimitState:
        """
        Obtiene o crea el estado de rate limiting para una dimensión.

        Args:
            dimension:       Identificador de la dimensión.
            window_seconds:  Duración de la ventana.
            max_calls:       Máximo de llamadas en la ventana.

        Returns:
            RateLimitState para esa dimensión, creado si no existía.
        """
        if dimension not in self.rate_limit_states:
            self.rate_limit_states[dimension] = RateLimitState(
                dimension=dimension,
                window_seconds=window_seconds,
                max_calls=max_calls,
            )
        return self.rate_limit_states[dimension]

    def get_effective_rate_limit_multiplier(self) -> float:
        """
        Retorna el multiplicador de rate limiting según la severidad actual.

        Un multiplicador < 1.0 hace los límites más estrictos.
        Ejemplo: 0.5 = reducir todos los límites a la mitad.

        Returns:
            Multiplicador entre 0.1 y 1.0.
        """
        multipliers = {
            SecuritySeverity.NORMAL: 1.0,
            SecuritySeverity.ELEVATED: 0.75,
            SecuritySeverity.HIGH: 0.5,
            SecuritySeverity.CRITICAL: 0.25,
            SecuritySeverity.LOCKDOWN: 0.0,  # bloquear todo
        }
        return multipliers[self.severity]

    def to_summary_dict(self) -> dict[str, Any]:
        """Serializa el contexto para logging y el debug CLI."""
        return {
            "session_id": self.session_id,
            "severity": self.severity.label(),
            "attack_patterns_count": len(self.attack_patterns),
            "security_events_count": self.security_events_count,
            "recent_patterns": [
                {
                    "type": p.pattern_type.name,
                    "detected_at": p.detected_at.isoformat(),
                    "tool_id": p.tool_id,
                }
                for p in self.attack_patterns[-5:]  # últimos 5
            ],
            "denied_actions": self.denied_action_counts,
            "rate_limits": {
                dim: state.get_summary()
                for dim, state in self.rate_limit_states.items()
            },
        }


# =============================================================================
# VIOLACIÓN DE POLÍTICA
# =============================================================================


@dataclass(frozen=True)
class PolicyViolation:
    """
    Violación específica detectada durante la evaluación de políticas.

    Cada capa del PolicyEngine puede producir PolicyViolations. El
    RobustPolicyDecision los acumula para tener evidencia completa de
    por qué una acción fue denegada (o por qué se requiere aprobación).
    """

    layer: int
    """
    Capa del PolicyEngine que detectó la violación (0-8).
    Corresponde a las capas documentadas en el módulo.
    """

    violation_type: str
    """Tipo de violación (snake_case, ej: 'path_traversal', 'rate_limit_exceeded')."""

    description: str
    """Descripción legible de la violación para el audit log."""

    severity: str
    """Severidad de la violación: 'info', 'warning', 'error', 'critical'."""

    evidence: str = ""
    """Evidencia sanitizada de la violación."""

    policy_name: str | None = None
    """Nombre de la política que detectó la violación."""


# =============================================================================
# CONTEXTO DE EVALUACIÓN
# =============================================================================


@dataclass(frozen=True)
class PolicyEvaluationContext:
    """
    Contexto completo para una evaluación del PolicyEngine.

    Encapsula todo lo que el PolicyEngine necesita para tomar una decisión
    robusta: la acción propuesta, el estado de seguridad de la sesión,
    los permisos del usuario, y el análisis del input ya realizado.
    """

    proposed_action: ProposedAction
    """La acción que el agente quiere ejecutar."""

    session_security: SessionSecurityContext
    """Contexto de seguridad acumulado de la sesión."""

    permissions: PermissionSet
    """PermissionSet del usuario activo."""

    input_risk_analysis: InputRiskAnalysis
    """Análisis de riesgo del input ya realizado por DomainRiskClassifier."""

    conversation_context_summary: str = ""
    """
    Resumen de los últimos N turns de conversación.
    El PolicyEngine lo usa para verificar coherencia contextual:
    ¿tiene sentido esta acción en el contexto de la conversación?
    """

    task_description: str = ""
    """
    Descripción de la tarea que origina la acción.
    Permite verificar si la acción es coherente con la tarea declarada.
    """

    is_plan_step: bool = False
    """True si la acción es parte de un plan multi-step."""

    plan_step_number: int | None = None
    """Número del step en el plan (para análisis de secuencias)."""

    total_plan_steps: int | None = None
    """Total de steps en el plan (contexto para análisis de complejidad)."""


# =============================================================================
# RESULTADO ROBUSTO DE EVALUACIÓN
# =============================================================================


@dataclass(frozen=True)
class PolicyEvaluationLayerResult:
    """
    Resultado de la evaluación de una capa específica del PolicyEngine.

    Cada capa del engine produce su propio resultado, que se combina
    en el RobustPolicyDecision final.
    """

    layer: int
    """Número de la capa (0-8)."""

    layer_name: str
    """Nombre descriptivo de la capa."""

    passed: bool
    """True si la capa no produjo ningún bloqueo."""

    evaluation_time_ms: float
    """Tiempo de evaluación de esta capa en milisegundos."""

    violations: list[PolicyViolation] = field(default_factory=list)
    """Violaciones detectadas por esta capa."""

    decision_contribution: str = "none"
    """
    Contribución de esta capa a la decisión final:
    'none' | 'allow' | 'deny' | 'require_approval' | 'escalate_severity'
    """

    notes: str = ""
    """Notas adicionales para el audit log."""


@dataclass(frozen=True)
class RobustPolicyDecision:
    """
    Decisión enriquecida del PolicyEngine robusto.

    Extiende PolicyEvaluationResult de forge_core con la evidencia completa
    de todas las capas evaluadas, las violaciones detectadas, y el impacto
    en el contexto de seguridad de la sesión.

    Es la fuente de verdad completa para el audit log de seguridad.
    """

    # --- Decisión final ---
    decision: PolicyDecision
    """La decisión final del PolicyEngine."""

    risk_assessment: RiskAssessment
    """Evaluación de riesgo completa de la acción."""

    # --- Evidencia de evaluación ---
    layer_results: list[PolicyEvaluationLayerResult]
    """Resultados de cada capa del engine, en orden de evaluación."""

    violations: list[PolicyViolation]
    """Todas las violaciones detectadas en todas las capas."""

    winning_layer: int | None
    """
    Número de la capa que determinó la decisión final.
    None si fue la política de fallback (todas las capas pasaron sin decision).
    """

    winning_policy: str | None
    """Nombre de la política que determinó la decisión."""

    # --- Impacto de seguridad ---
    severity_escalated: bool
    """True si esta evaluación escaló la SecuritySeverity de la sesión."""

    new_severity: SecuritySeverity | None
    """Nueva severidad de la sesión si fue escalada."""

    attack_patterns_detected: list[AttackPattern]
    """Patrones de ataque detectados durante esta evaluación."""

    # --- Métricas ---
    total_evaluation_time_ms: float
    """Tiempo total de evaluación en milisegundos."""

    policies_evaluated_count: int
    """Número de políticas evaluadas en la capa 3."""

    # --- Metadatos ---
    evaluated_at: datetime
    """Timestamp de la evaluación (UTC)."""

    session_id: str
    """ID de la sesión para trazabilidad."""

    @property
    def is_blocked(self) -> bool:
        """True si la decisión es DENY."""
        return isinstance(self.decision, Deny)

    @property
    def requires_approval(self) -> bool:
        """True si la decisión requiere aprobación del usuario."""
        return isinstance(self.decision, RequireApproval)

    @property
    def is_allowed(self) -> bool:
        """True si la acción está permitida directamente."""
        return isinstance(self.decision, Allow)

    @property
    def is_security_violation(self) -> bool:
        """True si la evaluación detectó una violación de seguridad real."""
        return (
            isinstance(self.decision, Deny)
            and getattr(self.decision, "is_security_violation", False)
        ) or any(p.is_critical for p in self.attack_patterns_detected)

    @property
    def violation_count(self) -> int:
        """Total de violaciones detectadas en todas las capas."""
        return len(self.violations)

    def to_audit_dict(self) -> dict[str, Any]:
        """Serializa la decisión para el audit log de seguridad."""
        return {
            "decision": str(self.decision),
            "decision_type": type(self.decision).__name__,
            "risk_level": self.risk_assessment.effective_risk.name,
            "is_blocked": self.is_blocked,
            "requires_approval": self.requires_approval,
            "is_security_violation": self.is_security_violation,
            "winning_layer": self.winning_layer,
            "winning_policy": self.winning_policy,
            "violations_count": self.violation_count,
            "violations": [
                {
                    "layer": v.layer,
                    "type": v.violation_type,
                    "severity": v.severity,
                    "description": v.description,
                }
                for v in self.violations
            ],
            "severity_escalated": self.severity_escalated,
            "new_severity": self.new_severity.label() if self.new_severity else None,
            "attack_patterns": [
                {
                    "type": p.pattern_type.name,
                    "severity_impact": p.severity_impact,
                    "tool_id": p.tool_id,
                }
                for p in self.attack_patterns_detected
            ],
            "evaluation_time_ms": round(self.total_evaluation_time_ms, 3),
            "policies_evaluated": self.policies_evaluated_count,
            "evaluated_at": self.evaluated_at.isoformat(),
            "session_id": self.session_id,
        }

    def get_user_facing_reason(self) -> str:
        """
        Retorna una razón legible para el usuario sin revelar detalles técnicos.

        El usuario ve esta razón en la UI cuando su solicitud es bloqueada.
        No debe revelar detalles de implementación de las políticas de seguridad.

        Returns:
            String con la razón legible del bloqueo.
        """
        if isinstance(self.decision, Allow):
            return "Acción permitida."

        if isinstance(self.decision, RequireApproval):
            desc = getattr(self.decision, "description", "Esta acción requiere tu confirmación.")
            return desc

        if isinstance(self.decision, Deny):
            reason = getattr(self.decision, "reason", "")
            is_sec = getattr(self.decision, "is_security_violation", False)

            if is_sec:
                return (
                    "Esta acción fue bloqueada por razones de seguridad. "
                    "Si crees que es un error, intenta reformular tu solicitud."
                )

            if reason:
                # Limpiar detalles técnicos del reason para el usuario
                # Retornar solo la primera oración
                first_sentence = reason.split(".")[0].strip()
                return first_sentence or "Esta acción no está permitida."

        return "Esta acción no está permitida en este momento."


# =============================================================================
# PUERTO ABSTRACTO: POLICY ENGINE ROBUSTO
# =============================================================================


class PolicyEnginePort(abc.ABC):
    """
    Puerto del PolicyEngine robusto de HiperForge User.

    Define el contrato del sistema de evaluación de políticas multi-capa.
    Es el punto de control central de toda acción del agente.

    Contratos invariantes — TODOS son no negociables:

      1. FAIL-CLOSED ABSOLUTO: Cualquier error interno en la evaluación
         produce Deny, nunca Allow. Si el engine lanza una excepción internamente,
         la captura y retorna Deny con is_security_violation=False.

      2. DETERMINISMO POR DEFECTO: El mismo (action, context) siempre
         produce la misma decisión. Las únicas excepciones son rate limiting
         (depende del tiempo) y session security context (estado acumulado).

      3. VELOCIDAD: evaluate() debe completar en < 10ms en condiciones normales.
         No hace I/O. No llama al LLM. No accede a la red.

      4. INMUTABILIDAD DEL CONTEXTO: evaluate() no modifica el ProposedAction
         ni el PermissionSet. Solo puede modificar el SessionSecurityContext.

      5. AUDIT COMPLETO: Toda evaluación genera un RobustPolicyDecision con
         evidencia completa de todas las capas. No hay evaluaciones silenciosas.

      6. MONOTICIDAD DE SEVERIDAD: La SecuritySeverity de una sesión solo
         puede subir, nunca bajar. Una vez en LOCKDOWN, permanece hasta
         que el usuario reinicia la sesión explícitamente.

      7. CIRCUIT BREAKER: Si el engine falla más de N veces consecutivas
         (error interno), entra en modo fail-closed total: todas las evaluaciones
         retornan Deny hasta que se recupere. N = 3 por defecto.

      8. THREAD/ASYNC SAFETY: evaluate() es seguro para llamarse concurrentemente
         desde múltiples coroutines de asyncio. El estado mutable
         (SessionSecurityContext) usa locks internos.
    """

    @abc.abstractmethod
    async def evaluate(
        self,
        context: PolicyEvaluationContext,
    ) -> RobustPolicyDecision:
        """
        Evalúa una acción propuesta a través de todas las capas de seguridad.

        Esta es la operación central del PolicyEngine. Ejecuta las 9 capas
        de evaluación en orden y produce una RobustPolicyDecision con
        evidencia completa.

        La evaluación es fail-closed: cualquier error interno produce Deny.
        La evaluación nunca lanza excepciones al caller.

        Args:
            context: PolicyEvaluationContext con toda la información necesaria.

        Returns:
            RobustPolicyDecision con la decisión y evidencia completa.
            Nunca lanza excepciones — los errores van en la decisión.
        """

    @abc.abstractmethod
    async def evaluate_batch(
        self,
        contexts: list[PolicyEvaluationContext],
    ) -> list[RobustPolicyDecision]:
        """
        Evalúa múltiples acciones en batch.

        Útil para el LightweightPlanner cuando necesita pre-evaluar todos
        los steps de un plan antes de comenzar la ejecución.

        La evaluación se realiza secuencialmente — el contexto de seguridad
        de la sesión puede actualizarse entre evaluaciones si se detectan
        patrones de ataque.

        Args:
            contexts: Lista de contextos a evaluar en orden.

        Returns:
            Lista de RobustPolicyDecision en el mismo orden que contexts.
        """

    @abc.abstractmethod
    async def classify_risk(
        self,
        proposed_action: ProposedAction,
        session_security: SessionSecurityContext,
        input_analysis: InputRiskAnalysis,
    ) -> RiskAssessment:
        """
        Clasifica el riesgo de una acción sin ejecutar la evaluación completa.

        Más rápido que evaluate() — solo ejecuta las capas 1 y 2 (blacklist
        y análisis de input). Útil cuando el ContextBuilder necesita conocer
        el riesgo sin bloquear la ejecución.

        Args:
            proposed_action:  La acción a clasificar.
            session_security: Contexto de seguridad de la sesión.
            input_analysis:   Análisis del input ya realizado.

        Returns:
            RiskAssessment con el riesgo efectivo de la acción.
        """

    @abc.abstractmethod
    def get_session_security_context(
        self,
        session_id: str,
    ) -> SessionSecurityContext:
        """
        Obtiene o crea el contexto de seguridad de una sesión.

        El contexto se crea la primera vez que se solicita para una sesión
        y se mantiene en memoria durante toda la sesión.

        Args:
            session_id: ID de la sesión.

        Returns:
            SessionSecurityContext de la sesión.
        """

    @abc.abstractmethod
    def reset_session_security(self, session_id: str) -> None:
        """
        Reinicia el contexto de seguridad de una sesión.

        Solo el usuario puede invocar esto explícitamente desde la UI.
        Permite salir del estado LOCKDOWN cuando el usuario así lo decide.
        No está disponible para el agente — solo para el usuario.

        Args:
            session_id: ID de la sesión a reiniciar.
        """

    @abc.abstractmethod
    def register_policy(self, policy: Policy) -> None:
        """
        Registra una nueva política en el engine.

        Args:
            policy: Política a registrar. Si ya existe una con el mismo nombre,
                    la reemplaza.
        """

    @abc.abstractmethod
    def remove_policy(self, policy_name: str) -> bool:
        """
        Elimina una política por nombre.

        Args:
            policy_name: Nombre de la política a eliminar.

        Returns:
            True si existía y fue eliminada.
        """

    @abc.abstractmethod
    def get_active_policies(self) -> list[Policy]:
        """
        Retorna las políticas activas ordenadas por prioridad descendente.

        Returns:
            Lista de Policy activas.
        """

    @abc.abstractmethod
    def get_circuit_breaker_state(self) -> dict[str, Any]:
        """
        Retorna el estado del circuit breaker del PolicyEngine.

        Returns:
            Dict con estado: {'is_open': bool, 'failure_count': int, ...}.
        """

    @abc.abstractmethod
    async def get_security_report(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Genera un reporte de seguridad de la sesión para el debug CLI.

        Args:
            session_id: ID de la sesión.

        Returns:
            Dict con el reporte completo de seguridad de la sesión.
        """

    @abc.abstractmethod
    def is_in_lockdown(self, session_id: str) -> bool:
        """
        Verifica si una sesión está en modo lockdown.

        Args:
            session_id: ID de la sesión.

        Returns:
            True si la sesión está en SecuritySeverity.LOCKDOWN.
        """

    @property
    @abc.abstractmethod
    def is_circuit_breaker_open(self) -> bool:
        """
        True si el circuit breaker está abierto (modo fail-closed total).

        Cuando está abierto, todas las evaluaciones retornan Deny
        independientemente de la acción o el contexto.
        """

    @property
    @abc.abstractmethod
    def total_evaluations(self) -> int:
        """Total de evaluaciones realizadas desde el inicio."""

    @property
    @abc.abstractmethod
    def total_denials(self) -> int:
        """Total de denials emitidos desde el inicio."""

    @property
    @abc.abstractmethod
    def total_security_events(self) -> int:
        """Total de eventos de seguridad detectados desde el inicio."""


# =============================================================================
# TIPOS DEL FLUJO DE APROBACIONES
# =============================================================================


class ApprovalStatus(Enum):
    """Estado del ciclo de vida de una solicitud de aprobación."""

    PENDING = "pending"
    """La aprobación está esperando la decisión del usuario."""

    GRANTED = "granted"
    """El usuario aprobó la acción."""

    DENIED = "denied"
    """El usuario denegó la acción."""

    EXPIRED = "expired"
    """El timeout expiró sin respuesta — auto-denegada."""

    CANCELLED = "cancelled"
    """La tarea fue cancelada antes de que el usuario respondiera."""


@dataclass(frozen=True)
class ApprovalRequest:
    """
    Solicitud de aprobación presentada al usuario.

    El ApprovalWorkflow crea este objeto y lo presenta al usuario via UI.
    Contiene toda la información necesaria para que el usuario tome una
    decisión informada — sin detalles técnicos, en lenguaje natural.
    """

    approval_id: ApprovalId
    """ID único de esta solicitud."""

    session_id: SessionId
    """ID de la sesión que originó la solicitud."""

    description: str
    """
    Descripción legible para el usuario de qué se quiere hacer.
    Sin IDs técnicos, sin schemas JSON, sin paths completos sensibles.
    Ejemplo: 'Quiero crear un archivo en tu carpeta Documentos'
    """

    risk_level: RiskLevel
    """Nivel de riesgo de la acción."""

    risk_label: str
    """Label legible del riesgo para la UI."""

    tool_display_name: str
    """Nombre legible de la tool que se quiere ejecutar."""

    is_first_time: bool
    """True si es la primera vez que el usuario ve esta acción."""

    can_remember_decision: bool
    """
    True si el sistema puede recordar la decisión del usuario para no
    preguntar de nuevo en el futuro.
    """

    expires_at: datetime
    """Timestamp de expiración de la solicitud (UTC)."""

    created_at: datetime
    """Timestamp de creación (UTC)."""

    context_summary: str = ""
    """
    Contexto adicional para ayudar al usuario a decidir.
    Ej: 'Estás pidiendo analizar el archivo biología_celular.pdf'
    """

    risk_explanation: str = ""
    """
    Explicación de por qué esta acción requiere aprobación.
    En lenguaje no técnico.
    """

    @property
    def is_expired(self) -> bool:
        """True si la solicitud ya expiró."""
        return datetime.now(tz=timezone.utc) > self.expires_at

    @property
    def seconds_remaining(self) -> float:
        """Segundos hasta la expiración. Negativo si ya expiró."""
        delta = self.expires_at - datetime.now(tz=timezone.utc)
        return delta.total_seconds()

    def to_ui_dict(self) -> dict[str, Any]:
        """Serializa la solicitud para la UI."""
        return {
            "approval_id": self.approval_id.to_str(),
            "description": self.description,
            "risk_level": self.risk_level.name.lower(),
            "risk_label": self.risk_label,
            "tool_name": self.tool_display_name,
            "is_first_time": self.is_first_time,
            "can_remember": self.can_remember_decision,
            "context": self.context_summary,
            "risk_explanation": self.risk_explanation,
            "expires_in_seconds": max(0, int(self.seconds_remaining)),
            "is_expired": self.is_expired,
        }


@dataclass(frozen=True)
class ApprovalDecision:
    """
    Decisión del usuario sobre una solicitud de aprobación.

    El ApprovalPort crea este objeto cuando el usuario responde.
    """

    approval_id: ApprovalId
    """ID de la solicitud de aprobación."""

    granted: bool
    """True si el usuario aprobó, False si denegó."""

    remember_decision: bool
    """True si el usuario quiere que se recuerde esta decisión."""

    decided_at: datetime
    """Timestamp de la decisión (UTC)."""

    decided_by: str = "user"
    """
    Quién tomó la decisión: 'user' (explícita), 'timeout' (auto-deny),
    'cancelled' (tarea cancelada antes de decidir).
    """

    @property
    def status(self) -> ApprovalStatus:
        """Estado derivado de la decisión."""
        if self.decided_by == "timeout":
            return ApprovalStatus.EXPIRED
        if self.decided_by == "cancelled":
            return ApprovalStatus.CANCELLED
        return ApprovalStatus.GRANTED if self.granted else ApprovalStatus.DENIED


# =============================================================================
# PUERTO ABSTRACTO: APPROVAL WORKFLOW
# =============================================================================


class ApprovalPort(abc.ABC):
    """
    Puerto del flujo de aprobaciones del usuario.

    Gestiona el ciclo de vida completo de las solicitudes de aprobación:
    crear la solicitud, presentarla al usuario, esperar la respuesta,
    manejar timeouts, y registrar la decisión.

    El flujo típico es:
      1. TaskExecutor detecta que una acción requiere aprobación (REQUIRE_APPROVAL)
      2. TaskExecutor llama a request_approval() → obtiene ApprovalRequest
      3. TaskExecutor llama a present_to_user() → la UI muestra la solicitud
      4. El usuario responde → la UI llama a resolve() con la decisión
      5. TaskExecutor llama a await_decision() → obtiene ApprovalDecision
      6. Si granted → continúa la ejecución
         Si denied o expired → cancela la acción

    Contratos invariantes:
      1. request_approval() crea la solicitud con timeout configurable.
         Una vez creada, el timeout no puede extenderse.
      2. await_decision() es async y se resuelve cuando:
         - El usuario responde (via resolve())
         - El timeout expira (auto-deny)
         - La tarea es cancelada (via cancel())
      3. Toda solicitud tiene exactamente una decisión — no se puede
         responder dos veces.
      4. Las solicitudes expiradas retornan ApprovalDecision con decided_by='timeout'.
      5. El audit log siempre registra la decisión independientemente del resultado.
    """

    @abc.abstractmethod
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
        timeout_seconds: float = 300.0,
    ) -> ApprovalRequest:
        """
        Crea una nueva solicitud de aprobación.

        Genera el ApprovalRequest y lo almacena internamente para que
        await_decision() pueda resolverlo cuando el usuario responda.

        Args:
            session_id:          ID de la sesión.
            description:         Descripción legible para el usuario.
            risk_level:          Nivel de riesgo de la acción.
            tool_id:             ID de la tool a aprobar.
            tool_display_name:   Nombre legible de la tool.
            is_first_time:       True si es primera vez.
            can_remember_decision: Si el usuario puede elegir recordar.
            context_summary:     Contexto adicional para el usuario.
            risk_explanation:    Explicación del riesgo en lenguaje natural.
            timeout_seconds:     Segundos hasta auto-deny si no hay respuesta.

        Returns:
            ApprovalRequest creado y listo para presentar al usuario.
        """

    @abc.abstractmethod
    async def await_decision(
        self,
        approval_id: ApprovalId,
    ) -> ApprovalDecision:
        """
        Espera la decisión del usuario de forma asíncrona.

        Se suspende hasta que el usuario responda o el timeout expire.
        Nunca bloquea el event loop — usa asyncio.Event internamente.

        Args:
            approval_id: ID de la solicitud a esperar.

        Returns:
            ApprovalDecision con la decisión del usuario o auto-deny por timeout.
        """

    @abc.abstractmethod
    async def resolve(
        self,
        approval_id: ApprovalId,
        *,
        granted: bool,
        remember_decision: bool = False,
    ) -> None:
        """
        Registra la decisión del usuario sobre una solicitud de aprobación.

        La UI llama a este método cuando el usuario hace clic en
        "Aprobar" o "Denegar". Resuelve el await_decision() correspondiente.

        Args:
            approval_id:       ID de la solicitud.
            granted:           True si aprobó, False si denegó.
            remember_decision: True si el usuario quiere recordar la decisión.

        Raises:
            ValueError: Si la solicitud no existe o ya fue resuelta.
        """

    @abc.abstractmethod
    async def cancel(self, approval_id: ApprovalId) -> None:
        """
        Cancela una solicitud de aprobación pendiente.

        Se llama cuando la tarea es cancelada antes de que el usuario responda.

        Args:
            approval_id: ID de la solicitud a cancelar.
        """

    @abc.abstractmethod
    async def get_pending(
        self,
        session_id: SessionId,
    ) -> list[ApprovalRequest]:
        """
        Retorna todas las solicitudes de aprobación pendientes de la sesión.

        Args:
            session_id: ID de la sesión.

        Returns:
            Lista de ApprovalRequest pendientes, ordenadas por created_at.
        """

    @abc.abstractmethod
    async def expire_stale(self, max_age_seconds: float) -> int:
        """
        Expira y resuelve solicitudes que llevan más de max_age_seconds pendientes.

        El sistema llama a este método periódicamente para limpiar
        solicitudes antiguas que el usuario no respondió.

        Args:
            max_age_seconds: Edad máxima en segundos antes de expirar.

        Returns:
            Número de solicitudes expiradas en esta llamada.
        """

    @abc.abstractmethod
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

    @property
    @abc.abstractmethod
    def pending_count(self) -> int:
        """Número de solicitudes de aprobación pendientes en total."""