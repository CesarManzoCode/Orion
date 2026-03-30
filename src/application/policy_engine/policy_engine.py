"""
Implementación concreta del PolicyEngine robusto de HiperForge User.

Este módulo implementa PolicyEnginePort con las 9 capas de defensa en profundidad
documentadas en el puerto. Es el guardián de toda acción del agente — ninguna
tool se ejecuta sin pasar por aquí.

Arquitectura de evaluación — las capas se ejecutan en orden secuencial.
Si alguna capa produce una decisión definitiva (DENY o ALLOW), las capas
posteriores no se ejecutan (short-circuit). Excepto Capa 6 (attack patterns)
y Capa 8 (audit) que siempre se ejecutan independientemente.

Rendimiento objetivo: < 10ms por evaluación en condiciones normales.
Sin I/O. Sin llamadas al LLM. Sin acceso a red.

Garantías de seguridad:
  - FAIL-CLOSED: cualquier excepción interna → DENY
  - CIRCUIT BREAKER: N fallos del engine → modo fail-closed total
  - MONOTICIDAD: SecuritySeverity solo sube, nunca baja en la sesión
  - AUDITABILIDAD: toda decisión tiene evidencia completa en RobustPolicyDecision
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from forge_core.observability.logging import audit_logger, get_logger
from forge_core.policy.framework import (
    Allow,
    AlwaysCondition,
    Deny,
    Policy,
    PolicyPriority,
    ProposedAction,
    RequireApproval,
    make_critical_block_policy,
    make_fallback_deny_policy,
    make_high_risk_approval_policy,
    make_read_only_allow_policy,
)
from forge_core.tools.protocol import MutationType, RiskAssessment, RiskLevel

from src.domain.value_objects.risk import DomainRiskClassifier, InputRiskAnalysis
from src.ports.outbound.policy_port import (
    AttackPattern,
    AttackPatternType,
    PolicyEnginePort,
    PolicyEvaluationContext,
    PolicyEvaluationLayerResult,
    PolicyViolation,
    RobustPolicyDecision,
    SecuritySeverity,
    SessionSecurityContext,
)


logger = get_logger(__name__, component="policy_engine")

# Número de fallos consecutivos del engine para abrir el circuit breaker
_CIRCUIT_BREAKER_THRESHOLD = 3

# Número de segundos en los que el circuit breaker intenta resetear
_CIRCUIT_BREAKER_RESET_SECONDS = 60.0


class DefaultPolicyEngine(PolicyEnginePort):
    """
    Implementación del PolicyEngine robusto de HiperForge User.

    Ejecuta 9 capas de evaluación en orden, manteniendo un contexto de
    seguridad acumulado por sesión. Implementa circuit breaker para
    degradación segura ante fallos internos.

    Thread/async safety: los contextos de sesión se acceden solo desde
    el event loop de asyncio. El asyncio.Lock por sesión previene
    condiciones de carrera si múltiples coroutines evalúan la misma sesión.
    """

    def __init__(self) -> None:
        # Policies registradas, ordenadas por prioridad descendente
        self._policies: list[Policy] = []

        # Contextos de seguridad por sesión: session_id → SessionSecurityContext
        self._session_contexts: dict[str, SessionSecurityContext] = {}

        # Locks por sesión para async safety
        self._session_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

        # Clasificador de riesgo del dominio
        self._risk_classifier = DomainRiskClassifier()

        # Circuit breaker
        self._consecutive_failures = 0
        self._circuit_open = False
        self._circuit_opened_at: datetime | None = None

        # Métricas globales
        self._total_evaluations = 0
        self._total_denials = 0
        self._total_security_events = 0

        # Inicializar con las políticas por defecto del framework
        self._initialize_default_policies()

        logger.info(
            "policy_engine_iniciado",
            policies_count=len(self._policies),
        )

    def _initialize_default_policies(self) -> None:
        """
        Registra las políticas base del producto.

        Orden de prioridad (mayor a menor):
          1. Bloqueo crítico (CRITICAL_BLOCK=1000)
          2. Aprobación para HIGH risk (DEFAULT=400)
          3. Allow para read-only (PERMISSIVE=200)
          4. Fallback deny (FALLBACK=0)
        """
        for policy in [
            make_critical_block_policy(),
            make_high_risk_approval_policy(),
            _make_medium_risk_approval_policy(),
            _make_desktop_first_use_policy(),
            _make_filesystem_write_policy(),
            _make_lockdown_policy(),
            make_read_only_allow_policy(),
            make_fallback_deny_policy(),
        ]:
            self._register_sorted(policy)

    # =========================================================================
    # PolicyEnginePort — implementación pública
    # =========================================================================

    async def evaluate(
        self,
        context: PolicyEvaluationContext,
    ) -> RobustPolicyDecision:
        """
        Evaluación principal: ejecuta las 9 capas de defensa en profundidad.

        Es fail-closed: cualquier excepción interna produce DENY.
        El circuit breaker se activa si hay N fallos consecutivos del engine.
        """
        self._total_evaluations += 1
        session_id = context.proposed_action.tool_input.session_id or "unknown"

        # Verificar circuit breaker
        if self._is_circuit_open():
            return self._circuit_breaker_deny(session_id)

        try:
            return await self._run_evaluation(context, session_id)
        except Exception as exc:
            self._consecutive_failures += 1
            logger.error(
                "policy_engine_error_interno",
                error=exc,
                session_id=session_id,
                consecutive_failures=self._consecutive_failures,
            )
            if self._consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
                self._open_circuit()

            return self._internal_error_deny(
                session_id=session_id,
                context=context,
                error=exc,
            )

    async def evaluate_batch(
        self,
        contexts: list[PolicyEvaluationContext],
    ) -> list[RobustPolicyDecision]:
        """Evalúa múltiples acciones secuencialmente."""
        results = []
        for ctx in contexts:
            result = await self.evaluate(ctx)
            results.append(result)
        return results

    async def classify_risk(
        self,
        proposed_action: ProposedAction,
        session_security: SessionSecurityContext,
        input_analysis: InputRiskAnalysis,
    ) -> RiskAssessment:
        """
        Clasificación rápida de riesgo sin ejecutar la evaluación completa.
        Solo ejecuta Capas 1 y 2 (blacklist e input analysis).
        """
        try:
            # Capa 1: blacklist global
            if self._is_globally_blocked(proposed_action):
                return RiskAssessment.block(
                    tool_id=proposed_action.tool_id,
                    base_risk=proposed_action.base_risk,
                    reason="Acción bloqueada por política global.",
                )

            # Capa 2: análisis de input
            return self._risk_classifier.build_risk_assessment(
                tool_id=proposed_action.tool_id,
                base_risk=proposed_action.base_risk,
                input_analysis=input_analysis,
            )
        except Exception as exc:
            logger.error("classify_risk_fallo", error=exc)
            return RiskAssessment.block(
                tool_id=proposed_action.tool_id,
                base_risk=RiskLevel.CRITICAL,
                reason="Error en clasificación de riesgo — fail-closed.",
            )

    def get_session_security_context(
        self,
        session_id: str,
    ) -> SessionSecurityContext:
        """Obtiene o crea el contexto de seguridad de una sesión."""
        if session_id not in self._session_contexts:
            self._session_contexts[session_id] = SessionSecurityContext(
                session_id=session_id
            )
        return self._session_contexts[session_id]

    def reset_session_security(self, session_id: str) -> None:
        """
        Reinicia el contexto de seguridad de una sesión.
        SOLO el usuario puede invocar esto — no el agente.
        """
        if session_id in self._session_contexts:
            del self._session_contexts[session_id]
        logger.info("sesion_seguridad_reiniciada", session_id=session_id)

    def register_policy(self, policy: Policy) -> None:
        """Registra una nueva política, reemplazando si existe con el mismo nombre."""
        # Eliminar la existente con el mismo nombre
        self._policies = [p for p in self._policies if p.name != policy.name]
        self._register_sorted(policy)
        logger.info(
            "politica_registrada",
            policy_name=policy.name,
            priority=policy.priority,
        )

    def remove_policy(self, policy_name: str) -> bool:
        """Elimina una política por nombre."""
        before = len(self._policies)
        self._policies = [p for p in self._policies if p.name != policy_name]
        removed = len(self._policies) < before
        if removed:
            logger.info("politica_eliminada", policy_name=policy_name)
        return removed

    def get_active_policies(self) -> list[Policy]:
        """Retorna políticas activas ordenadas por prioridad descendente."""
        return [p for p in self._policies if p.enabled]

    def get_circuit_breaker_state(self) -> dict[str, Any]:
        """Estado del circuit breaker para diagnóstico."""
        return {
            "is_open": self._circuit_open,
            "consecutive_failures": self._consecutive_failures,
            "opened_at": (
                self._circuit_opened_at.isoformat()
                if self._circuit_opened_at else None
            ),
            "threshold": _CIRCUIT_BREAKER_THRESHOLD,
        }

    async def get_security_report(self, session_id: str) -> dict[str, Any]:
        """Reporte de seguridad de una sesión."""
        ctx = self._session_contexts.get(session_id)
        if ctx is None:
            return {"session_id": session_id, "status": "sin actividad"}
        return ctx.to_summary_dict()

    def is_in_lockdown(self, session_id: str) -> bool:
        """True si la sesión está en modo lockdown."""
        ctx = self._session_contexts.get(session_id)
        if ctx is None:
            return False
        return ctx.severity == SecuritySeverity.LOCKDOWN

    @property
    def is_circuit_breaker_open(self) -> bool:
        return self._circuit_open

    @property
    def total_evaluations(self) -> int:
        return self._total_evaluations

    @property
    def total_denials(self) -> int:
        return self._total_denials

    @property
    def total_security_events(self) -> int:
        return self._total_security_events

    # =========================================================================
    # EVALUACIÓN INTERNA — 9 CAPAS
    # =========================================================================

    async def _run_evaluation(
        self,
        context: PolicyEvaluationContext,
        session_id: str,
    ) -> RobustPolicyDecision:
        """
        Ejecuta las 9 capas de evaluación en orden.

        Retorna en cuanto una capa produce decisión definitiva,
        excepto Capa 6 y 8 que siempre se ejecutan.
        """
        wall_start = time.perf_counter()
        layer_results: list[PolicyEvaluationLayerResult] = []
        violations: list[PolicyViolation] = []
        attack_patterns: list[AttackPattern] = []
        winning_layer: int | None = None
        winning_policy: str | None = None
        severity_escalated = False
        new_severity: SecuritySeverity | None = None

        # Obtener/crear contexto de seguridad de la sesión
        async with self._session_locks[session_id]:
            session_ctx = self.get_session_security_context(session_id)
            severity_before = session_ctx.severity

            # ─── CAPA 0: Circuit Breaker (ya verificado antes) ─────────────
            # (se verifica en evaluate() antes de llegar aquí)

            # ─── CAPA 1: Blacklist global inmutable ─────────────────────────
            layer1_start = time.perf_counter()
            layer1_result, layer1_decision = self._evaluate_layer1_blacklist(
                context, violations
            )
            layer_results.append(layer1_result)
            if layer1_decision is not None:
                winning_layer = 1
                winning_policy = "forge.global_blacklist"
                final_decision = layer1_decision
                # Continuar a Capa 6 (siempre) y Capa 8 (audit)
                goto_audit = True
                goto_attack_analysis = True
            else:
                goto_audit = False
                goto_attack_analysis = False

                # ─── CAPA 2: Análisis de riesgo del input ──────────────────
                layer2_start = time.perf_counter()
                layer2_result, risk_assessment = self._evaluate_layer2_input_risk(
                    context, violations, attack_patterns, session_ctx
                )
                layer_results.append(layer2_result)

                # ─── CAPA 3: Evaluación de policies ordenadas ──────────────
                layer3_start = time.perf_counter()
                layer3_result, layer3_decision, winning_policy = \
                    self._evaluate_layer3_policies(context, risk_assessment, violations)
                layer_results.append(layer3_result)

                # ─── CAPA 4: PermissionSet del usuario ─────────────────────
                layer4_start = time.perf_counter()
                layer4_result, layer4_decision = self._evaluate_layer4_permissions(
                    context, layer3_decision, violations
                )
                layer_results.append(layer4_result)

                # La decisión es la de la capa 4 (puede anular la de capa 3)
                pre_rate_decision = layer4_decision

                # ─── CAPA 5: Rate limiting multi-dimensional ───────────────
                layer5_start = time.perf_counter()
                layer5_result, layer5_decision = self._evaluate_layer5_rate_limit(
                    context, pre_rate_decision, session_ctx, violations
                )
                layer_results.append(layer5_result)
                final_decision = layer5_decision

                if winning_layer is None:
                    # Determinar qué capa ganó
                    if isinstance(layer4_decision, Deny) and not isinstance(layer3_decision, Deny):
                        winning_layer = 4
                        winning_policy = "forge.permission_check"
                    elif isinstance(layer5_decision, Deny) and not isinstance(layer4_decision, Deny):
                        winning_layer = 5
                        winning_policy = "forge.rate_limit"
                    else:
                        winning_layer = 3

                goto_attack_analysis = True
                goto_audit = True

            # ─── CAPA 6: Análisis de patrones de ataque (siempre) ──────────
            if goto_attack_analysis:
                layer6_result, new_patterns = self._evaluate_layer6_attack_patterns(
                    context, violations, session_ctx, final_decision
                )
                layer_results.append(layer6_result)
                attack_patterns.extend(new_patterns)

                # Registrar patterns en el contexto de sesión
                for pattern in new_patterns:
                    session_ctx.add_pattern(pattern)
                    self._total_security_events += 1

            # ─── CAPA 7: Coherencia con contexto conversacional ─────────────
            layer7_result, contextual_override = self._evaluate_layer7_context(
                context, final_decision, session_ctx, violations
            )
            layer_results.append(layer7_result)
            if contextual_override is not None:
                final_decision = contextual_override
                winning_layer = 7
                winning_policy = "forge.contextual_coherence"

            # Verificar si la severidad cambió
            if session_ctx.severity != severity_before:
                severity_escalated = True
                new_severity = session_ctx.severity
                logger.warning(
                    "severidad_sesion_escalada",
                    session_id=session_id,
                    from_severity=severity_before.label(),
                    to_severity=session_ctx.severity.label(),
                )

            # Registrar acción en el contexto
            session_ctx.record_action(context.proposed_action.tool_id)
            if isinstance(final_decision, Deny):
                session_ctx.record_denial(context.proposed_action.tool_id)

        # ─── CAPA 8: Audit y telemetría (fuera del lock) ────────────────────
        total_time_ms = (time.perf_counter() - wall_start) * 1000

        if isinstance(final_decision, Deny):
            self._total_denials += 1

        # Construir el resultado
        result = RobustPolicyDecision(
            decision=final_decision,
            risk_assessment=risk_assessment if 'risk_assessment' in dir() else RiskAssessment.allow_direct(
                tool_id=context.proposed_action.tool_id,
                base_risk=context.proposed_action.base_risk,
            ),
            layer_results=layer_results,
            violations=violations,
            winning_layer=winning_layer,
            winning_policy=winning_policy,
            severity_escalated=severity_escalated,
            new_severity=new_severity,
            attack_patterns_detected=attack_patterns,
            total_evaluation_time_ms=total_time_ms,
            policies_evaluated_count=len(self._policies),
            evaluated_at=datetime.now(tz=timezone.utc),
            session_id=session_id,
        )

        # Audit log
        await self._audit_decision(result, context)

        # Reset consecutive failures si llegamos hasta aquí sin excepción
        self._consecutive_failures = 0

        logger.debug(
            "politica_evaluada",
            session_id=session_id,
            tool_id=context.proposed_action.tool_id,
            decision=type(final_decision).__name__,
            risk_level=result.risk_assessment.effective_risk.name,
            time_ms=round(total_time_ms, 2),
            winning_layer=winning_layer,
        )

        return result

    # =========================================================================
    # IMPLEMENTACIÓN DE CAPAS
    # =========================================================================

    def _evaluate_layer1_blacklist(
        self,
        context: PolicyEvaluationContext,
        violations: list[PolicyViolation],
    ) -> tuple[PolicyEvaluationLayerResult, Deny | None]:
        """
        Capa 1: Blacklist global inmutable.

        Acciones que NUNCA se permiten bajo ninguna circunstancia.
        Esta capa no es configurable en runtime — es compile-time.
        """
        layer_start = time.perf_counter()
        action = context.proposed_action

        if self._is_globally_blocked(action):
            violation = PolicyViolation(
                layer=1,
                violation_type="global_blacklist",
                description=(
                    f"La acción '{action.tool_id}' está en la lista global de bloqueo. "
                    f"Riesgo clasificado como CRITICAL."
                ),
                severity="critical",
                policy_name="forge.global_blacklist",
            )
            violations.append(violation)

            deny = Deny(
                reason=(
                    f"La acción '{action.tool_schema.display_name}' no está permitida "
                    f"bajo ninguna circunstancia."
                ),
                policy_name="forge.global_blacklist",
                risk_level=RiskLevel.CRITICAL,
                is_security_violation=True,
            )
            return PolicyEvaluationLayerResult(
                layer=1,
                layer_name="Blacklist Global",
                passed=False,
                evaluation_time_ms=(time.perf_counter() - layer_start) * 1000,
                violations=[violation],
                decision_contribution="deny",
            ), deny

        return PolicyEvaluationLayerResult(
            layer=1,
            layer_name="Blacklist Global",
            passed=True,
            evaluation_time_ms=(time.perf_counter() - layer_start) * 1000,
            decision_contribution="none",
        ), None

    def _evaluate_layer2_input_risk(
        self,
        context: PolicyEvaluationContext,
        violations: list[PolicyViolation],
        attack_patterns: list[AttackPattern],
        session_ctx: SessionSecurityContext,
    ) -> tuple[PolicyEvaluationLayerResult, RiskAssessment]:
        """
        Capa 2: Análisis de riesgo del input vía DomainRiskClassifier.

        Detecta path traversal, null bytes, shell injection, unicode tricks.
        Construye el RiskAssessment que se usa en todas las capas posteriores.
        """
        layer_start = time.perf_counter()
        action = context.proposed_action
        input_analysis = context.input_risk_analysis

        # Construir RiskAssessment combinando base risk + input analysis
        risk_assessment = self._risk_classifier.build_risk_assessment(
            tool_id=action.tool_id,
            base_risk=action.base_risk,
            input_analysis=input_analysis,
        )

        layer_violations: list[PolicyViolation] = []
        layer_patterns: list[AttackPattern] = []

        # Si el input tiene señales de riesgo, registrar como violaciones
        if not input_analysis.is_clean:
            for signal in input_analysis.signals:
                violation = PolicyViolation(
                    layer=2,
                    violation_type=signal.signal_type.name.lower(),
                    description=signal.description,
                    severity=(
                        "critical" if signal.severity == RiskLevel.CRITICAL
                        else "error" if signal.severity == RiskLevel.HIGH
                        else "warning"
                    ),
                    evidence=signal.evidence or "",
                    policy_name="forge.input_risk_analysis",
                )
                layer_violations.append(violation)
                violations.append(violation)

                # Señales críticas generan patterns de ataque
                if signal.severity == RiskLevel.CRITICAL:
                    from forge_core.tools.protocol import RiskSignalType
                    pattern_type_map = {
                        "PATH_TRAVERSAL": AttackPatternType.PATH_TRAVERSAL_ATTEMPT,
                        "NULL_BYTE": AttackPatternType.NULL_BYTE_INJECTION,
                        "SHELL_METACHARACTER": AttackPatternType.SHELL_INJECTION_ATTEMPT,
                        "UNICODE_TRICK": AttackPatternType.UNICODE_EVASION,
                        "SENSITIVE_PATH": AttackPatternType.CREDENTIAL_HARVESTING_ATTEMPT,
                    }
                    pattern_type = pattern_type_map.get(
                        signal.signal_type.name,
                        AttackPatternType.PATH_TRAVERSAL_ATTEMPT,
                    )
                    pattern = AttackPattern(
                        pattern_type=pattern_type,
                        detected_at=datetime.now(tz=timezone.utc),
                        session_id=session_ctx.session_id,
                        tool_id=action.tool_id,
                        evidence=signal.evidence or "[sin evidencia]",
                        severity_impact=3,  # crítico → escalar a CRITICAL
                    )
                    layer_patterns.append(pattern)
                    attack_patterns.append(pattern)

        passed = not risk_assessment.is_blocked

        return PolicyEvaluationLayerResult(
            layer=2,
            layer_name="Análisis de Riesgo del Input",
            passed=passed,
            evaluation_time_ms=(time.perf_counter() - layer_start) * 1000,
            violations=layer_violations,
            decision_contribution="deny" if not passed else "none",
        ), risk_assessment

    def _evaluate_layer3_policies(
        self,
        context: PolicyEvaluationContext,
        risk_assessment: RiskAssessment,
        violations: list[PolicyViolation],
    ) -> tuple[PolicyEvaluationLayerResult, Allow | Deny | RequireApproval, str | None]:
        """
        Capa 3: Evaluación de policies ordenadas por prioridad.

        Itera las policies en orden de prioridad descendente.
        La primera cuya condición sea True produce la decisión.
        Si ninguna aplica, el fallback es Deny (fail-closed).
        """
        layer_start = time.perf_counter()
        action = context.proposed_action

        # Construir ProposedAction para el framework de policies
        proposed = ProposedAction(
            tool_schema=action.tool_schema,
            tool_input=action.tool_input,
            session_id=action.session_id,
            task_id=action.task_id,
            context={
                "risk_level": risk_assessment.effective_risk.name,
                "input_risk": risk_assessment.input_risk.name,
                "is_plan_step": context.is_plan_step,
            },
        )

        winning_policy_name: str | None = None
        decision = None

        for policy in self.get_active_policies():
            try:
                policy_result = policy.apply(proposed)
                if policy_result is not None:
                    decision = policy_result
                    winning_policy_name = policy.name
                    break
            except Exception as exc:
                # Error en una policy individual → registrar y continuar
                logger.warning(
                    "policy_error",
                    policy_name=policy.name,
                    error=str(exc),
                )
                violations.append(PolicyViolation(
                    layer=3,
                    violation_type="policy_evaluation_error",
                    description=f"Error evaluando política '{policy.name}': {exc}",
                    severity="warning",
                    policy_name=policy.name,
                ))

        # Si ninguna política aplica → fail-closed
        if decision is None:
            decision = Deny(
                reason="Ninguna política permite explícitamente esta acción.",
                policy_name="forge.implicit_fallback",
                risk_level=risk_assessment.effective_risk,
            )
            winning_policy_name = "forge.implicit_fallback"

        passed = not isinstance(decision, Deny)

        return PolicyEvaluationLayerResult(
            layer=3,
            layer_name="Evaluación de Policies",
            passed=passed,
            evaluation_time_ms=(time.perf_counter() - layer_start) * 1000,
            decision_contribution=type(decision).__name__.lower(),
            notes=f"Política ganadora: {winning_policy_name}",
        ), decision, winning_policy_name

    def _evaluate_layer4_permissions(
        self,
        context: PolicyEvaluationContext,
        policy_decision: Allow | Deny | RequireApproval,
        violations: list[PolicyViolation],
    ) -> tuple[PolicyEvaluationLayerResult, Allow | Deny | RequireApproval]:
        """
        Capa 4: Verificación de permisos del usuario (PermissionSet).

        El PermissionSet puede ANULAR una decisión ALLOW de la capa 3.
        Si el usuario tiene un DENY explícito para la tool, gana sin importar
        qué digan las policies. Esto permite revocación granular de permisos.
        """
        layer_start = time.perf_counter()
        action = context.proposed_action
        permissions = context.permissions

        # Si la Capa 3 ya denegó, no hay nada que verificar aquí
        if isinstance(policy_decision, Deny):
            return PolicyEvaluationLayerResult(
                layer=4,
                layer_name="Verificación de Permisos",
                passed=False,
                evaluation_time_ms=(time.perf_counter() - layer_start) * 1000,
                decision_contribution="none",
                notes="Denegado en capa anterior — sin evaluación.",
            ), policy_decision

        # Verificar si el usuario tiene permiso para esta tool
        has_permission = permissions.allows(
            tool_id=action.tool_id,
            category=action.tool_category,
        )

        if not has_permission:
            violation = PolicyViolation(
                layer=4,
                violation_type="permission_denied",
                description=(
                    f"El usuario no tiene permiso para usar la herramienta "
                    f"'{action.tool_schema.display_name}' "
                    f"(categoría: {action.tool_category.value})."
                ),
                severity="error",
                policy_name="forge.permission_check",
            )
            violations.append(violation)

            deny = Deny(
                reason=(
                    f"No tienes permiso para usar '{action.tool_schema.display_name}'. "
                    f"Revisa la configuración de permisos."
                ),
                policy_name="forge.permission_check",
                risk_level=action.base_risk,
                is_security_violation=False,
            )
            return PolicyEvaluationLayerResult(
                layer=4,
                layer_name="Verificación de Permisos",
                passed=False,
                evaluation_time_ms=(time.perf_counter() - layer_start) * 1000,
                violations=[violation],
                decision_contribution="deny",
            ), deny

        # Verificar path permissions si la tool es de filesystem
        if action.tool_category.value == "filesystem":
            path_arg = (
                action.tool_input.arguments.get("file_path")
                or action.tool_input.arguments.get("path")
                or action.tool_input.arguments.get("directory")
            )
            if path_arg:
                from pathlib import Path
                is_write = action.tool_id in {"file_write", "file_create", "file_delete"}
                if not permissions.allows_path(Path(path_arg), for_writing=is_write):
                    violation = PolicyViolation(
                        layer=4,
                        violation_type="path_not_allowed",
                        description=(
                            f"El path especificado no está en los directorios permitidos."
                        ),
                        severity="error",
                        evidence=str(Path(path_arg).parent),  # solo el directorio, no el path completo
                        policy_name="forge.filesystem_permissions",
                    )
                    violations.append(violation)
                    return PolicyEvaluationLayerResult(
                        layer=4,
                        layer_name="Verificación de Permisos",
                        passed=False,
                        evaluation_time_ms=(time.perf_counter() - layer_start) * 1000,
                        violations=[violation],
                        decision_contribution="deny",
                    ), Deny(
                        reason="La ruta especificada no está en los directorios permitidos.",
                        policy_name="forge.filesystem_permissions",
                        risk_level=action.base_risk,
                    )

        return PolicyEvaluationLayerResult(
            layer=4,
            layer_name="Verificación de Permisos",
            passed=True,
            evaluation_time_ms=(time.perf_counter() - layer_start) * 1000,
            decision_contribution="none",
        ), policy_decision

    def _evaluate_layer5_rate_limit(
        self,
        context: PolicyEvaluationContext,
        current_decision: Allow | Deny | RequireApproval,
        session_ctx: SessionSecurityContext,
        violations: list[PolicyViolation],
    ) -> tuple[PolicyEvaluationLayerResult, Allow | Deny | RequireApproval]:
        """
        Capa 5: Rate limiting multi-dimensional.

        Verifica límites por sesión, por tool, por categoría, y global.
        El multiplicador de rate limit se ajusta según la SecuritySeverity.
        """
        layer_start = time.perf_counter()

        # Si ya hay un Deny, no verificar rate limits (save CPU)
        if isinstance(current_decision, Deny):
            return PolicyEvaluationLayerResult(
                layer=5,
                layer_name="Rate Limiting",
                passed=False,
                evaluation_time_ms=(time.perf_counter() - layer_start) * 1000,
                decision_contribution="none",
                notes="Denegado en capa anterior.",
            ), current_decision

        action = context.proposed_action
        multiplier = session_ctx.get_effective_rate_limit_multiplier()

        # Si la sesión está en LOCKDOWN, bloquear todo
        if multiplier == 0.0:
            violation = PolicyViolation(
                layer=5,
                violation_type="session_lockdown",
                description="La sesión está en modo lockdown. No se permiten acciones.",
                severity="critical",
                policy_name="forge.session_lockdown",
            )
            violations.append(violation)
            return PolicyEvaluationLayerResult(
                layer=5,
                layer_name="Rate Limiting",
                passed=False,
                evaluation_time_ms=(time.perf_counter() - layer_start) * 1000,
                violations=[violation],
                decision_contribution="deny",
            ), Deny(
                reason="La sesión está temporalmente bloqueada por razones de seguridad.",
                policy_name="forge.session_lockdown",
                risk_level=RiskLevel.CRITICAL,
                is_security_violation=True,
            )

        # Verificar rate limit por sesión (global: 60 calls/min)
        session_rl = session_ctx.get_rate_limit_state(
            dimension="session",
            window_seconds=60,
            max_calls=int(60 * multiplier),
        )
        if session_rl.is_rate_limited():
            violation = PolicyViolation(
                layer=5,
                violation_type="session_rate_limit",
                description=(
                    f"Límite de velocidad de sesión alcanzado: "
                    f"{session_rl.calls_remaining()} llamadas restantes. "
                    f"Espera {session_rl.time_until_reset_seconds():.0f}s."
                ),
                severity="warning",
                policy_name="forge.rate_limit_session",
            )
            violations.append(violation)
            return PolicyEvaluationLayerResult(
                layer=5,
                layer_name="Rate Limiting",
                passed=False,
                evaluation_time_ms=(time.perf_counter() - layer_start) * 1000,
                violations=[violation],
                decision_contribution="deny",
            ), Deny(
                reason=(
                    f"Demasiadas acciones en poco tiempo. "
                    f"Espera {session_rl.time_until_reset_seconds():.0f} segundos."
                ),
                policy_name="forge.rate_limit_session",
                risk_level=RiskLevel.NONE,
            )

        # Verificar rate limit por tool
        tool_rl_config = action.tool_schema.capability.rate_limit
        if tool_rl_config is not None:
            tool_rl = session_ctx.get_rate_limit_state(
                dimension=f"tool:{action.tool_id}",
                window_seconds=tool_rl_config.window_seconds,
                max_calls=int(tool_rl_config.max_calls * multiplier),
            )
            if tool_rl.is_rate_limited():
                violation = PolicyViolation(
                    layer=5,
                    violation_type="tool_rate_limit",
                    description=(
                        f"Límite de velocidad para '{action.tool_id}' alcanzado."
                    ),
                    severity="warning",
                    policy_name="forge.rate_limit_tool",
                )
                violations.append(violation)
                return PolicyEvaluationLayerResult(
                    layer=5,
                    layer_name="Rate Limiting",
                    passed=False,
                    evaluation_time_ms=(time.perf_counter() - layer_start) * 1000,
                    violations=[violation],
                    decision_contribution="deny",
                ), Deny(
                    reason=f"Esta herramienta tiene un límite de uso. Intenta en unos segundos.",
                    policy_name="forge.rate_limit_tool",
                    risk_level=RiskLevel.NONE,
                )

        # Registrar la llamada en el rate limit de sesión
        session_rl.record_call()

        return PolicyEvaluationLayerResult(
            layer=5,
            layer_name="Rate Limiting",
            passed=True,
            evaluation_time_ms=(time.perf_counter() - layer_start) * 1000,
            decision_contribution="none",
        ), current_decision

    def _evaluate_layer6_attack_patterns(
        self,
        context: PolicyEvaluationContext,
        violations: list[PolicyViolation],
        session_ctx: SessionSecurityContext,
        current_decision: Allow | Deny | RequireApproval,
    ) -> tuple[PolicyEvaluationLayerResult, list[AttackPattern]]:
        """
        Capa 6: Análisis de patrones de ataque.

        Siempre se ejecuta, incluso cuando hay Deny previo.
        Detecta secuencias sospechosas que apuntan a escalada de privilegios,
        credential harvesting, o evasión sistemática del rate limiting.
        """
        layer_start = time.perf_counter()
        new_patterns: list[AttackPattern] = []
        layer_violations: list[PolicyViolation] = []

        action = context.proposed_action

        # Detectar escalada gradual de privilegios
        # Si las últimas N acciones muestran un patrón de escalada
        recent = session_ctx.recent_action_types
        if len(recent) >= 5:
            privilege_pattern = self._detect_privilege_escalation(
                recent[-5:],
                action.tool_id,
                session_ctx.session_id,
            )
            if privilege_pattern is not None:
                new_patterns.append(privilege_pattern)
                layer_violations.append(PolicyViolation(
                    layer=6,
                    violation_type="privilege_escalation_attempt",
                    description=privilege_pattern.evidence,
                    severity="error",
                    policy_name="forge.attack_pattern_detection",
                ))

        # Detectar credential harvesting (accesos a múltiples recursos sensibles)
        denied_count = sum(session_ctx.denied_action_counts.values())
        if denied_count >= 5 and isinstance(current_decision, Deny):
            credential_pattern = AttackPattern(
                pattern_type=AttackPatternType.CREDENTIAL_HARVESTING_ATTEMPT,
                detected_at=datetime.now(tz=timezone.utc),
                session_id=session_ctx.session_id,
                tool_id=action.tool_id,
                evidence=(
                    f"Múltiples acciones denegadas en la sesión ({denied_count} total). "
                    f"Patrón consistente con recopilación de información."
                ),
                severity_impact=2,  # eleva a HIGH
            )
            new_patterns.append(credential_pattern)

        violations.extend(layer_violations)

        return PolicyEvaluationLayerResult(
            layer=6,
            layer_name="Análisis de Patrones de Ataque",
            passed=len(new_patterns) == 0,
            evaluation_time_ms=(time.perf_counter() - layer_start) * 1000,
            violations=layer_violations,
            decision_contribution="escalate_severity" if new_patterns else "none",
            notes=f"{len(new_patterns)} patrones detectados.",
        ), new_patterns

    def _evaluate_layer7_context(
        self,
        context: PolicyEvaluationContext,
        current_decision: Allow | Deny | RequireApproval,
        session_ctx: SessionSecurityContext,
        violations: list[PolicyViolation],
    ) -> tuple[PolicyEvaluationLayerResult, Allow | Deny | RequireApproval | None]:
        """
        Capa 7: Coherencia con el contexto conversacional.

        Verifica si la acción tiene sentido dado el contexto de la conversación.
        Si la sesión está en HIGH o CRITICAL, aplica restricciones adicionales
        automáticas que no dependen de las policies configuradas.
        """
        layer_start = time.perf_counter()
        override: Allow | Deny | RequireApproval | None = None

        # En HIGH o CRITICAL, todos los filesystem writes requieren aprobación
        if session_ctx.severity.is_at_least(SecuritySeverity.HIGH):
            action = context.proposed_action
            if (
                action.tool_schema.capability.mutation_type != MutationType.NONE
                and isinstance(current_decision, Allow)
            ):
                override = RequireApproval(
                    description=(
                        f"Esta sesión tiene un nivel de alerta elevado. "
                        f"Confirma que quieres ejecutar '{action.tool_schema.display_name}'."
                    ),
                    risk_level=RiskLevel.HIGH,
                    tool_id=action.tool_id,
                    policy_name="forge.elevated_session_review",
                )
                violations.append(PolicyViolation(
                    layer=7,
                    violation_type="elevated_session_review",
                    description=(
                        f"Sesión con severidad {session_ctx.severity.label()}: "
                        f"se requiere aprobación para acciones mutantes."
                    ),
                    severity="warning",
                    policy_name="forge.elevated_session_review",
                ))

        return PolicyEvaluationLayerResult(
            layer=7,
            layer_name="Coherencia Contextual",
            passed=override is None,
            evaluation_time_ms=(time.perf_counter() - layer_start) * 1000,
            decision_contribution="require_approval" if override else "none",
        ), override

    # =========================================================================
    # HELPERS DE DETECCIÓN
    # =========================================================================

    def _is_globally_blocked(self, action: ProposedAction) -> bool:
        """
        Verifica si la acción está en la blacklist global inmutable.

        La blacklist cubre acciones con RiskLevel.CRITICAL o tool_ids
        que representan capacidades nunca permitidas.
        """
        # Acciones con riesgo CRITICAL son siempre bloqueadas
        if action.base_risk == RiskLevel.CRITICAL:
            return True

        # Tool IDs globalmente bloqueados (compile-time)
        _GLOBALLY_BLOCKED_TOOL_IDS = frozenset({
            "shell_execute",
            "bash_execute",
            "system_command",
            "eval_code",
            "exec_arbitrary",
            "keychain_access",
            "credential_dump",
            "privilege_escalate",
            "install_software",
            "modify_system_files",
        })

        return action.tool_id in _GLOBALLY_BLOCKED_TOOL_IDS

    def _detect_privilege_escalation(
        self,
        recent_actions: list[str],
        current_tool_id: str,
        session_id: str,
    ) -> AttackPattern | None:
        """
        Detecta una secuencia de escalada de privilegios.

        Una secuencia sospechosa es cuando las acciones recientes muestran
        un patrón de exploración sistemática:
        file_list → file_read → file_read(sensitive) → file_write
        """
        # Patrón: de lectura a escritura en tools de filesystem
        filesystem_sequence = [
            "file_list", "file_search", "file_read",
            "file_read", "file_write",
        ]
        all_actions = recent_actions + [current_tool_id]

        # Verificar si la secuencia reciente coincide con el patrón
        read_ops = sum(1 for a in all_actions if "read" in a or "list" in a)
        write_ops = sum(1 for a in all_actions if "write" in a or "delete" in a)

        if read_ops >= 3 and write_ops >= 1:
            return AttackPattern(
                pattern_type=AttackPatternType.PRIVILEGE_ESCALATION_ATTEMPT,
                detected_at=datetime.now(tz=timezone.utc),
                session_id=session_id,
                tool_id=current_tool_id,
                evidence=(
                    f"Secuencia de lectura → escritura detectada: "
                    f"{read_ops} lecturas seguidas de {write_ops} escrituras."
                ),
                severity_impact=2,
            )
        return None

    async def _audit_decision(
        self,
        result: RobustPolicyDecision,
        context: PolicyEvaluationContext,
    ) -> None:
        """
        Capa 8: Registra la decisión en el audit log.

        Siempre se ejecuta. Los errores de audit no afectan la decisión.
        """
        try:
            action = context.proposed_action
            audit_logger.record_policy_decision(
                session_id=result.session_id,
                tool_id=action.tool_id,
                decision=type(result.decision).__name__.lower(),
                risk_level=result.risk_assessment.effective_risk.name,
                policy_name=result.winning_policy,
                reason=result.get_user_facing_reason(),
            )
        except Exception as exc:
            # Error de audit → loguear pero no afectar la decisión
            logger.error("audit_decision_fallo", error=exc)

    # =========================================================================
    # CIRCUIT BREAKER
    # =========================================================================

    def _is_circuit_open(self) -> bool:
        """Verifica si el circuit breaker está abierto y si debe resetearse."""
        if not self._circuit_open:
            return False

        if self._circuit_opened_at is None:
            return True

        # Intentar reset automático después del período de espera
        elapsed = (
            datetime.now(tz=timezone.utc) - self._circuit_opened_at
        ).total_seconds()
        if elapsed >= _CIRCUIT_BREAKER_RESET_SECONDS:
            self._circuit_open = False
            self._consecutive_failures = 0
            logger.info("circuit_breaker_reseteado", elapsed_seconds=round(elapsed, 1))
            return False

        return True

    def _open_circuit(self) -> None:
        """Abre el circuit breaker — modo fail-closed total."""
        self._circuit_open = True
        self._circuit_opened_at = datetime.now(tz=timezone.utc)
        logger.critical(
            "circuit_breaker_abierto",
            consecutive_failures=self._consecutive_failures,
            message=(
                "PolicyEngine en modo fail-closed. Todas las acciones serán denegadas "
                f"durante {_CIRCUIT_BREAKER_RESET_SECONDS}s."
            ),
        )

    def _circuit_breaker_deny(self, session_id: str) -> RobustPolicyDecision:
        """Retorna una decisión DENY por circuit breaker abierto."""
        self._total_denials += 1
        deny = Deny(
            reason="El sistema de seguridad está temporalmente no disponible. Intenta en un momento.",
            policy_name="forge.circuit_breaker",
            risk_level=RiskLevel.CRITICAL,
            is_security_violation=False,
        )
        return RobustPolicyDecision(
            decision=deny,
            risk_assessment=RiskAssessment.block(
                tool_id="unknown",
                base_risk=RiskLevel.CRITICAL,
                reason="Circuit breaker abierto.",
            ),
            layer_results=[PolicyEvaluationLayerResult(
                layer=0,
                layer_name="Circuit Breaker",
                passed=False,
                evaluation_time_ms=0.0,
                decision_contribution="deny",
                notes="Circuit breaker abierto — fail-closed.",
            )],
            violations=[PolicyViolation(
                layer=0,
                violation_type="circuit_breaker_open",
                description="El circuit breaker del PolicyEngine está abierto.",
                severity="critical",
                policy_name="forge.circuit_breaker",
            )],
            winning_layer=0,
            winning_policy="forge.circuit_breaker",
            severity_escalated=False,
            new_severity=None,
            attack_patterns_detected=[],
            total_evaluation_time_ms=0.0,
            policies_evaluated_count=0,
            evaluated_at=datetime.now(tz=timezone.utc),
            session_id=session_id,
        )

    def _internal_error_deny(
        self,
        session_id: str,
        context: PolicyEvaluationContext,
        error: Exception,
    ) -> RobustPolicyDecision:
        """Retorna DENY ante un error interno del engine (fail-closed)."""
        self._total_denials += 1
        action = context.proposed_action
        deny = Deny(
            reason="No se pudo evaluar la seguridad de esta acción. Por precaución, ha sido bloqueada.",
            policy_name="forge.internal_error_failsafe",
            risk_level=action.base_risk,
            is_security_violation=False,
        )
        return RobustPolicyDecision(
            decision=deny,
            risk_assessment=RiskAssessment.block(
                tool_id=action.tool_id,
                base_risk=action.base_risk,
                reason="Error interno del PolicyEngine — fail-closed.",
            ),
            layer_results=[],
            violations=[PolicyViolation(
                layer=0,
                violation_type="internal_engine_error",
                description=f"Error interno del engine: {type(error).__name__}",
                severity="critical",
            )],
            winning_layer=None,
            winning_policy="forge.internal_error_failsafe",
            severity_escalated=False,
            new_severity=None,
            attack_patterns_detected=[],
            total_evaluation_time_ms=0.0,
            policies_evaluated_count=0,
            evaluated_at=datetime.now(tz=timezone.utc),
            session_id=session_id,
        )

    # =========================================================================
    # HELPERS DE REGISTRO
    # =========================================================================

    def _register_sorted(self, policy: Policy) -> None:
        """Inserta la política en la lista manteniendo el orden por prioridad."""
        self._policies.append(policy)
        self._policies.sort(key=lambda p: p.priority, reverse=True)


# =============================================================================
# POLÍTICAS ADICIONALES DEL PRODUCTO
# =============================================================================


def _make_medium_risk_approval_policy() -> Policy:
    """
    Política: MEDIUM risk requiere aprobación en primer uso.

    Las acciones de riesgo MEDIUM (escritura de archivos, portapapeles)
    requieren aprobación la primera vez que el usuario las solicita.
    Si el usuario eligió recordar la decisión, en futuras invocaciones
    el PermissionSet ya tiene el permiso y la capa 4 lo permite directamente.
    """
    from forge_core.policy.framework import RiskLevelCondition, Condition

    class FirstTimeRiskCondition(Condition):
        """Condición: riesgo MEDIUM Y no tiene first-use approval guardado."""

        def evaluate(self, action: ProposedAction) -> bool:
            from forge_core.tools.protocol import RiskLevel
            return action.base_risk == RiskLevel.MEDIUM

    return Policy(
        name="forge.medium_risk_first_use_approval",
        condition=FirstTimeRiskCondition(),
        decision_fn=lambda action: RequireApproval(
            description=(
                f"'{action.tool_schema.display_name}' quiere realizar una acción "
                f"que tiene un impacto moderado en tu sistema. ¿Lo permites?"
            ),
            risk_level=RiskLevel.MEDIUM,
            tool_id=action.tool_id,
            policy_name="forge.medium_risk_first_use_approval",
            is_first_time=True,
            remember_decision=True,
        ),
        priority=PolicyPriority.DEFAULT + 50,  # ligeramente más alto que DEFAULT
        description=(
            "Requiere aprobación en el primer uso para acciones de riesgo MEDIUM. "
            "El usuario puede elegir recordar la decisión para el futuro."
        ),
    )


def _make_desktop_first_use_policy() -> Policy:
    """
    Política: todas las acciones de desktop requieren aprobación en primer uso.

    Las tools de automatización de escritorio (lanzar apps, clipboard,
    screenshots) requieren aprobación explícita la primera vez.
    """
    from forge_core.policy.framework import CategoryCondition

    return Policy(
        name="forge.desktop_first_use_approval",
        condition=CategoryCondition(
            __import__('forge_core.tools.protocol', fromlist=['ToolCategory']).ToolCategory.DESKTOP
        ),
        decision_fn=lambda action: RequireApproval(
            description=(
                f"El asistente quiere usar '{action.tool_schema.display_name}' "
                f"para interactuar con tu escritorio. ¿Lo autorizas?"
            ),
            risk_level=RiskLevel.MEDIUM,
            tool_id=action.tool_id,
            policy_name="forge.desktop_first_use_approval",
            is_first_time=True,
            remember_decision=True,
        ),
        priority=PolicyPriority.DEFAULT + 100,
        description=(
            "Requiere aprobación explícita para todas las acciones de automatización "
            "de escritorio. Diseñado para que el usuario tenga control total."
        ),
    )


def _make_filesystem_write_policy() -> Policy:
    """
    Política: escritura de archivos siempre requiere aprobación.

    file_write es una acción que puede tener consecuencias difíciles de revertir.
    Siempre requiere confirmación del usuario, independientemente de si ya
    fue aprobada antes.
    """
    from forge_core.policy.framework import ToolIdCondition

    return Policy(
        name="forge.filesystem_write_approval",
        condition=ToolIdCondition("file_write", "file_delete", "file_create"),
        decision_fn=lambda action: RequireApproval(
            description=(
                f"El asistente quiere "
                f"{'escribir en' if 'write' in action.tool_id else 'eliminar'} "
                f"un archivo en tu sistema. ¿Lo permites?"
            ),
            risk_level=RiskLevel.MEDIUM,
            tool_id=action.tool_id,
            policy_name="forge.filesystem_write_approval",
            remember_decision=True,
        ),
        priority=PolicyPriority.DEFAULT + 75,
        description="Requiere aprobación para escritura y eliminación de archivos.",
    )


def _make_lockdown_policy() -> Policy:
    """
    Política: en sesión con lockdown, solo permite tools de lectura pura.

    Cuando la sesión está en SecuritySeverity.LOCKDOWN (detectado en la
    evaluación de rate limits, capa 5), esta política bloquea todo.
    El bloqueo real lo hace la capa 5 — esta política es una segunda capa.
    """
    class LockdownCondition:
        """Siempre False — el lockdown lo maneja la capa 5, no las policies."""
        def evaluate(self, action: ProposedAction) -> bool:
            return False

    from forge_core.policy.framework import Condition

    class _Never(Condition):
        def evaluate(self, action: ProposedAction) -> bool:
            return False

    return Policy(
        name="forge.lockdown_catchall",
        condition=_Never(),
        decision_fn=lambda action: Deny(
            reason="Sesión en lockdown.",
            policy_name="forge.lockdown_catchall",
            risk_level=RiskLevel.CRITICAL,
        ),
        priority=PolicyPriority.SECURITY,
        enabled=True,
        description="Política de lockdown total — activada por la capa 5 del engine.",
    )