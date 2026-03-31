"""
DefaultToolDispatch — implementación del ToolDispatchPort.

El ToolDispatchPort es el guardián de la ejecución de tools. Toda tool
que el sistema quiere ejecutar pasa por aquí, donde se aplica el
enforcement completo de seguridad en 10 pasos:

  1. Resolver tool_id → ToolRegistration via ToolRegistryPort
  2. Validar el input contra el schema de la tool
  3. Clasificar riesgo del input (DomainRiskClassifier)
  4. Ejecutar PolicyEngine.evaluate() con contexto completo
  5. Si DENY → retornar result.success=False sin ejecutar
  6. Si REQUIRE_APPROVAL → crear solicitud y esperar decisión del usuario
  7. Si ALLOW → proceder a la ejecución
  8. Ejecutar la tool con timeout enforcement (asyncio.wait_for)
  9. Registrar en audit log y actualizar estadísticas del registry
  10. Retornar ToolExecutionResult completo

Contrato de seguridad (invariantes):
  - dispatch() NUNCA lanza excepciones al caller — siempre retorna result.
  - Si el PolicyEngine falla → DENY (fail-closed).
  - Toda ejecución genera audit entry, independientemente del resultado.
  - El timeout es un hard limit — asyncio.wait_for() lo enforcea.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from forge_core.observability.logging import get_logger
from forge_core.tools.protocol import (
    RiskAssessment,
    RiskLevel,
    ToolExecutionRecord,
    ToolInput,
    ToolOutput,
)

from src.domain.value_objects.identifiers import ApprovalId, InvocationId, SessionId
from src.domain.value_objects.permission import PermissionSet
from src.domain.value_objects.risk import DomainRiskClassifier
from src.ports.outbound.policy_port import (
    ApprovalPort,
    PolicyEnginePort,
    PolicyEvaluationContext,
)
from src.ports.outbound.storage_port import AuditLogPort
from src.ports.outbound.tool_port import (
    ToolDispatchPort,
    ToolExecutionRequest,
    ToolExecutionResult,
    ToolRegistryPort,
)


logger = get_logger(__name__, component="tool_dispatch")

# Timeout por defecto para ejecución de tools (segundos)
_DEFAULT_TOOL_TIMEOUT = 30.0

# Máximo de timeout permitido (segundos)
_MAX_TOOL_TIMEOUT = 120.0


class DefaultToolDispatch(ToolDispatchPort):
    """
    Implementación del ToolDispatchPort con enforcement completo de seguridad.

    Coordina el flujo completo: policy check → approval → ejecución con
    timeout → audit log → estadísticas del registry.

    Es stateless entre invocaciones — el estado de aprobaciones pendientes
    vive en el ApprovalPort.
    """

    def __init__(
        self,
        tool_registry: ToolRegistryPort,
        policy_engine: PolicyEnginePort,
        approval_port: ApprovalPort,
        audit_log: AuditLogPort,
        *,
        default_timeout_seconds: float = _DEFAULT_TOOL_TIMEOUT,
    ) -> None:
        """
        Args:
            tool_registry:           Catálogo de tools disponibles.
            policy_engine:           Motor de evaluación de políticas.
            approval_port:           Gestión del flujo de aprobaciones.
            audit_log:               Puerto del audit log.
            default_timeout_seconds: Timeout por defecto para ejecución.
        """
        self._registry = tool_registry
        self._policy_engine = policy_engine
        self._approval_port = approval_port
        self._audit_log = audit_log
        self._default_timeout = min(default_timeout_seconds, _MAX_TOOL_TIMEOUT)
        self._risk_classifier = DomainRiskClassifier()

    # =========================================================================
    # ToolDispatchPort — implementación
    # =========================================================================

    async def dispatch(
        self,
        request: ToolExecutionRequest,
        permissions: PermissionSet,
    ) -> ToolExecutionResult:
        """
        Ejecuta una tool con enforcement completo de seguridad.

        Nunca lanza excepciones al caller — los errores van en el resultado.

        Args:
            request:     Solicitud de ejecución con toda la metadata.
            permissions: PermissionSet del usuario activo.

        Returns:
            ToolExecutionResult con el resultado completo.
        """
        start_time = time.perf_counter()
        session_id_str = request.session_id.to_str()

        try:
            return await self._dispatch_internal(
                request, permissions, start_time, session_id_str
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "dispatch_error_inesperado",
                tool_id=request.tool_id,
                session_id=session_id_str,
                error=str(exc),
            )
            return self._build_error_result(
                request=request,
                error_code="FORGE_DISPATCH_UNEXPECTED_ERROR",
                error_message=f"Error inesperado en la ejecución: {type(exc).__name__}",
                duration_ms=duration_ms,
            )

    async def dispatch_batch(
        self,
        requests: list[ToolExecutionRequest],
        permissions: PermissionSet,
    ) -> list[ToolExecutionResult]:
        """
        Ejecuta múltiples tools secuencialmente.

        La ejecución continúa aunque alguna tool falle — el caller recibe
        todos los resultados y decide qué hacer con los fallos.

        Args:
            requests:    Lista de solicitudes en orden.
            permissions: PermissionSet del usuario activo.

        Returns:
            Lista de ToolExecutionResult en el mismo orden que requests.
        """
        results = []
        for req in requests:
            result = await self.dispatch(req, permissions)
            results.append(result)
        return results

    async def validate_input(
        self,
        tool_id: str,
        arguments: dict[str, Any],
    ) -> list[str]:
        """
        Valida el input de una tool contra su schema sin ejecutarla.

        Args:
            tool_id:   ID de la tool a validar.
            arguments: Argumentos a validar.

        Returns:
            Lista de errores de validación. Vacía si el input es válido.
        """
        registration = self._registry.get(tool_id)
        if registration is None:
            return [f"Tool '{tool_id}' no encontrada en el registro."]

        errors: list[str] = []
        schema = registration.schema

        # Verificar campos requeridos según el schema de la tool
        if hasattr(schema, "required_arguments"):
            for required_field in schema.required_arguments:
                if required_field not in arguments:
                    errors.append(
                        f"Campo requerido faltante: '{required_field}'"
                    )

        # Verificar tipos básicos si el schema los define
        if hasattr(schema, "argument_types") and schema.argument_types:
            for field, expected_type in schema.argument_types.items():
                if field in arguments:
                    value = arguments[field]
                    if expected_type == "string" and not isinstance(value, str):
                        errors.append(
                            f"El campo '{field}' debe ser un string, "
                            f"recibido: {type(value).__name__}"
                        )
                    elif expected_type == "integer" and not isinstance(value, int):
                        errors.append(
                            f"El campo '{field}' debe ser un entero, "
                            f"recibido: {type(value).__name__}"
                        )
                    elif expected_type == "boolean" and not isinstance(value, bool):
                        errors.append(
                            f"El campo '{field}' debe ser un booleano, "
                            f"recibido: {type(value).__name__}"
                        )

        return errors

    async def classify_risk(
        self,
        tool_id: str,
        arguments: dict[str, Any],
        session_id: SessionId,
    ) -> RiskAssessment:
        """
        Clasifica el riesgo de una ejecución sin ejecutarla.

        Útil para que el LightweightPlanner anticipe qué steps del plan
        requerirán aprobación antes de ejecutar nada.

        Args:
            tool_id:    ID de la tool.
            arguments:  Argumentos con los que se ejecutaría.
            session_id: ID de la sesión para contexto.

        Returns:
            RiskAssessment con el riesgo efectivo.
        """
        registration = self._registry.get(tool_id)
        if registration is None:
            return RiskAssessment.block(
                tool_id=tool_id,
                base_risk=RiskLevel.CRITICAL,
                reason=f"Tool '{tool_id}' no encontrada.",
            )

        args_text = " ".join(str(v) for v in arguments.values())
        input_analysis = self._risk_classifier.analyze(args_text)

        session_ctx = self._policy_engine.get_session_security_context(
            session_id.to_str()
        )

        return await self._policy_engine.classify_risk(
            proposed_action=self._build_proposed_action(
                registration, arguments, session_id
            ),
            session_security=session_ctx,
            input_analysis=input_analysis,
        )

    async def get_pending_approvals(
        self,
        session_id: SessionId,
    ) -> list[dict[str, Any]]:
        """
        Retorna las aprobaciones pendientes para una sesión.

        Args:
            session_id: ID de la sesión.

        Returns:
            Lista de dicts con metadata de cada aprobación pendiente.
        """
        pending = await self._approval_port.get_pending(session_id)
        return [req.to_ui_dict() for req in pending]

    async def resolve_approval(
        self,
        approval_id: ApprovalId,
        granted: bool,
    ) -> None:
        """
        Resuelve una aprobación pendiente.

        Args:
            approval_id: ID de la aprobación a resolver.
            granted:     True si el usuario aprobó, False si denegó.
        """
        await self._approval_port.resolve(approval_id, granted=granted)

    # =========================================================================
    # FLUJO INTERNO DE DISPATCH
    # =========================================================================

    async def _dispatch_internal(
        self,
        request: ToolExecutionRequest,
        permissions: PermissionSet,
        start_time: float,
        session_id_str: str,
    ) -> ToolExecutionResult:
        """
        Implementación interna del dispatch — puede lanzar excepciones.
        Capturadas y convertidas en result.success=False en dispatch().
        """

        # ─── Paso 1: Resolver tool_id → ToolRegistration ────────────────────
        registration = self._registry.get(request.tool_id)
        if registration is None:
            registration = self._registry.get_by_llm_name(request.tool_id)

        if registration is None:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                "tool_no_encontrada",
                tool_id=request.tool_id,
                session_id=session_id_str,
            )
            return self._build_error_result(
                request=request,
                error_code="FORGE_TOOL_NOT_FOUND",
                error_message=f"La herramienta '{request.tool_id}' no está disponible.",
                duration_ms=duration_ms,
            )

        schema = registration.schema

        # ─── Paso 2: Validar input ───────────────────────────────────────────
        if not request.skip_policy_check:
            validation_errors = await self.validate_input(
                request.tool_id, request.arguments
            )
            if validation_errors:
                duration_ms = (time.perf_counter() - start_time) * 1000
                return self._build_error_result(
                    request=request,
                    error_code="FORGE_TOOL_INPUT_INVALID",
                    error_message="; ".join(validation_errors[:3]),
                    duration_ms=duration_ms,
                )

        # ─── Paso 3: Clasificar riesgo del input ────────────────────────────
        args_text = " ".join(str(v) for v in request.arguments.values())
        input_analysis = self._risk_classifier.analyze(args_text)

        risk_assessment = self._risk_classifier.build_risk_assessment(
            tool_id=request.tool_id,
            base_risk=schema.capability.risk_classification,
            input_analysis=input_analysis,
        )

        # ─── Paso 4: Policy check ────────────────────────────────────────────
        policy_decision_obj = None
        if not request.skip_policy_check:
            session_ctx = self._policy_engine.get_session_security_context(
                session_id_str
            )

            from forge_core.policy.framework import ProposedAction
            proposed_action = self._build_proposed_action(
                registration, request.arguments, request.session_id
            )

            from src.domain.value_objects.risk import InputRiskAnalysis
            eval_context = PolicyEvaluationContext(
                proposed_action=proposed_action,
                session_security=session_ctx,
                permissions=permissions,
                input_risk_analysis=input_analysis,
                is_plan_step=request.step_id is not None,
            )

            policy_result = await self._policy_engine.evaluate(eval_context)
            policy_decision_obj = policy_result

            # ─── Paso 5: Si DENY → retornar sin ejecutar ────────────────────
            if policy_result.is_blocked:
                duration_ms = (time.perf_counter() - start_time) * 1000
                user_reason = policy_result.get_user_facing_reason()

                # Audit log de denegación
                await self._record_audit(
                    request=request,
                    risk_assessment=risk_assessment,
                    policy_decision="deny",
                    success=False,
                    duration_ms=duration_ms,
                    error_code="FORGE_POLICY_DENIED",
                    is_security=policy_result.is_security_violation,
                    policy_name=policy_result.winning_policy,
                )

                return self._build_error_result(
                    request=request,
                    error_code="FORGE_POLICY_DENIED",
                    error_message=user_reason,
                    duration_ms=duration_ms,
                    risk_assessment=risk_assessment,
                    was_policy_denied=True,
                )

            # ─── Paso 6: Si REQUIRE_APPROVAL → suspender y esperar ──────────
            if policy_result.requires_approval:
                from forge_core.policy.framework import RequireApproval
                decision_obj = policy_result.decision
                description = getattr(decision_obj, "description", "Esta acción requiere tu confirmación.")

                approval_req = await self._approval_port.request_approval(
                    session_id=request.session_id,
                    description=description,
                    risk_level=risk_assessment.effective_risk,
                    tool_id=request.tool_id,
                    tool_display_name=schema.display_name,
                    is_first_time=True,
                    can_remember_decision=True,
                    timeout_seconds=300.0,
                )

                approval_decision = await self._approval_port.await_decision(
                    approval_req.approval_id
                )

                if not approval_decision.granted:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    reason = (
                        "La acción fue cancelada (tiempo expirado)."
                        if approval_decision.decided_by == "timeout"
                        else "El usuario no aprobó la acción."
                    )

                    await self._record_audit(
                        request=request,
                        risk_assessment=risk_assessment,
                        policy_decision="require_approval",
                        success=False,
                        duration_ms=duration_ms,
                        error_code="FORGE_APPROVAL_DENIED",
                        approval_id=approval_req.approval_id,
                    )

                    return self._build_error_result(
                        request=request,
                        error_code="FORGE_APPROVAL_DENIED",
                        error_message=reason,
                        duration_ms=duration_ms,
                        risk_assessment=risk_assessment,
                        approval_id=approval_req.approval_id,
                    )

                # Aprobado — continuar con la ejecución
                request = ToolExecutionRequest(
                    tool_id=request.tool_id,
                    arguments=request.arguments,
                    session_id=request.session_id,
                    invocation_id=request.invocation_id,
                    task_id=request.task_id,
                    step_id=request.step_id,
                    timeout_override_seconds=request.timeout_override_seconds,
                    skip_policy_check=True,
                    approval_id=approval_req.approval_id,
                )

        # ─── Paso 7: Ejecutar la tool con timeout ────────────────────────────
        timeout = min(
            request.timeout_override_seconds or self._default_timeout,
            _MAX_TOOL_TIMEOUT,
        )

        tool_input = request.to_tool_input()
        tool_port = self._get_tool_port(registration)

        if tool_port is None:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return self._build_error_result(
                request=request,
                error_code="FORGE_TOOL_NO_EXECUTOR",
                error_message=f"No hay implementación registrada para '{request.tool_id}'.",
                duration_ms=duration_ms,
                risk_assessment=risk_assessment,
            )

        try:
            tool_output = await asyncio.wait_for(
                tool_port.execute(tool_input),
                timeout=timeout,
            )
            success = True
            error_code = None
            error_message = None

        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                "tool_timeout",
                tool_id=request.tool_id,
                timeout_seconds=timeout,
                session_id=session_id_str,
            )

            await self._record_audit(
                request=request,
                risk_assessment=risk_assessment,
                policy_decision="allow",
                success=False,
                duration_ms=duration_ms,
                error_code="FORGE_TOOL_TIMEOUT",
            )
            self._registry.record_invocation(request.tool_id, success=False)

            return self._build_error_result(
                request=request,
                error_code="FORGE_TOOL_TIMEOUT",
                error_message=(
                    f"La herramienta '{schema.display_name}' tardó demasiado. "
                    f"Intenta con una solicitud más simple."
                ),
                duration_ms=duration_ms,
                risk_assessment=risk_assessment,
            )

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            error_msg = str(exc)
            logger.error(
                "tool_ejecucion_error",
                tool_id=request.tool_id,
                error=error_msg,
            )

            await self._record_audit(
                request=request,
                risk_assessment=risk_assessment,
                policy_decision="allow",
                success=False,
                duration_ms=duration_ms,
                error_code="FORGE_TOOL_EXECUTION_ERROR",
            )
            self._registry.record_invocation(request.tool_id, success=False)

            return self._build_error_result(
                request=request,
                error_code="FORGE_TOOL_EXECUTION_ERROR",
                error_message=f"Error al ejecutar la herramienta: {error_msg[:200]}",
                duration_ms=duration_ms,
                risk_assessment=risk_assessment,
            )

        # ─── Paso 9: Audit log + actualizar registry ─────────────────────────
        duration_ms = (time.perf_counter() - start_time) * 1000

        await self._record_audit(
            request=request,
            risk_assessment=risk_assessment,
            policy_decision="allow",
            success=True,
            duration_ms=duration_ms,
            approval_id=request.approval_id,
        )
        self._registry.record_invocation(request.tool_id, success=True)

        # ─── Paso 10: Construir y retornar ToolExecutionResult ───────────────
        execution_record = ToolExecutionRecord(
            invocation_id=request.invocation_id.to_str(),
            tool_id=request.tool_id,
            session_id=session_id_str,
            started_at=datetime.now(tz=timezone.utc),
            duration_ms=duration_ms,
            success=True,
        )

        logger.info(
            "tool_ejecutada_ok",
            tool_id=request.tool_id,
            session_id=session_id_str,
            duration_ms=round(duration_ms, 2),
            risk=risk_assessment.effective_risk.name,
        )

        return ToolExecutionResult(
            invocation_id=request.invocation_id,
            tool_id=request.tool_id,
            success=True,
            output=tool_output,
            risk_assessment=risk_assessment,
            execution_record=execution_record,
            approval_id=request.approval_id,
            sandbox_used=False,
            duration_ms=duration_ms,
        )

    # =========================================================================
    # HELPERS INTERNOS
    # =========================================================================

    def _get_tool_port(self, registration: Any) -> Any | None:
        """
        Obtiene la instancia del ToolPort desde el registro.

        En V1, el ToolPort está almacenado directamente en el schema
        como atributo 'executor'. En V2, se usaría un ServiceLocator.

        Args:
            registration: ToolRegistration con el schema de la tool.

        Returns:
            Instancia del ToolPort, o None si no está disponible.
        """
        return getattr(registration.schema, "executor", None)

    def _build_proposed_action(
        self,
        registration: Any,
        arguments: dict[str, Any],
        session_id: SessionId,
    ) -> Any:
        """
        Construye un ProposedAction del framework de policies.

        Args:
            registration: ToolRegistration de la tool.
            arguments:    Argumentos de la invocación.
            session_id:   ID de la sesión activa.

        Returns:
            ProposedAction para el PolicyEngine.
        """
        from forge_core.policy.framework import ProposedAction
        from forge_core.tools.protocol import ToolInput

        tool_input = ToolInput(
            tool_id=registration.tool_id,
            invocation_id=str(InvocationId.generate()),
            arguments=arguments,
            session_id=session_id.to_str(),
        )

        return ProposedAction(
            tool_schema=registration.schema,
            tool_input=tool_input,
            session_id=session_id.to_str(),
        )

    async def _record_audit(
        self,
        *,
        request: ToolExecutionRequest,
        risk_assessment: RiskAssessment,
        policy_decision: str,
        success: bool,
        duration_ms: float,
        error_code: str | None = None,
        approval_id: ApprovalId | None = None,
        is_security: bool = False,
        policy_name: str | None = None,
    ) -> None:
        """
        Registra la ejecución en el audit log.

        Los errores de audit no afectan el resultado de la ejecución.

        Args:
            request:         Solicitud de ejecución original.
            risk_assessment: Evaluación de riesgo aplicada.
            policy_decision: Decisión de la policy ('allow', 'deny', etc.).
            success:         Si la ejecución fue exitosa.
            duration_ms:     Duración de la ejecución.
            error_code:      Código de error si falló.
            approval_id:     ID de aprobación si fue requerida.
            is_security:     Si fue un evento de seguridad.
            policy_name:     Nombre de la política que decidió.
        """
        try:
            await self._audit_log.record_tool_execution(
                session_id=request.session_id.to_str(),
                tool_id=request.tool_id,
                risk_level=risk_assessment.effective_risk.name.lower(),
                policy_decision=policy_decision,
                success=success,
                duration_ms=duration_ms,
                sandbox_used=False,
                approval_id=approval_id.to_str() if approval_id else None,
                error_code=error_code,
                policy_name=policy_name,
            )
        except Exception as audit_exc:
            logger.error(
                "audit_record_fallo",
                tool_id=request.tool_id,
                error=str(audit_exc),
            )

    def _build_error_result(
        self,
        *,
        request: ToolExecutionRequest,
        error_code: str,
        error_message: str,
        duration_ms: float,
        risk_assessment: RiskAssessment | None = None,
        approval_id: ApprovalId | None = None,
        was_policy_denied: bool = False,
    ) -> ToolExecutionResult:
        """
        Construye un ToolExecutionResult de error.

        Args:
            request:           Solicitud original.
            error_code:        Código de error ForgeError.
            error_message:     Mensaje legible del error.
            duration_ms:       Duración hasta el error.
            risk_assessment:   Evaluación de riesgo (puede ser None).
            approval_id:       ID de aprobación si aplica.
            was_policy_denied: True si fue denegado por policy.

        Returns:
            ToolExecutionResult con success=False.
        """
        if risk_assessment is None:
            risk_assessment = RiskAssessment.allow_direct(
                tool_id=request.tool_id,
                base_risk=RiskLevel.NONE,
            )

        execution_record = ToolExecutionRecord(
            invocation_id=request.invocation_id.to_str(),
            tool_id=request.tool_id,
            session_id=request.session_id.to_str(),
            started_at=datetime.now(tz=timezone.utc),
            duration_ms=duration_ms,
            success=False,
            error_code=error_code,
        )

        return ToolExecutionResult(
            invocation_id=request.invocation_id,
            tool_id=request.tool_id,
            success=False,
            output=None,
            risk_assessment=risk_assessment,
            execution_record=execution_record,
            approval_id=approval_id,
            sandbox_used=False,
            error_code=error_code,
            error_message=error_message,
            duration_ms=duration_ms,
        )