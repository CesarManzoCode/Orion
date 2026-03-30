"""
Puertos de salida de tools de HiperForge User.

Este módulo define los contratos de integración del sistema con las tools
del agente. Extiende y especializa el protocolo base de forge_core con
operaciones de alto nivel alineadas a los flujos de negocio del producto.

Separación de responsabilidades:

  forge_core/tools/protocol.py:
    - ToolPort: contrato que cada tool individual implementa
    - ToolSchema, ToolCapability, RiskAssessment: tipos de datos de tools

  este módulo:
    - ToolRegistryPort: gestión del catálogo de tools disponibles
    - ToolDispatchPort: ejecución de tools con enforcement completo
      (policy check → approval → sandbox → ejecución → audit)

Los dos puertos tienen roles distintos:
  - ToolRegistryPort: "¿Qué tools existen? ¿Qué schemas tienen?"
  - ToolDispatchPort: "Ejecuta esta tool con este input, ahora."

El TaskExecutor depende de ToolDispatchPort para la ejecución.
El ContextBuilder depende de ToolRegistryPort para obtener los schemas.
Ninguno de los dos conoce las implementaciones concretas.

Jerarquía de tipos:

  ToolFilter           — criterios de filtrado del registry
  ToolRegistration     — registro de una tool en el registry
  ToolExecutionRequest — solicitud de ejecución de una tool
  ToolExecutionResult  — resultado completo de la ejecución
  ToolRegistryPort     — puerto abstracto de gestión del catálogo
  ToolDispatchPort     — puerto abstracto de ejecución
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from forge_core.llm.protocol import ToolDefinition
from forge_core.tools.protocol import (
    Platform,
    RiskAssessment,
    RiskLevel,
    ToolCategory,
    ToolCapability,
    ToolExecutionRecord,
    ToolInput,
    ToolOutput,
    ToolSchema,
)

from src.domain.value_objects.identifiers import (
    ApprovalId,
    InvocationId,
    SessionId,
    TaskId,
)
from src.domain.value_objects.permission import PermissionSet


# =============================================================================
# TIPOS DE FILTRADO DEL REGISTRY
# =============================================================================


@dataclass(frozen=True)
class ToolFilter:
    """
    Criterios de filtrado para consultas al ToolRegistry.

    Permite obtener subconjuntos del catálogo de tools según categoría,
    nivel de riesgo, plataforma, o estado de habilitación.

    Todos los campos son opcionales — un ToolFilter vacío retorna todas las tools.
    """

    categories: frozenset[ToolCategory] | None = None
    """Si se especifica, solo retorna tools de estas categorías."""

    max_risk_level: RiskLevel | None = None
    """Si se especifica, solo retorna tools con riesgo <= este nivel."""

    platform: Platform | None = None
    """Si se especifica, solo retorna tools compatibles con esta plataforma."""

    enabled_only: bool = True
    """Si True (default), solo retorna tools habilitadas."""

    allowed_by_permissions: PermissionSet | None = None
    """
    Si se especifica, filtra por las tools que el usuario tiene permiso de usar.
    Excluye tools con AccessLevel.DENY en el PermissionSet del usuario.
    """

    @classmethod
    def for_llm_context(
        cls,
        platform: Platform,
        permissions: PermissionSet,
    ) -> ToolFilter:
        """
        Factory: crea un filtro para obtener las tools a enviar al LLM.

        Retorna solo las tools habilitadas, compatibles con la plataforma,
        y que el usuario tiene permiso de usar (al menos READ).

        Args:
            platform:    Plataforma actual del sistema.
            permissions: PermissionSet del usuario activo.

        Returns:
            ToolFilter configurado para el context del LLM.
        """
        return cls(
            platform=platform,
            enabled_only=True,
            allowed_by_permissions=permissions,
        )

    @classmethod
    def by_category(cls, *categories: ToolCategory) -> ToolFilter:
        """
        Factory: crea un filtro por categorías específicas.

        Args:
            *categories: Categorías a incluir.

        Returns:
            ToolFilter con las categorías especificadas.
        """
        return cls(
            categories=frozenset(categories),
            enabled_only=True,
        )

    @classmethod
    def safe_only(cls) -> ToolFilter:
        """
        Factory: crea un filtro que solo retorna tools sin riesgo.

        Útil para contextos donde solo se quieren tools de lectura pura.

        Returns:
            ToolFilter con riesgo máximo NONE.
        """
        return cls(
            max_risk_level=RiskLevel.NONE,
            enabled_only=True,
        )


# =============================================================================
# REGISTRO DE UNA TOOL
# =============================================================================


@dataclass(frozen=True)
class ToolRegistration:
    """
    Entrada en el registro del catálogo de tools.

    Combina el ToolSchema (definición completa de la tool) con metadata
    de registro: cuándo se registró, si está activa, y estadísticas de uso.
    """

    schema: ToolSchema
    """Schema completo de la tool registrada."""

    registered_at: datetime
    """Timestamp de registro en el catálogo."""

    invocation_count: int = 0
    """Número de veces que la tool ha sido invocada en la sesión actual."""

    last_invoked_at: datetime | None = None
    """Timestamp de la última invocación. None si nunca se ha usado."""

    failure_count: int = 0
    """Número de fallos recientes. Usado para circuit breaking."""

    @property
    def tool_id(self) -> str:
        """Atajo al tool_id del schema."""
        return self.schema.tool_id

    @property
    def category(self) -> ToolCategory:
        """Atajo a la categoría del schema."""
        return self.schema.category

    @property
    def capability(self) -> ToolCapability:
        """Atajo a la capability del schema."""
        return self.schema.capability

    @property
    def risk_level(self) -> RiskLevel:
        """Atajo al nivel de riesgo de la capability."""
        return self.schema.capability.risk_classification

    @property
    def is_healthy(self) -> bool:
        """
        True si la tool está en estado saludable (pocos fallos recientes).

        Una tool se considera no saludable si tiene más de 3 fallos
        consecutivos recientes — circuit breaker básico.
        """
        return self.failure_count < 3

    def to_llm_definition(self) -> ToolDefinition:
        """
        Convierte el registro en un ToolDefinition para el contexto del LLM.

        Returns:
            ToolDefinition listo para incluir en LLMContext.tools.
        """
        return self.schema.to_tool_definition()


# =============================================================================
# SOLICITUD Y RESULTADO DE EJECUCIÓN
# =============================================================================


@dataclass(frozen=True)
class ToolExecutionRequest:
    """
    Solicitud de ejecución de una tool al ToolDispatchPort.

    El TaskExecutor construye este objeto con toda la información necesaria
    para que el DispatchPort pueda ejecutar la tool con enforcement completo:
    policy check, approval si necesario, sandbox si aplica.
    """

    tool_id: str
    """ID de la tool a ejecutar."""

    arguments: dict[str, Any]
    """Argumentos para la tool, tal como los generó el LLM."""

    session_id: SessionId
    """ID de la sesión activa."""

    invocation_id: InvocationId
    """ID único de esta invocación (correlaciona con el audit log)."""

    task_id: TaskId | None = None
    """ID de la task padre (si la ejecución es parte de un plan)."""

    step_id: str | None = None
    """ID del step del plan (si aplica)."""

    timeout_override_seconds: float | None = None
    """
    Override del timeout para esta invocación específica.
    None usa el timeout por defecto de la tool.
    """

    skip_policy_check: bool = False
    """
    Si True, omite el policy check (SOLO para re-ejecución post-aprobación).
    El ToolDispatchPort verifica que la invocación tenga un ApprovalId válido
    antes de aceptar skip_policy_check=True.
    """

    approval_id: ApprovalId | None = None
    """
    ID de la aprobación del usuario si esta ejecución fue pre-aprobada.
    Requerido cuando skip_policy_check=True.
    """

    def to_tool_input(self) -> ToolInput:
        """
        Convierte la solicitud en el ToolInput del protocolo de forge_core.

        Returns:
            ToolInput validado listo para pasarle al ToolPort.
        """
        return ToolInput(
            tool_id=self.tool_id,
            invocation_id=self.invocation_id.to_str(),
            arguments=self.arguments,
            session_id=self.session_id.to_str(),
            task_id=self.task_id.to_str() if self.task_id else None,
            timeout_seconds=self.timeout_override_seconds,
        )


@dataclass(frozen=True)
class ToolExecutionResult:
    """
    Resultado completo de la ejecución de una tool vía ToolDispatchPort.

    Encapsula el output de la tool junto con toda la metadata de la ejecución:
    el risk assessment, si se usó sandbox, si se requirió aprobación, y el
    execution record para el audit log.

    El TaskExecutor recibe este objeto después de cada ejecución y lo usa para:
    - Construir la StepExecution de dominio
    - Añadir evidencia al Task
    - Generar el SynthesisContext para el LLM
    """

    invocation_id: InvocationId
    """ID único de la invocación."""

    tool_id: str
    """ID de la tool que se ejecutó."""

    success: bool
    """True si la ejecución completó exitosamente."""

    output: ToolOutput | None
    """Output de la tool si fue exitosa. None si falló."""

    risk_assessment: RiskAssessment
    """Evaluación de riesgo aplicada a esta ejecución."""

    execution_record: ToolExecutionRecord
    """Registro completo para el audit log."""

    approval_id: ApprovalId | None = None
    """ID de la aprobación si fue requerida y otorgada."""

    sandbox_used: bool = False
    """True si la ejecución ocurrió en el proceso sandbox."""

    error_code: str | None = None
    """Código de error ForgeError si la ejecución falló."""

    error_message: str | None = None
    """Mensaje de error legible si la ejecución falló."""

    duration_ms: float = 0.0
    """Duración de la ejecución en milisegundos."""

    @property
    def was_policy_denied(self) -> bool:
        """True si el PolicyEngine denegó la ejecución."""
        return (
            not self.success
            and self.error_code is not None
            and "POLICY_DENIED" in self.error_code
        )

    @property
    def required_approval(self) -> bool:
        """True si la ejecución requirió aprobación del usuario."""
        return self.approval_id is not None

    @property
    def output_for_llm(self) -> str:
        """
        Retorna el output de la tool serializado para el contexto del LLM.

        Si la ejecución falló, retorna un mensaje de error estructurado
        para que el LLM pueda reportarlo al usuario.

        Returns:
            String con el contenido del output o el error.
        """
        if self.success and self.output is not None:
            return self.output.serialize_for_llm()

        error_parts = [f"Error ejecutando '{self.tool_id}'"]
        if self.error_message:
            error_parts.append(f"Razón: {self.error_message}")
        return ". ".join(error_parts)

    def to_summary_dict(self) -> dict[str, Any]:
        """Serializa el resultado para el audit log."""
        return {
            "invocation_id": self.invocation_id.to_str(),
            "tool_id": self.tool_id,
            "success": self.success,
            "risk_level": self.risk_assessment.effective_risk.name,
            "sandbox_used": self.sandbox_used,
            "approval_id": self.approval_id.to_str() if self.approval_id else None,
            "duration_ms": round(self.duration_ms, 2),
            "error_code": self.error_code,
        }


# =============================================================================
# PUERTO ABSTRACTO: TOOL REGISTRY
# =============================================================================


class ToolRegistryPort(abc.ABC):
    """
    Puerto de gestión del catálogo de tools disponibles.

    El ToolRegistry es el único punto donde se descubren y gestionan las
    tools del sistema. Ningún otro componente accede a los ToolPort
    directamente — siempre pasan por el registry.

    El ContextBuilder lo consulta para obtener los schemas de tools a enviar
    al LLM. El TaskExecutor lo consulta para resolver tool_id → ToolPort.

    Contratos invariantes:
      1. El registry es la fuente de verdad de todas las tools disponibles.
      2. register() es idempotente — registrar la misma tool dos veces
         actualiza el registro existente (versionado).
      3. get_schemas_for_llm() solo retorna tools habilitadas y compatibles
         con la plataforma actual.
      4. La lista de tools registradas no cambia durante una sesión activa
         (en V1). El registry se llena en el bootstrap y permanece estático.
      5. get() retorna None si la tool no existe — nunca lanza KeyError.
    """

    @abc.abstractmethod
    def register(self, schema: ToolSchema) -> None:
        """
        Registra una tool en el catálogo.

        Si ya existe una tool con el mismo tool_id, la reemplaza.
        El registro se hace al inicio del sistema (bootstrap), antes de
        que comiencen las sesiones.

        Args:
            schema: Schema completo de la tool a registrar.
        """

    @abc.abstractmethod
    def unregister(self, tool_id: str) -> bool:
        """
        Elimina una tool del catálogo.

        Args:
            tool_id: ID de la tool a eliminar.

        Returns:
            True si existía y fue eliminada. False si no existía.
        """

    @abc.abstractmethod
    def get(self, tool_id: str) -> ToolRegistration | None:
        """
        Obtiene el registro de una tool por su ID.

        Args:
            tool_id: ID de la tool a buscar.

        Returns:
            ToolRegistration si existe, None si no.
        """

    @abc.abstractmethod
    def get_by_llm_name(self, llm_name: str) -> ToolRegistration | None:
        """
        Obtiene el registro de una tool por el nombre que el LLM usa.

        El LLM invoca tools por su llm_name (del ToolSchema), que puede
        diferir del tool_id interno. Este método resuelve esa diferencia.

        Args:
            llm_name: Nombre de la tool tal como el LLM la invocó.

        Returns:
            ToolRegistration si existe una tool con ese llm_name, None si no.
        """

    @abc.abstractmethod
    def list(self, filter: ToolFilter | None = None) -> list[ToolRegistration]:
        """
        Lista las tools del catálogo según los criterios del filtro.

        Args:
            filter: Criterios de filtrado. None retorna todas las tools.

        Returns:
            Lista de ToolRegistration que cumplen los criterios.
            Ordenada por categoría y luego por tool_id.
        """

    @abc.abstractmethod
    def get_schemas_for_llm(
        self,
        platform: Platform,
        permissions: PermissionSet,
    ) -> list[ToolDefinition]:
        """
        Retorna los schemas de tools disponibles para enviar al LLM.

        Este es el método que el ContextBuilder llama para construir la
        lista de tools que el LLM puede invocar en el siguiente turno.

        Solo incluye tools que son:
          - Habilitadas (enabled=True)
          - Compatibles con la plataforma actual
          - Permitidas por el PermissionSet del usuario

        Args:
            platform:    Plataforma actual (linux, windows).
            permissions: PermissionSet del usuario activo.

        Returns:
            Lista de ToolDefinition listos para incluir en LLMContext.tools.
            Ordenada de forma consistente para hashability del contexto.
        """

    @abc.abstractmethod
    def estimate_schemas_token_count(
        self,
        platform: Platform,
        permissions: PermissionSet,
    ) -> int:
        """
        Estima los tokens de los schemas de tools disponibles.

        El ContextBuilder lo usa para calcular el BudgetAllocation correcto
        antes de construir el contexto completo.

        Args:
            platform:    Plataforma actual.
            permissions: PermissionSet del usuario.

        Returns:
            Estimación de tokens de todos los schemas de tools disponibles.
        """

    @abc.abstractmethod
    def get_mutation_tool_ids(self) -> frozenset[str]:
        """
        Retorna el conjunto de tool_ids que producen mutación del sistema.

        El LLMResponseAnalyzer lo usa para clasificar correctamente si un
        SingleToolCall es SINGLE_ACTION (mutación) o QUERY (lectura).

        Returns:
            frozenset de tool_ids de tools con MutationType != NONE.
        """

    @abc.abstractmethod
    def record_invocation(
        self,
        tool_id: str,
        *,
        success: bool,
    ) -> None:
        """
        Registra una invocación de tool para estadísticas y circuit breaking.

        El ToolDispatchPort llama a este método después de cada ejecución.
        Actualiza invocation_count, last_invoked_at, y failure_count.

        Args:
            tool_id: ID de la tool invocada.
            success: True si la invocación fue exitosa.
        """

    @abc.abstractmethod
    def is_registered(self, tool_id: str) -> bool:
        """
        Verifica si una tool está registrada en el catálogo.

        Args:
            tool_id: ID de la tool a verificar.

        Returns:
            True si la tool existe en el catálogo (independientemente de si está habilitada).
        """

    @property
    @abc.abstractmethod
    def registered_count(self) -> int:
        """Número total de tools registradas (habilitadas o no)."""

    @property
    @abc.abstractmethod
    def enabled_count(self) -> int:
        """Número de tools habilitadas."""


# =============================================================================
# PUERTO ABSTRACTO: TOOL DISPATCH
# =============================================================================


class ToolDispatchPort(abc.ABC):
    """
    Puerto de ejecución de tools con enforcement completo de seguridad.

    El ToolDispatchPort es el guardián de la ejecución. Toda tool que el
    sistema quiere ejecutar pasa por este puerto, que garantiza:

      1. Policy enforcement: el PolicyEngine evalúa cada ejecución
      2. Approval workflow: si el riesgo lo requiere, suspende y espera
      3. Input validation: valida el input contra el schema de la tool
      4. Sandbox routing: las tools de desktop van al proceso sandbox
      5. Timeout enforcement: toda ejecución tiene un timeout máximo
      6. Audit recording: toda ejecución genera un ToolExecutionRecord
      7. Registry update: actualiza las estadísticas de la tool

    El TaskExecutor depende de este puerto y NUNCA llama directamente
    a un ToolPort — todo pasa por el ToolDispatchPort.

    Contratos invariantes:
      1. dispatch() SIEMPRE retorna un ToolExecutionResult, nunca lanza
         excepciones al caller (las convierte en result.success=False).
      2. Si policy_decision=DENY, retorna result.success=False sin ejecutar.
      3. Si policy_decision=REQUIRE_APPROVAL, suspende hasta que el
         ApprovalWorkflow retorne y luego procede o cancela.
      4. Toda ejecución genera un ToolExecutionRecord en el audit log,
         independientemente del resultado.
      5. El timeout es un hard limit — si la tool excede el timeout,
         la ejecución se cancela y se retorna result.success=False.
    """

    @abc.abstractmethod
    async def dispatch(
        self,
        request: ToolExecutionRequest,
        permissions: PermissionSet,
    ) -> ToolExecutionResult:
        """
        Ejecuta una tool con enforcement completo de seguridad.

        Flujo de ejecución:
          1. Resolver tool_id → ToolRegistration via ToolRegistryPort
          2. Validar input contra el schema de la tool
          3. Ejecutar PolicyEngine.evaluate(ProposedAction)
          4. Si DENY → retornar result.success=False inmediatamente
          5. Si REQUIRE_APPROVAL → iniciar ApprovalWorkflow y esperar
          6. Si ALLOW → proceder a la ejecución
          7. Routing: tool.requires_sandbox → SandboxProcess | directa
          8. Ejecutar con timeout enforcement
          9. Registrar ToolExecutionRecord en audit log
          10. Actualizar ToolRegistry con estadísticas
          11. Retornar ToolExecutionResult

        Args:
            request:     Solicitud de ejecución con toda la metadata necesaria.
            permissions: PermissionSet del usuario activo.

        Returns:
            ToolExecutionResult con el resultado completo.
            Nunca lanza excepciones — los errores van en result.success=False.
        """

    @abc.abstractmethod
    async def dispatch_batch(
        self,
        requests: list[ToolExecutionRequest],
        permissions: PermissionSet,
    ) -> list[ToolExecutionResult]:
        """
        Ejecuta múltiples tools secuencialmente.

        En V1, la ejecución es estrictamente secuencial — una tool a la vez.
        El TaskExecutor usa este método para el caso MultipleToolCalls donde
        el LLM emitió varios tool_calls en una sola respuesta.

        La ejecución continúa aunque alguna tool falle — el caller recibe
        todos los resultados y decide qué hacer con los fallos.

        Args:
            requests:    Lista de solicitudes de ejecución en orden.
            permissions: PermissionSet del usuario activo.

        Returns:
            Lista de ToolExecutionResult en el mismo orden que requests.
            Siempre tiene la misma longitud que requests.
        """

    @abc.abstractmethod
    async def validate_input(
        self,
        tool_id: str,
        arguments: dict[str, Any],
    ) -> list[str]:
        """
        Valida el input de una tool sin ejecutarla.

        El TaskExecutor puede pre-validar el input antes de iniciar el
        flujo de ejecución completo (policy check, approval, etc.).

        Args:
            tool_id:   ID de la tool a validar.
            arguments: Argumentos a validar contra el schema de la tool.

        Returns:
            Lista de errores de validación. Vacía si el input es válido.
        """

    @abc.abstractmethod
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
            RiskAssessment con el riesgo efectivo de esta ejecución.
        """

    @abc.abstractmethod
    async def get_pending_approvals(
        self,
        session_id: SessionId,
    ) -> list[dict[str, Any]]:
        """
        Retorna las aprobaciones pendientes para una sesión.

        El ConversationCoordinator lo consulta cuando el usuario responde
        a una solicitud de aprobación para saber qué ejecuciones están
        esperando.

        Args:
            session_id: ID de la sesión.

        Returns:
            Lista de dicts con la metadata de cada aprobación pendiente.
        """

    @abc.abstractmethod
    async def resolve_approval(
        self,
        approval_id: ApprovalId,
        granted: bool,
    ) -> None:
        """
        Resuelve una aprobación pendiente.

        El ApprovalWorkflow llama a este método cuando el usuario toma
        una decisión sobre una aprobación pendiente.

        Args:
            approval_id: ID de la aprobación a resolver.
            granted:     True si el usuario aprobó, False si denegó.
        """