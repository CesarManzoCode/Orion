"""
Framework de policies del ecosistema Forge Platform.

Este módulo define la infraestructura base del sistema de políticas de seguridad.
Es el tercer pilar de la arquitectura hexagonal de Forge — junto con LLMPort y
ToolPort, completa el triángulo de contratos fundamentales del sistema.

El framework de policies es deliberadamente fail-closed: ante cualquier duda
o error de evaluación, la decisión por defecto es DENY. Nunca ALLOW por defecto.

Arquitectura del sistema de políticas:

  ProposedAction          — acción que el agente quiere ejecutar
       │
       ▼
  PolicyEvaluationPort    — puerto abstracto (implementado por PolicyEngine)
       │
       ├── evalúa contra:
       │   ├── GlobalDenyList     — acciones siempre bloqueadas
       │   ├── ToolPolicies       — políticas específicas por tool
       │   ├── InputAnalysis      — análisis del input concreto
       │   ├── PermissionCheck    — permisos del usuario
       │   └── RateLimiting       — límites de tasa
       │
       ▼
  PolicyDecision          — resultado: ALLOW | DENY | REQUIRE_APPROVAL
       │
       ├── ALLOW           → TaskExecutor ejecuta directamente
       ├── DENY            → TaskExecutor aborta con razón clara
       └── REQUIRE_APPROVAL→ ApprovalWorkflow solicita confirmación al usuario

Principios de diseño:
  1. PolicyDecision es un tipo algebraico (sealed hierarchy), no un enum.
     Cada variante lleva su propio payload tipado.
  2. Las políticas son datos, no código hardcoded. Se cargan desde configuración
     y pueden actualizarse sin recompilar (preparado para V3 enterprise).
  3. El PolicyEvaluationPort es el único punto de evaluación. El TaskExecutor
     NUNCA hace bypass del PolicyEngine, incluso para acciones que parecen seguras.
  4. La evaluación es síncrona dentro del proceso — no hay llamadas de red.
     Latencia predecible y ausencia de dependencias externas para seguridad.
  5. Los errores de evaluación son DENY, no ALLOW. Fail-closed siempre.

Jerarquía de tipos:

  PolicyDecision (base)
  ├── Allow               — acción permitida, ejecutar directamente
  ├── Deny                — acción denegada, no ejecutar
  └── RequireApproval     — acción requiere confirmación del usuario

  Condition (base ABC)    — condición evaluable
  ├── ToolIdCondition     — condición basada en el ID de la tool
  ├── CategoryCondition   — condición basada en la categoría de la tool
  ├── RiskLevelCondition  — condición basada en el nivel de riesgo
  └── AlwaysCondition     — condición que siempre es verdadera/falsa

  Policy                  — regla de política: Condition → Decision
  ProposedAction          — descripción de la acción a evaluar
  PolicyEvaluationPort    — contrato del evaluador de políticas
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from forge_core.tools.protocol import (
    RiskAssessment,
    RiskLevel,
    ToolCategory,
    ToolInput,
    ToolSchema,
)


# =============================================================================
# POLÍTICA DE DECISIÓN — TIPO ALGEBRAICO
# =============================================================================


@dataclass(frozen=True)
class PolicyDecision:
    """
    Clase base de la decisión de política.

    PolicyDecision es un tipo algebraico sellado (sealed hierarchy) implementado
    como dataclasses frozen. Se usa como base para las tres variantes posibles:
    Allow, Deny y RequireApproval.

    Diseño deliberado: NO es un enum con campos opcionales. Cada variante tiene
    exactamente los campos que necesita, sin campos None innecesarios. Esto fuerza
    al caller a hacer pattern matching explícito y elimina los isinstance() anidados.

    Uso recomendado (pattern matching Python 3.10+):
        decision = await policy_engine.evaluate(action)
        match decision:
            case Allow():
                await executor.run(action)
            case Deny(reason=r):
                logger.warning('accion_denegada', reason=r)
                raise PolicyDeniedError(r)
            case RequireApproval(description=d, risk_level=rl):
                await approval_workflow.request(d, rl)

    Para compatibilidad con Python 3.11 (mínimo del proyecto):
        if isinstance(decision, Allow):
            ...
        elif isinstance(decision, Deny):
            ...
        elif isinstance(decision, RequireApproval):
            ...
    """

    def is_allow(self) -> bool:
        """True si la decisión es ALLOW."""
        return isinstance(self, Allow)

    def is_deny(self) -> bool:
        """True si la decisión es DENY."""
        return isinstance(self, Deny)

    def is_require_approval(self) -> bool:
        """True si la decisión requiere aprobación del usuario."""
        return isinstance(self, RequireApproval)

    def blocks_execution(self) -> bool:
        """True si la decisión impide la ejecución (DENY o pendiente de approval)."""
        return not isinstance(self, Allow)


@dataclass(frozen=True)
class Allow(PolicyDecision):
    """
    La acción está permitida y puede ejecutarse directamente.

    El TaskExecutor puede proceder sin ninguna intervención adicional.
    El audit log registrará la decisión como 'allow' con el nivel de riesgo
    efectivo para trazabilidad.
    """

    risk_level: RiskLevel = field(
        default=RiskLevel.NONE,
        metadata={"description": "Nivel de riesgo efectivo de la acción permitida."},
    )
    policy_name: str | None = field(
        default=None,
        metadata={"description": "Nombre de la política que permitió la acción."},
    )
    requires_sandbox: bool = field(
        default=False,
        metadata={
            "description": (
                "True si la acción, aunque permitida, debe ejecutarse en sandbox. "
                "Permite ejecución sin approval pero con aislamiento."
            )
        },
    )

    def __str__(self) -> str:
        sandbox_note = " (en sandbox)" if self.requires_sandbox else ""
        return f"ALLOW [{self.risk_level.name}]{sandbox_note}"


@dataclass(frozen=True)
class Deny(PolicyDecision):
    """
    La acción está denegada y no puede ejecutarse bajo ninguna circunstancia.

    Deny es terminal — no hay mecanismo para convertirla en Allow ni en
    RequireApproval. Si el PolicyEngine retorna Deny, la acción muere ahí.

    El reason debe ser informativo pero no revelar detalles de implementación
    de las políticas que puedan ayudar a un atacante a bypassearlas.
    """

    reason: str = field(
        metadata={"description": "Razón legible de la denegación (para logs y UX)."},
    )
    policy_name: str | None = field(
        default=None,
        metadata={"description": "Nombre de la política que denegó la acción."},
    )
    risk_level: RiskLevel = field(
        default=RiskLevel.CRITICAL,
        metadata={"description": "Nivel de riesgo que motivó la denegación."},
    )
    is_security_violation: bool = field(
        default=False,
        metadata={
            "description": (
                "True si la denegación es consecuencia de una violación de "
                "seguridad (path traversal, injection attempt, etc.). "
                "Activa logging con nivel CRITICAL en el audit trail."
            )
        },
    )

    def __str__(self) -> str:
        violation_note = " [SECURITY VIOLATION]" if self.is_security_violation else ""
        return f"DENY{violation_note}: {self.reason}"


@dataclass(frozen=True)
class RequireApproval(PolicyDecision):
    """
    La acción requiere confirmación explícita del usuario antes de ejecutarse.

    El TaskExecutor suspende la ejecución, activa el ApprovalWorkflow, y
    espera la decisión del usuario. Si el usuario aprueba, la acción se
    ejecuta. Si deniega o el timeout expira, la acción es cancelada.

    description debe ser legible para un usuario no técnico — sin tool IDs,
    sin paths internos, sin schemas JSON. Solo una descripción clara de qué
    va a hacer el sistema.
    """

    description: str = field(
        metadata={
            "description": (
                "Descripción legible para el usuario de la acción a aprobar. "
                "Debe ser clara, concisa y no técnica. "
                "Ejemplo: 'Quiero escribir un archivo en tu carpeta Documentos'"
            )
        },
    )
    risk_level: RiskLevel = field(
        metadata={"description": "Nivel de riesgo de la acción que requiere aprobación."},
    )
    tool_id: str = field(
        metadata={"description": "ID de la tool que requiere aprobación."},
    )
    policy_name: str | None = field(
        default=None,
        metadata={"description": "Nombre de la política que requiere la aprobación."},
    )
    is_first_time: bool = field(
        default=False,
        metadata={
            "description": (
                "True si es la primera vez que el usuario aprueba esta tool/acción. "
                "Permite mostrar un mensaje de onboarding adicional en la UI."
            )
        },
    )
    remember_decision: bool = field(
        default=True,
        metadata={
            "description": (
                "Si True, el sistema ofrecerá al usuario recordar su decisión "
                "para no preguntar de nuevo en el futuro para esta misma acción."
            )
        },
    )

    def __str__(self) -> str:
        first_time = " [primera vez]" if self.is_first_time else ""
        return f"REQUIRE_APPROVAL [{self.risk_level.name}]{first_time}: {self.description}"


# =============================================================================
# ACCIÓN PROPUESTA
# =============================================================================


@dataclass(frozen=True)
class ProposedAction:
    """
    Descripción completa de una acción que el agente quiere ejecutar.

    El PolicyEngine recibe un ProposedAction y produce un PolicyDecision.
    ProposedAction agrupa toda la información necesaria para evaluar el riesgo:
    la tool a usar, su schema, el input concreto y el contexto de sesión.

    Es el objeto central que viaja desde el TaskExecutor hasta el PolicyEngine
    y de vuelta. Immutable por diseño — no se modifica durante la evaluación.
    """

    tool_schema: ToolSchema = field(
        metadata={"description": "Schema completo de la tool a ejecutar."},
    )
    tool_input: ToolInput = field(
        metadata={"description": "Input validado para la invocación de la tool."},
    )
    session_id: str = field(
        metadata={"description": "ID de la sesión en la que ocurre la acción."},
    )
    task_id: str | None = field(
        default=None,
        metadata={"description": "ID de la tarea padre si la acción es parte de un plan."},
    )
    context: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": (
                "Contexto adicional para la evaluación de la política. "
                "Ejemplos: historial de acciones en la sesión, estado del plan, "
                "información del perfil del usuario."
            )
        },
    )

    @property
    def tool_id(self) -> str:
        """Atajo para acceder al tool_id sin navegar el schema."""
        return self.tool_schema.tool_id

    @property
    def tool_category(self) -> ToolCategory:
        """Atajo para acceder a la categoría de la tool."""
        return self.tool_schema.category

    @property
    def base_risk(self) -> RiskLevel:
        """Nivel de riesgo base declarado por la tool."""
        return self.tool_schema.capability.risk_classification


# =============================================================================
# SISTEMA DE CONDICIONES
# =============================================================================


class Condition(abc.ABC):
    """
    Condición evaluable en el contexto de una ProposedAction.

    Las condiciones son los bloques de construcción de las políticas. Cada
    política tiene una condición que determina cuándo aplica y una decisión
    que retorna cuando la condición es verdadera.

    Las condiciones son composables via operadores lógicos: AND, OR, NOT.
    Se evalúan de forma lazy — si la primera condición de un AND es False,
    no se evalúa la segunda.
    """

    @abc.abstractmethod
    def evaluate(self, action: ProposedAction) -> bool:
        """
        Evalúa si la condición aplica a la acción propuesta.

        Args:
            action: La acción a evaluar.

        Returns:
            True si la condición aplica a esta acción.
        """

    def __and__(self, other: Condition) -> AndCondition:
        """Operador &: combina dos condiciones con AND lógico."""
        return AndCondition(left=self, right=other)

    def __or__(self, other: Condition) -> OrCondition:
        """Operador |: combina dos condiciones con OR lógico."""
        return OrCondition(left=self, right=other)

    def __invert__(self) -> NotCondition:
        """Operador ~: invierte la condición (NOT lógico)."""
        return NotCondition(condition=self)


class AndCondition(Condition):
    """
    Condición compuesta: verdadera solo si AMBAS sub-condiciones son verdaderas.
    Evaluación lazy — no evalúa right si left es False.
    """

    def __init__(self, left: Condition, right: Condition) -> None:
        self.left = left
        self.right = right

    def evaluate(self, action: ProposedAction) -> bool:
        return self.left.evaluate(action) and self.right.evaluate(action)


class OrCondition(Condition):
    """
    Condición compuesta: verdadera si AL MENOS UNA sub-condición es verdadera.
    Evaluación lazy — no evalúa right si left es True.
    """

    def __init__(self, left: Condition, right: Condition) -> None:
        self.left = left
        self.right = right

    def evaluate(self, action: ProposedAction) -> bool:
        return self.left.evaluate(action) or self.right.evaluate(action)


class NotCondition(Condition):
    """Condición negada: verdadera si la sub-condición es falsa."""

    def __init__(self, condition: Condition) -> None:
        self.condition = condition

    def evaluate(self, action: ProposedAction) -> bool:
        return not self.condition.evaluate(action)


class AlwaysCondition(Condition):
    """
    Condición que siempre retorna el valor especificado.

    Útil para políticas catch-all al final de una lista de políticas ordenadas
    por prioridad (ej: 'si ninguna otra política aplica, DENY por defecto').
    """

    def __init__(self, value: bool = True) -> None:
        """
        Args:
            value: True para una condición que siempre es verdadera,
                   False para una que siempre es falsa.
        """
        self._value = value

    def evaluate(self, action: ProposedAction) -> bool:
        return self._value


class ToolIdCondition(Condition):
    """
    Condición que verifica si la tool tiene uno de los IDs especificados.

    Permite definir políticas específicas para una tool concreta o un
    conjunto de tools relacionadas.
    """

    def __init__(self, *tool_ids: str) -> None:
        """
        Args:
            *tool_ids: Uno o más tool IDs que activan esta condición.
        """
        self._tool_ids = frozenset(tool_ids)

    def evaluate(self, action: ProposedAction) -> bool:
        return action.tool_id in self._tool_ids


class CategoryCondition(Condition):
    """
    Condición que verifica si la tool pertenece a una de las categorías especificadas.

    Permite definir políticas que aplican a toda una categoría de tools,
    simplificando la gestión cuando se añaden nuevas tools a una categoría.
    """

    def __init__(self, *categories: ToolCategory) -> None:
        """
        Args:
            *categories: Una o más categorías que activan esta condición.
        """
        self._categories = frozenset(categories)

    def evaluate(self, action: ProposedAction) -> bool:
        return action.tool_category in self._categories


class RiskLevelCondition(Condition):
    """
    Condición que verifica si el nivel de riesgo base de la tool
    cumple la comparación especificada.

    Permite definir políticas basadas en rangos de riesgo sin
    necesidad de enumerar tool IDs específicos.
    """

    class Operator(Enum):
        """Operador de comparación para el nivel de riesgo."""
        GTE = auto()  # >=
        GT = auto()   # >
        LTE = auto()  # <=
        LT = auto()   # <
        EQ = auto()   # ==

    def __init__(
        self,
        risk_level: RiskLevel,
        operator: Operator = Operator.GTE,
    ) -> None:
        """
        Args:
            risk_level: Nivel de riesgo de referencia.
            operator:   Operador de comparación (por defecto >=).
        """
        self._risk_level = risk_level
        self._operator = operator

    def evaluate(self, action: ProposedAction) -> bool:
        base = action.base_risk
        ref = self._risk_level
        match self._operator:
            case self.Operator.GTE: return base >= ref
            case self.Operator.GT:  return base > ref
            case self.Operator.LTE: return base <= ref
            case self.Operator.LT:  return base < ref
            case self.Operator.EQ:  return base == ref
            case _:                 return False


class MutationTypeCondition(Condition):
    """
    Condición que verifica si la tool tiene el tipo de mutación especificado.

    Útil para políticas que aplican a todas las tools que modifican
    el estado del sistema operativo, independientemente de su categoría.
    """

    def __init__(self, *mutation_types: Any) -> None:
        """
        Args:
            *mutation_types: Uno o más MutationType que activan esta condición.
        """
        self._mutation_types = frozenset(mutation_types)

    def evaluate(self, action: ProposedAction) -> bool:
        return action.tool_schema.capability.mutation_type in self._mutation_types


# =============================================================================
# POLÍTICA
# =============================================================================


class PolicyPriority(int, Enum):
    """
    Prioridades predefinidas para ordenar el conjunto de políticas.

    Las políticas se evalúan en orden de prioridad descendente (mayor número
    primero). La primera política cuya condición sea True gana.

    Usar estas constantes garantiza un ordenamiento semántico claro y evita
    conflictos entre políticas definidas en diferentes partes del sistema.
    """

    CRITICAL_BLOCK = 1000
    """
    Bloqueo de seguridad crítico. Se evalúa primero, antes de cualquier
    otra política. Ejemplos: global deny list, path traversal detection.
    """

    SECURITY = 800
    """
    Políticas de seguridad de alta prioridad. Ejemplos: bloqueo de
    acceso a archivos sensibles del sistema.
    """

    ADMIN = 600
    """
    Políticas administrativas configuradas por el usuario avanzado.
    Ejemplos: restricciones corporativas cargadas desde servidor.
    """

    DEFAULT = 400
    """
    Políticas por defecto del producto. Cubren el caso normal de uso.
    Ejemplos: require_approval para desktop automation.
    """

    PERMISSIVE = 200
    """
    Políticas permisivas para herramientas de confianza explícita.
    Ejemplos: allow para tools de solo lectura de documentos.
    """

    FALLBACK = 0
    """
    Política de fallback. La última en evaluarse. Si ninguna otra política
    aplica, esta decide el comportamiento por defecto (generalmente DENY).
    """


@dataclass
class Policy:
    """
    Regla de política: Condition → PolicyDecision.

    Una política tiene una condición que determina cuándo aplica y una
    función que produce la decisión cuando la condición es True.

    Las políticas son datos en el sentido de que se crean declarativamente y
    se pasan al PolicyEngine como lista ordenada. El engine itera sobre ellas
    en orden de prioridad y retorna la primera decisión de la primera política
    cuya condición es verdadera.

    Ejemplo de política:
        Policy(
            name="require_approval_for_clipboard",
            condition=ToolIdCondition("clipboard_write", "clipboard_read"),
            decision_fn=lambda action: RequireApproval(
                description="Quiero acceder a tu portapapeles",
                risk_level=RiskLevel.MEDIUM,
                tool_id=action.tool_id,
            ),
            priority=PolicyPriority.DEFAULT,
            description="El acceso al portapapeles requiere aprobación explícita.",
        )
    """

    name: str
    """Nombre único de la política para identificación en logs y debugging."""

    condition: Condition
    """
    Condición que determina cuándo aplica esta política.
    Si evaluate() retorna True, se aplica decision_fn.
    """

    decision_fn: Any  # Callable[[ProposedAction], PolicyDecision]
    """
    Función que produce la decisión cuando la condición es True.
    Recibe el ProposedAction y retorna un PolicyDecision.
    Usar callable para permitir decisiones dinámicas basadas en el input.
    """

    priority: int = PolicyPriority.DEFAULT
    """
    Prioridad de la política. Mayor número = mayor prioridad.
    Se evalúan en orden descendente.
    """

    description: str = ""
    """Descripción legible de la política para documentación y debugging."""

    enabled: bool = True
    """
    Si False, la política se ignora en la evaluación.
    Permite deshabilitar políticas temporalmente sin eliminarlas.
    """

    def apply(self, action: ProposedAction) -> PolicyDecision | None:
        """
        Evalúa la condición y retorna la decisión si aplica.

        Args:
            action: La acción propuesta a evaluar.

        Returns:
            PolicyDecision si la condición es verdadera y la política está habilitada.
            None si la condición es falsa o la política está deshabilitada.
        """
        if not self.enabled:
            return None
        if self.condition.evaluate(action):
            return self.decision_fn(action)
        return None


# =============================================================================
# RESULTADO ENRIQUECIDO DE EVALUACIÓN
# =============================================================================


@dataclass(frozen=True)
class PolicyEvaluationResult:
    """
    Resultado enriquecido de la evaluación de políticas por el PolicyEngine.

    Contiene la PolicyDecision final junto con metadata de auditoría sobre
    cómo se llegó a esa decisión: qué política ganó, el risk assessment
    completo, y métricas de la evaluación.
    """

    decision: PolicyDecision
    """La decisión final del PolicyEngine."""

    risk_assessment: RiskAssessment
    """Evaluación de riesgo completa de la acción."""

    winning_policy: str | None
    """Nombre de la política que produjo la decisión. None si fue el fallback."""

    policies_evaluated: int
    """Número de políticas evaluadas antes de llegar a la decisión."""

    evaluation_time_ms: float
    """Tiempo de evaluación en milisegundos."""

    additional_context: dict[str, Any] = field(default_factory=dict)
    """Contexto adicional de la evaluación para debugging."""

    def to_audit_dict(self) -> dict[str, Any]:
        """
        Serializa el resultado para el audit log.

        Returns:
            Diccionario con toda la información de auditoría de la evaluación.
        """
        return {
            "decision": str(self.decision),
            "decision_type": type(self.decision).__name__,
            "risk_level": self.risk_assessment.effective_risk.name,
            "base_risk": self.risk_assessment.base_risk.name,
            "input_risk": self.risk_assessment.input_risk.name,
            "winning_policy": self.winning_policy,
            "policies_evaluated": self.policies_evaluated,
            "evaluation_time_ms": round(self.evaluation_time_ms, 3),
            "requires_approval": self.decision.is_require_approval(),
            "is_blocked": self.decision.is_deny(),
            "justification": self.risk_assessment.justification,
        }


# =============================================================================
# PUERTO ABSTRACTO DE EVALUACIÓN DE POLÍTICAS
# =============================================================================


class PolicyEvaluationPort(abc.ABC):
    """
    Puerto abstracto del evaluador de políticas.

    Define el contrato que el PolicyEngine concreto debe implementar.
    La capa de aplicación (TaskExecutor) solo conoce este puerto — nunca
    la implementación concreta del engine.

    La implementación concreta (DefaultPolicyEngine) en src/application/
    mantiene la lista de políticas ordenadas por prioridad y las evalúa
    secuencialmente para cada ProposedAction.

    Contratos invariantes que toda implementación DEBE respetar:
      1. evaluate() SIEMPRE retorna un PolicyEvaluationResult — nunca lanza
         excepciones al caller. Los errores internos se convierten en Deny.
      2. Si ninguna política aplica, el resultado por defecto es Deny.
         Fail-closed: si no hay una Allow explícita, se deniega.
      3. La evaluación es determinista: el mismo (action, context) produce
         siempre el mismo resultado (sin estado interno mutable).
      4. evaluate() es eficiente: debe completar en < 10ms en condiciones normales.
         No hace I/O, no llama al LLM, no accede a la red.
      5. Las políticas CRITICAL_BLOCK se evalúan primero, antes que cualquier
         otra. Esto garantiza que los bloqueos de seguridad no sean bypasseados
         por políticas de menor prioridad.
    """

    @abc.abstractmethod
    async def evaluate(
        self,
        action: ProposedAction,
    ) -> PolicyEvaluationResult:
        """
        Evalúa una acción propuesta contra el conjunto activo de políticas.

        Itera sobre las políticas en orden de prioridad descendente y retorna
        la decisión de la primera política cuya condición sea verdadera.
        Si ninguna política aplica, retorna Deny (fail-closed).

        Args:
            action: La acción propuesta que el agente quiere ejecutar.

        Returns:
            PolicyEvaluationResult con la decisión y metadata de auditoría.
            Nunca lanza excepciones — los errores internos se convierten en
            Deny con is_security_violation=False para distinguirlos de
            violaciones de seguridad reales.
        """

    @abc.abstractmethod
    async def classify_risk(
        self,
        action: ProposedAction,
    ) -> RiskAssessment:
        """
        Clasifica el riesgo de una acción sin evaluar la decisión completa.

        Permite al sistema obtener el risk assessment sin ejecutar todo el
        pipeline de políticas. Útil para el ContextBuilder (incluir el riesgo
        en el contexto del LLM) y para pre-filtrado.

        Args:
            action: La acción propuesta a clasificar.

        Returns:
            RiskAssessment con el riesgo base, input risk y effective risk.
        """

    @abc.abstractmethod
    def get_active_policies(self) -> list[Policy]:
        """
        Retorna la lista de políticas activas ordenadas por prioridad.

        Útil para debugging, auditoría y para el debug CLI que muestra
        las políticas vigentes al administrador del sistema.

        Returns:
            Lista de Policy ordenada por prioridad descendente.
            Solo incluye políticas con enabled=True.
        """

    @abc.abstractmethod
    def register_policy(self, policy: Policy) -> None:
        """
        Registra una nueva política en el engine.

        Las políticas se pueden registrar en cualquier momento — el engine
        las ordena automáticamente por prioridad. Las políticas CRITICAL_BLOCK
        siempre se evalúan primero, independientemente del orden de registro.

        Args:
            policy: La política a registrar. Si ya existe una con el mismo
                    nombre, la reemplaza (permite actualización de políticas
                    sin reiniciar el sistema).
        """

    @abc.abstractmethod
    def remove_policy(self, policy_name: str) -> bool:
        """
        Elimina una política del engine por nombre.

        Args:
            policy_name: Nombre de la política a eliminar.

        Returns:
            True si la política existía y fue eliminada.
            False si no existía ninguna política con ese nombre.
        """


# =============================================================================
# POLÍTICAS BASE PREDEFINIDAS
# =============================================================================
# Estas son políticas de conveniencia que todo PolicyEngine debería incluir.
# Se definen aquí como constantes del framework para reutilización.


def make_critical_block_policy() -> Policy:
    """
    Crea la política de bloqueo crítico para acciones CRITICAL risk.

    Esta política debe registrarse con CRITICAL_BLOCK priority en todo
    PolicyEngine. Garantiza que ninguna acción clasificada como CRITICAL
    pueda ejecutarse, independientemente de otras políticas.

    Returns:
        Policy con prioridad CRITICAL_BLOCK que deniega RiskLevel.CRITICAL.
    """
    return Policy(
        name="forge.critical_risk_block",
        condition=RiskLevelCondition(
            RiskLevel.CRITICAL,
            operator=RiskLevelCondition.Operator.EQ,
        ),
        decision_fn=lambda action: Deny(
            reason=(
                f"La acción '{action.tool_id}' está clasificada como riesgo crítico "
                f"y no puede ejecutarse bajo ninguna circunstancia."
            ),
            policy_name="forge.critical_risk_block",
            risk_level=RiskLevel.CRITICAL,
            is_security_violation=False,
        ),
        priority=PolicyPriority.CRITICAL_BLOCK,
        description=(
            "Bloquea cualquier acción clasificada con RiskLevel.CRITICAL. "
            "Esta política no puede deshabilitarse."
        ),
    )


def make_high_risk_approval_policy() -> Policy:
    """
    Crea la política de aprobación para acciones HIGH risk.

    Las acciones de riesgo alto siempre requieren confirmación explícita
    del usuario. Esta política debe registrarse con DEFAULT priority.

    Returns:
        Policy con prioridad DEFAULT que requiere aprobación para RiskLevel.HIGH.
    """
    return Policy(
        name="forge.high_risk_require_approval",
        condition=RiskLevelCondition(
            RiskLevel.HIGH,
            operator=RiskLevelCondition.Operator.EQ,
        ),
        decision_fn=lambda action: RequireApproval(
            description=(
                f"El agente quiere realizar una acción de riesgo alto "
                f"usando {action.tool_schema.display_name}. "
                f"Esta acción puede tener efectos significativos en tu sistema."
            ),
            risk_level=RiskLevel.HIGH,
            tool_id=action.tool_id,
            policy_name="forge.high_risk_require_approval",
        ),
        priority=PolicyPriority.DEFAULT,
        description=(
            "Requiere aprobación explícita del usuario para acciones de riesgo HIGH. "
            "Se aplica a todas las tools con risk_classification=HIGH."
        ),
    )


def make_read_only_allow_policy() -> Policy:
    """
    Crea la política de permiso directo para tools de solo lectura.

    Las tools sin mutación (MutationType.NONE) y con riesgo NONE o LOW
    pueden ejecutarse directamente sin aprobación ni sandbox.

    Returns:
        Policy con prioridad PERMISSIVE que permite tools de solo lectura.
    """
    from forge_core.tools.protocol import MutationType

    return Policy(
        name="forge.read_only_allow",
        condition=MutationTypeCondition(MutationType.NONE),
        decision_fn=lambda action: Allow(
            risk_level=action.base_risk,
            policy_name="forge.read_only_allow",
            requires_sandbox=False,
        ),
        priority=PolicyPriority.PERMISSIVE,
        description=(
            "Permite la ejecución directa de tools de solo lectura (MutationType.NONE). "
            "Sin sandbox, sin approval."
        ),
    )


def make_fallback_deny_policy() -> Policy:
    """
    Crea la política de fallback que deniega cualquier acción no cubierta.

    Esta política implementa el principio de fail-closed: si ninguna otra
    política aplica explícitamente, la acción se deniega. Debe registrarse
    con FALLBACK priority (la más baja).

    Returns:
        Policy con prioridad FALLBACK que deniega cualquier acción.
    """
    return Policy(
        name="forge.fallback_deny",
        condition=AlwaysCondition(value=True),
        decision_fn=lambda action: Deny(
            reason=(
                f"Ninguna política permite explícitamente la acción '{action.tool_id}'. "
                f"Por defecto, las acciones no autorizadas explícitamente son denegadas."
            ),
            policy_name="forge.fallback_deny",
            risk_level=action.base_risk,
            is_security_violation=False,
        ),
        priority=PolicyPriority.FALLBACK,
        description=(
            "Política de fallback fail-closed: deniega cualquier acción que no haya "
            "sido explícitamente permitida por una política de mayor prioridad."
        ),
    )