"""
Entidades Task, Plan, PlannedStep y StepExecution del dominio HiperForge User.

Este módulo define las entidades que modelan la ejecución de trabajo del agente.
Cuando el usuario pide algo que requiere tools, el sistema crea una Task. Si
la tarea es compleja, genera un Plan con PlannedSteps. Cada step ejecutado
produce un StepExecution con su resultado.

Jerarquía de entidades:

  Task (aggregate)
  ├── intent: Intent                       (qué quiere el usuario)
  ├── plan: Plan | None                    (plan multi-step, si aplica)
  │   └── steps: list[PlannedStep]         (pasos del plan)
  ├── executions: list[StepExecution]      (ejecuciones realizadas)
  └── evidence: list[Evidence]             (artefactos de evidencia)

Ciclos de vida:

  TaskStatus:
    CREATED → PLANNING? → EXECUTING → AWAITING_APPROVAL → EXECUTING
                                    → COMPLETED
                                    → FAILED
                                    → CANCELLED

  PlanStatus:
    DRAFT → VALIDATED → IN_PROGRESS → COMPLETED
                                    → FAILED
                                    → REVISED (si necesita re-planning)

  StepStatus:
    PENDING → POLICY_CHECK → AWAITING_APPROVAL → EXECUTING → COMPLETED
                           → DENIED                        → FAILED
                                                           → TIMED_OUT
                                                           → SKIPPED

Principios de diseño:
  1. Task es el aggregate — todas las operaciones pasan por él.
  2. Plan es parte del aggregate Task, no una entidad independiente.
  3. Las transiciones de estado son explícitas y validadas en cada comando.
  4. StepExecution es append-only — los resultados no se modifican.
  5. El dominio no sabe cómo se ejecutan las tools — solo modela el estado
     y las invariantes de la ejecución.
  6. Los eventos de dominio permiten al TaskExecutor reaccionar a cambios
     de estado sin polling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from forge_core.errors.types import ForgeDomainError, InvalidStateTransitionError
from forge_core.tools.protocol import RiskAssessment, RiskLevel

from src.domain.value_objects.identifiers import (
    ApprovalId,
    ArtifactId,
    InvocationId,
    PlanId,
    SessionId,
    StepId,
    TaskId,
    TurnId,
)
from src.domain.value_objects.intent import Intent


# =============================================================================
# ENUMERACIONES DE ESTADO
# =============================================================================


class TaskStatus(Enum):
    """Estado del ciclo de vida de una Task."""

    CREATED = "created"
    """Task creada, aún no ha comenzado."""

    PLANNING = "planning"
    """El LightweightPlanner está generando el plan multi-step."""

    EXECUTING = "executing"
    """La task está ejecutando tools (una o varias)."""

    AWAITING_APPROVAL = "awaiting_approval"
    """En espera de aprobación del usuario para continuar."""

    COMPLETED = "completed"
    """La task completó exitosamente todos sus pasos."""

    FAILED = "failed"
    """La task falló y no puede continuar."""

    CANCELLED = "cancelled"
    """La task fue cancelada por el usuario o el sistema."""


class PlanStatus(Enum):
    """Estado del ciclo de vida de un Plan."""

    DRAFT = "draft"
    """Plan generado, pendiente de validación contra policies."""

    VALIDATED = "validated"
    """Plan validado — todos los steps pasaron el policy check."""

    IN_PROGRESS = "in_progress"
    """El plan está en ejecución."""

    COMPLETED = "completed"
    """Todos los steps del plan completaron exitosamente."""

    FAILED = "failed"
    """Uno o más steps fallaron y el plan no puede continuar."""

    REVISED = "revised"
    """El plan necesita revisión (re-planning) por cambio de contexto."""


class StepStatus(Enum):
    """Estado del ciclo de vida de un PlannedStep."""

    PENDING = "pending"
    """Step pendiente de ejecución."""

    POLICY_CHECK = "policy_check"
    """Evaluando el step contra el PolicyEngine."""

    AWAITING_APPROVAL = "awaiting_approval"
    """Esperando aprobación del usuario."""

    EXECUTING = "executing"
    """El step está ejecutando su tool."""

    COMPLETED = "completed"
    """El step completó exitosamente."""

    FAILED = "failed"
    """El step falló."""

    TIMED_OUT = "timed_out"
    """El step excedió su timeout."""

    SKIPPED = "skipped"
    """El step fue omitido (dependencia fallida u opcional)."""

    DENIED = "denied"
    """El PolicyEngine denegó la ejecución del step."""


# Transiciones válidas por entidad
_TASK_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.CREATED: frozenset({TaskStatus.PLANNING, TaskStatus.EXECUTING}),
    TaskStatus.PLANNING: frozenset({TaskStatus.EXECUTING, TaskStatus.FAILED, TaskStatus.CANCELLED}),
    TaskStatus.EXECUTING: frozenset({
        TaskStatus.AWAITING_APPROVAL,
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    }),
    TaskStatus.AWAITING_APPROVAL: frozenset({
        TaskStatus.EXECUTING,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    }),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.FAILED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
}

_PLAN_TRANSITIONS: dict[PlanStatus, frozenset[PlanStatus]] = {
    PlanStatus.DRAFT: frozenset({PlanStatus.VALIDATED, PlanStatus.FAILED}),
    PlanStatus.VALIDATED: frozenset({PlanStatus.IN_PROGRESS}),
    PlanStatus.IN_PROGRESS: frozenset({
        PlanStatus.COMPLETED,
        PlanStatus.FAILED,
        PlanStatus.REVISED,
    }),
    PlanStatus.COMPLETED: frozenset(),
    PlanStatus.FAILED: frozenset(),
    PlanStatus.REVISED: frozenset({PlanStatus.DRAFT}),
}

_STEP_TRANSITIONS: dict[StepStatus, frozenset[StepStatus]] = {
    StepStatus.PENDING: frozenset({StepStatus.POLICY_CHECK, StepStatus.SKIPPED}),
    StepStatus.POLICY_CHECK: frozenset({
        StepStatus.AWAITING_APPROVAL,
        StepStatus.EXECUTING,
        StepStatus.DENIED,
        StepStatus.SKIPPED,
    }),
    StepStatus.AWAITING_APPROVAL: frozenset({
        StepStatus.EXECUTING,
        StepStatus.DENIED,
        StepStatus.SKIPPED,
    }),
    StepStatus.EXECUTING: frozenset({
        StepStatus.COMPLETED,
        StepStatus.FAILED,
        StepStatus.TIMED_OUT,
    }),
    StepStatus.COMPLETED: frozenset(),
    StepStatus.FAILED: frozenset(),
    StepStatus.TIMED_OUT: frozenset(),
    StepStatus.SKIPPED: frozenset(),
    StepStatus.DENIED: frozenset(),
}


def _validate_task_transition(
    task_id: TaskId,
    from_status: TaskStatus,
    to_status: TaskStatus,
) -> None:
    """Valida una transición de estado de Task."""
    allowed = _TASK_TRANSITIONS.get(from_status, frozenset())
    if to_status not in allowed:
        raise InvalidStateTransitionError(
            "Task",
            from_status.value,
            to_status.value,
            entity_id=task_id.to_str(),
        )


def _validate_step_transition(
    step_id: StepId,
    from_status: StepStatus,
    to_status: StepStatus,
) -> None:
    """Valida una transición de estado de PlannedStep."""
    allowed = _STEP_TRANSITIONS.get(from_status, frozenset())
    if to_status not in allowed:
        raise InvalidStateTransitionError(
            "PlannedStep",
            from_status.value,
            to_status.value,
            entity_id=step_id.to_str(),
        )


# =============================================================================
# EVIDENCIA Y RESULTADOS
# =============================================================================


@dataclass(frozen=True)
class Evidence:
    """
    Pieza de evidencia producida durante la ejecución de una task.

    La evidencia es el rastro auditeable de lo que el agente hizo:
    los resultados de búsquedas, el contenido de documentos analizados,
    los outputs de tools ejecutadas. Se acumula durante la ejecución
    y el LLM la usa para sintetizar la respuesta final.
    """

    evidence_id: str
    """ID único de esta pieza de evidencia (ulid sin prefijo tipado)."""

    source_tool_id: str
    """ID de la tool que produjo esta evidencia."""

    content_summary: str
    """
    Resumen del contenido de la evidencia (no el contenido completo).
    El contenido completo puede ser muy grande y está en el ArtifactStore.
    """

    artifact_id: ArtifactId | None
    """Artifact asociado si la evidencia fue almacenada como artifact."""

    produced_at: datetime
    """Timestamp de producción (UTC)."""

    token_count: int = 0
    """Estimación de tokens del contenido completo de esta evidencia."""


@dataclass(frozen=True)
class StepExecution:
    """
    Registro de la ejecución de un paso del plan.

    Es append-only — una vez creado no se modifica. Si necesita actualizarse
    (para registrar el resultado), se crea un nuevo StepExecution que reemplaza
    al anterior en la lista del Task.

    Diseño immutable: se crea con `StepExecution.started()` y se completa
    con `StepExecution.complete()` que retorna una nueva instancia.
    """

    invocation_id: InvocationId
    """ID de la invocación de tool asociada a este step."""

    step_id: StepId | None
    """ID del PlannedStep si es parte de un plan. None si es ejecución directa."""

    tool_id: str
    """ID de la tool ejecutada en este step."""

    status: StepStatus
    """Estado actual de la ejecución."""

    started_at: datetime
    """Timestamp de inicio de la ejecución (UTC)."""

    risk_assessment: RiskAssessment
    """Evaluación de riesgo de esta ejecución, realizada por el PolicyEngine."""

    approval_id: ApprovalId | None = None
    """ID de la aprobación del usuario si fue requerida."""

    completed_at: datetime | None = None
    """Timestamp de finalización. None si aún en ejecución."""

    error_code: str | None = None
    """Código de error ForgeError si la ejecución falló."""

    output_summary: str | None = None
    """Resumen sanitizado del output para el audit log."""

    sandbox_used: bool = False
    """True si la ejecución ocurrió en el proceso sandbox."""

    @property
    def is_terminal(self) -> bool:
        """True si la ejecución llegó a un estado terminal."""
        return self.status in {
            StepStatus.COMPLETED,
            StepStatus.FAILED,
            StepStatus.TIMED_OUT,
            StepStatus.SKIPPED,
            StepStatus.DENIED,
        }

    @property
    def duration_ms(self) -> float | None:
        """Duración de la ejecución en milisegundos. None si aún en progreso."""
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds() * 1000

    @property
    def was_successful(self) -> bool:
        """True si la ejecución completó exitosamente."""
        return self.status == StepStatus.COMPLETED

    @classmethod
    def started(
        cls,
        *,
        invocation_id: InvocationId,
        step_id: StepId | None,
        tool_id: str,
        risk_assessment: RiskAssessment,
        approval_id: ApprovalId | None = None,
        sandbox_used: bool = False,
    ) -> StepExecution:
        """
        Factory: crea una StepExecution en estado EXECUTING.

        Args:
            invocation_id:   ID único de la invocación.
            step_id:         ID del step del plan (None si directo).
            tool_id:         ID de la tool a ejecutar.
            risk_assessment: Evaluación de riesgo del PolicyEngine.
            approval_id:     ID de aprobación si fue requerida.
            sandbox_used:    True si se usa el proceso sandbox.

        Returns:
            Nueva StepExecution en estado EXECUTING.
        """
        return cls(
            invocation_id=invocation_id,
            step_id=step_id,
            tool_id=tool_id,
            status=StepStatus.EXECUTING,
            started_at=datetime.now(tz=timezone.utc),
            risk_assessment=risk_assessment,
            approval_id=approval_id,
            sandbox_used=sandbox_used,
        )

    def complete(
        self,
        *,
        output_summary: str | None = None,
    ) -> StepExecution:
        """
        Crea una nueva StepExecution marcada como COMPLETED.

        Args:
            output_summary: Resumen sanitizado del output.

        Returns:
            Nueva instancia con status=COMPLETED y completed_at fijado.
        """
        return StepExecution(
            invocation_id=self.invocation_id,
            step_id=self.step_id,
            tool_id=self.tool_id,
            status=StepStatus.COMPLETED,
            started_at=self.started_at,
            risk_assessment=self.risk_assessment,
            approval_id=self.approval_id,
            completed_at=datetime.now(tz=timezone.utc),
            sandbox_used=self.sandbox_used,
            output_summary=output_summary,
        )

    def fail(
        self,
        *,
        error_code: str,
        status: StepStatus = StepStatus.FAILED,
    ) -> StepExecution:
        """
        Crea una nueva StepExecution marcada como FAILED, TIMED_OUT o DENIED.

        Args:
            error_code: Código de error ForgeError.
            status:     Status terminal de fallo (FAILED, TIMED_OUT, DENIED).

        Returns:
            Nueva instancia con el status de fallo y completed_at fijado.
        """
        return StepExecution(
            invocation_id=self.invocation_id,
            step_id=self.step_id,
            tool_id=self.tool_id,
            status=status,
            started_at=self.started_at,
            risk_assessment=self.risk_assessment,
            approval_id=self.approval_id,
            completed_at=datetime.now(tz=timezone.utc),
            sandbox_used=self.sandbox_used,
            error_code=error_code,
        )


# =============================================================================
# PLANNED STEP
# =============================================================================


class PlannedStep:
    """
    Un paso dentro de un Plan multi-step.

    Representa una unidad de trabajo planificada: qué tool ejecutar,
    cuál es su nivel de riesgo, de qué otros steps depende, y si es opcional.

    Es mutable (a diferencia de los value objects) porque su estado
    cambia durante la ejecución del plan.
    """

    def __init__(
        self,
        step_id: StepId,
        description: str,
        tool_id: str,
        risk_level: RiskLevel,
        *,
        depends_on: list[StepId] | None = None,
        is_optional: bool = False,
        requires_approval: bool = False,
        sequence_number: int = 0,
    ) -> None:
        """
        Inicializa un PlannedStep.

        Args:
            step_id:          ID único del step.
            description:      Descripción legible del paso para el usuario y el LLM.
            tool_id:          ID de la tool a ejecutar en este step.
            risk_level:       Nivel de riesgo base del step.
            depends_on:       IDs de steps que deben completar antes de este.
            is_optional:      True si el step puede ser omitido sin fallar el plan.
            requires_approval: True si el step requiere aprobación del usuario.
            sequence_number:  Número de secuencia en el plan (1-indexed).
        """
        self._step_id = step_id
        self._description = description
        self._tool_id = tool_id
        self._risk_level = risk_level
        self._depends_on: list[StepId] = depends_on or []
        self._is_optional = is_optional
        self._requires_approval = requires_approval
        self._sequence_number = sequence_number
        self._status = StepStatus.PENDING
        self._execution: StepExecution | None = None

    # Propiedades (solo lectura)

    @property
    def step_id(self) -> StepId:
        return self._step_id

    @property
    def description(self) -> str:
        return self._description

    @property
    def tool_id(self) -> str:
        return self._tool_id

    @property
    def risk_level(self) -> RiskLevel:
        return self._risk_level

    @property
    def depends_on(self) -> list[StepId]:
        return list(self._depends_on)

    @property
    def is_optional(self) -> bool:
        return self._is_optional

    @property
    def requires_approval(self) -> bool:
        return self._requires_approval

    @property
    def sequence_number(self) -> int:
        return self._sequence_number

    @property
    def status(self) -> StepStatus:
        return self._status

    @property
    def execution(self) -> StepExecution | None:
        """La ejecución de este step, si se ha ejecutado."""
        return self._execution

    @property
    def is_completed(self) -> bool:
        return self._status == StepStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        return self._status in {
            StepStatus.FAILED,
            StepStatus.TIMED_OUT,
            StepStatus.DENIED,
        }

    @property
    def is_terminal(self) -> bool:
        """True si el step está en un estado terminal."""
        return self._status in {
            StepStatus.COMPLETED,
            StepStatus.FAILED,
            StepStatus.TIMED_OUT,
            StepStatus.SKIPPED,
            StepStatus.DENIED,
        }

    def can_execute(self, completed_step_ids: set[StepId]) -> bool:
        """
        Verifica si todas las dependencias del step están satisfechas.

        Args:
            completed_step_ids: Conjunto de IDs de steps ya completados.

        Returns:
            True si todas las dependencias están en completed_step_ids.
        """
        return all(dep in completed_step_ids for dep in self._depends_on)

    # Comandos de estado

    def begin_policy_check(self) -> None:
        """Transiciona el step a POLICY_CHECK."""
        _validate_step_transition(self._step_id, self._status, StepStatus.POLICY_CHECK)
        self._status = StepStatus.POLICY_CHECK

    def begin_awaiting_approval(self) -> None:
        """Transiciona el step a AWAITING_APPROVAL."""
        _validate_step_transition(
            self._step_id, self._status, StepStatus.AWAITING_APPROVAL
        )
        self._status = StepStatus.AWAITING_APPROVAL

    def begin_executing(self, execution: StepExecution) -> None:
        """
        Transiciona el step a EXECUTING y registra la ejecución.

        Args:
            execution: StepExecution iniciada para este step.
        """
        _validate_step_transition(self._step_id, self._status, StepStatus.EXECUTING)
        self._status = StepStatus.EXECUTING
        self._execution = execution

    def mark_completed(self, execution: StepExecution) -> None:
        """
        Marca el step como COMPLETED y actualiza la ejecución.

        Args:
            execution: StepExecution completada.
        """
        _validate_step_transition(self._step_id, self._status, StepStatus.COMPLETED)
        self._status = StepStatus.COMPLETED
        self._execution = execution

    def mark_failed(
        self,
        execution: StepExecution | None = None,
        status: StepStatus = StepStatus.FAILED,
    ) -> None:
        """
        Marca el step con el status de fallo especificado.

        Args:
            execution: StepExecution fallida (si existe).
            status:    Status de fallo (FAILED, TIMED_OUT, DENIED).
        """
        _validate_step_transition(self._step_id, self._status, status)
        self._status = status
        if execution is not None:
            self._execution = execution

    def skip(self) -> None:
        """Marca el step como SKIPPED (dependencia fallida o step opcional)."""
        _validate_step_transition(self._step_id, self._status, StepStatus.SKIPPED)
        self._status = StepStatus.SKIPPED

    def deny(self, execution: StepExecution | None = None) -> None:
        """Marca el step como DENIED (PolicyEngine denegó la ejecución)."""
        _validate_step_transition(self._step_id, self._status, StepStatus.DENIED)
        self._status = StepStatus.DENIED
        if execution is not None:
            self._execution = execution

    def to_dict(self) -> dict[str, Any]:
        """Serializa el step para el audit log y debugging."""
        return {
            "step_id": self._step_id.to_str(),
            "description": self._description,
            "tool_id": self._tool_id,
            "risk_level": self._risk_level.name,
            "sequence_number": self._sequence_number,
            "status": self._status.value,
            "is_optional": self._is_optional,
            "requires_approval": self._requires_approval,
            "depends_on": [d.to_str() for d in self._depends_on],
            "has_execution": self._execution is not None,
        }

    def __repr__(self) -> str:
        return (
            f"PlannedStep("
            f"id={self._step_id.to_str()!r}, "
            f"tool={self._tool_id!r}, "
            f"status={self._status.value!r}"
            f")"
        )


# =============================================================================
# PLAN
# =============================================================================


class Plan:
    """
    Plan de ejecución multi-step generado por el LightweightPlanner.

    Contiene la secuencia de PlannedSteps que el TaskExecutor ejecutará
    en orden. El plan conoce su propio progreso y puede determinar cuál
    es el siguiente step a ejecutar.

    Invariantes:
    - El plan no puede tener más de MAX_STEPS pasos.
    - Todos los steps tienen sequence_numbers únicos y consecutivos.
    - El plan solo avanza a IN_PROGRESS después de ser VALIDATED.
    - Un plan COMPLETED no puede modificarse.
    """

    MAX_STEPS: int = 10
    """
    Número máximo de steps permitidos en un plan.
    Definido en el documento de arquitectura para evitar over-planning.
    """

    def __init__(
        self,
        plan_id: PlanId,
        task_id: TaskId,
        steps: list[PlannedStep],
    ) -> None:
        """
        Inicializa un Plan.

        Args:
            plan_id: ID único del plan.
            task_id: ID de la task propietaria.
            steps:   Lista de PlannedSteps en orden de ejecución.

        Raises:
            ForgeDomainError: Si el número de steps supera MAX_STEPS.
            ForgeDomainError: Si los sequence_numbers son inválidos.
        """
        if len(steps) > self.MAX_STEPS:
            raise ForgeDomainError(
                f"El plan tiene {len(steps)} pasos, superando el máximo permitido "
                f"de {self.MAX_STEPS}. Simplifica la tarea o divídela en varias.",
                context={
                    "plan_id": plan_id.to_str(),
                    "steps_count": len(steps),
                    "max_steps": self.MAX_STEPS,
                },
            )

        self._plan_id = plan_id
        self._task_id = task_id
        self._steps = steps
        self._status = PlanStatus.DRAFT
        self._created_at = datetime.now(tz=timezone.utc)
        self._completed_at: datetime | None = None

        self._validate_step_sequence()

    def _validate_step_sequence(self) -> None:
        """
        Valida que los sequence_numbers de los steps son únicos y consecutivos.

        Raises:
            ForgeDomainError: Si hay sequence_numbers duplicados o no consecutivos.
        """
        if not self._steps:
            return
        seq_numbers = [s.sequence_number for s in self._steps]
        expected = list(range(1, len(self._steps) + 1))
        if sorted(seq_numbers) != expected:
            raise ForgeDomainError(
                f"Los sequence_numbers del plan no son válidos: {seq_numbers}. "
                f"Se esperaban: {expected}.",
                context={"plan_id": self._plan_id.to_str()},
            )

    # Propiedades

    @property
    def plan_id(self) -> PlanId:
        return self._plan_id

    @property
    def task_id(self) -> TaskId:
        return self._task_id

    @property
    def status(self) -> PlanStatus:
        return self._status

    @property
    def steps(self) -> list[PlannedStep]:
        return list(self._steps)

    @property
    def step_count(self) -> int:
        return len(self._steps)

    @property
    def created_at(self) -> datetime:
        return self._created_at

    @property
    def completed_at(self) -> datetime | None:
        return self._completed_at

    @property
    def is_complete(self) -> bool:
        return self._status == PlanStatus.COMPLETED

    @property
    def completed_steps(self) -> list[PlannedStep]:
        """Steps que han completado exitosamente."""
        return [s for s in self._steps if s.is_completed]

    @property
    def pending_steps(self) -> list[PlannedStep]:
        """Steps aún pendientes de ejecución."""
        return [s for s in self._steps if s.status == StepStatus.PENDING]

    @property
    def failed_steps(self) -> list[PlannedStep]:
        """Steps que fallaron."""
        return [s for s in self._steps if s.is_failed]

    @property
    def completion_ratio(self) -> float:
        """Fracción de steps completados [0.0, 1.0]."""
        if not self._steps:
            return 1.0
        return len(self.completed_steps) / len(self._steps)

    # Comandos

    def validate(self) -> None:
        """
        Transiciona el plan de DRAFT a VALIDATED.

        Debe llamarse después de que el PolicyEngine verificó todos los steps.

        Raises:
            InvalidStateTransitionError: Si el plan no está en DRAFT.
        """
        allowed = _PLAN_TRANSITIONS.get(self._status, frozenset())
        if PlanStatus.VALIDATED not in allowed:
            raise InvalidStateTransitionError(
                "Plan", self._status.value, PlanStatus.VALIDATED.value,
                entity_id=self._plan_id.to_str(),
            )
        self._status = PlanStatus.VALIDATED

    def begin(self) -> None:
        """
        Transiciona el plan a IN_PROGRESS.

        Raises:
            InvalidStateTransitionError: Si el plan no está en VALIDATED.
        """
        allowed = _PLAN_TRANSITIONS.get(self._status, frozenset())
        if PlanStatus.IN_PROGRESS not in allowed:
            raise InvalidStateTransitionError(
                "Plan", self._status.value, PlanStatus.IN_PROGRESS.value,
                entity_id=self._plan_id.to_str(),
            )
        self._status = PlanStatus.IN_PROGRESS

    def complete(self) -> None:
        """Marca el plan como COMPLETED."""
        allowed = _PLAN_TRANSITIONS.get(self._status, frozenset())
        if PlanStatus.COMPLETED not in allowed:
            raise InvalidStateTransitionError(
                "Plan", self._status.value, PlanStatus.COMPLETED.value,
                entity_id=self._plan_id.to_str(),
            )
        self._status = PlanStatus.COMPLETED
        self._completed_at = datetime.now(tz=timezone.utc)

    def fail(self) -> None:
        """Marca el plan como FAILED."""
        allowed = _PLAN_TRANSITIONS.get(self._status, frozenset())
        if PlanStatus.FAILED not in allowed:
            raise InvalidStateTransitionError(
                "Plan", self._status.value, PlanStatus.FAILED.value,
                entity_id=self._plan_id.to_str(),
            )
        self._status = PlanStatus.FAILED
        self._completed_at = datetime.now(tz=timezone.utc)

    def get_next_executable_step(self) -> PlannedStep | None:
        """
        Retorna el siguiente step que puede ejecutarse.

        Un step puede ejecutarse si está PENDING y todas sus dependencias
        están en el conjunto de steps completados.

        Returns:
            El siguiente PlannedStep ejecutable, o None si no hay ninguno.
        """
        completed_ids = {s.step_id for s in self.completed_steps}
        for step in self._steps:
            if (
                step.status == StepStatus.PENDING
                and step.can_execute(completed_ids)
            ):
                return step
        return None

    def get_step(self, step_id: StepId) -> PlannedStep | None:
        """
        Busca un step por su ID.

        Args:
            step_id: ID del step a buscar.

        Returns:
            PlannedStep con ese ID, o None si no existe.
        """
        for step in self._steps:
            if step.step_id == step_id:
                return step
        return None

    def can_continue(self) -> bool:
        """
        Determina si el plan puede continuar ejecutando.

        El plan puede continuar si hay al menos un step pendiente cuyas
        dependencias estén satisfechas, y ningún step no-opcional ha fallado.

        Returns:
            True si hay trabajo restante que puede ejecutarse.
        """
        if self._status not in {PlanStatus.IN_PROGRESS, PlanStatus.VALIDATED}:
            return False

        # Si hay un step no-opcional fallido, el plan no puede continuar
        for step in self._steps:
            if step.is_failed and not step.is_optional:
                return False

        return self.get_next_executable_step() is not None

    def to_dict(self) -> dict[str, Any]:
        """Serializa el plan para el audit log."""
        return {
            "plan_id": self._plan_id.to_str(),
            "task_id": self._task_id.to_str(),
            "status": self._status.value,
            "step_count": len(self._steps),
            "completed_steps": len(self.completed_steps),
            "completion_ratio": round(self.completion_ratio, 2),
            "steps": [s.to_dict() for s in self._steps],
        }

    def __repr__(self) -> str:
        return (
            f"Plan("
            f"id={self._plan_id.to_str()!r}, "
            f"status={self._status.value!r}, "
            f"steps={len(self._steps)}/{self.step_count}"
            f")"
        )


# =============================================================================
# EVENTOS DE DOMINIO DE TASK
# =============================================================================


@dataclass(frozen=True)
class TaskEvent:
    """Base de los eventos de dominio de Task."""

    task_id: TaskId
    """ID de la task que emitió el evento."""

    session_id: SessionId
    """ID de la sesión propietaria."""

    occurred_at: datetime
    """Timestamp del evento (UTC)."""


@dataclass(frozen=True)
class TaskCreated(TaskEvent):
    """Una nueva task fue creada."""

    intent_type: str
    """Tipo de intent que originó la task."""


@dataclass(frozen=True)
class TaskCompleted(TaskEvent):
    """La task completó exitosamente."""

    executions_count: int
    """Total de tool calls ejecutados."""

    evidence_count: int
    """Total de piezas de evidencia producidas."""

    duration_seconds: float
    """Duración total de la task en segundos."""


@dataclass(frozen=True)
class TaskFailed(TaskEvent):
    """La task falló."""

    error_code: str
    """Código de error que causó el fallo."""

    reason: str
    """Razón legible del fallo."""


@dataclass(frozen=True)
class StepExecuted(TaskEvent):
    """Un step del plan fue ejecutado (exitoso o fallido)."""

    step_id: StepId
    """ID del step ejecutado."""

    tool_id: str
    """ID de la tool ejecutada."""

    success: bool
    """True si el step completó exitosamente."""

    duration_ms: float | None
    """Duración de la ejecución en milisegundos."""


# =============================================================================
# TASK — AGGREGATE ROOT
# =============================================================================


class Task:
    """
    Aggregate root de una unidad de trabajo del agente.

    Una Task representa la intención del usuario transformada en ejecución:
    qué tools se invocaron, con qué resultados, y en qué estado terminó.

    Puede ser simple (una sola tool call) o compleja (un Plan multi-step).
    En ambos casos, el aggregate Task es la fuente de verdad.

    Invariantes:
    1. Una Task solo acepta nuevas ejecuciones mientras está EXECUTING.
    2. El plan (si existe) debe estar VALIDATED antes de ejecutar steps.
    3. La evidencia es append-only — nunca se modifica ni elimina.
    4. Una Task COMPLETED o FAILED es terminal — no acepta más cambios.
    5. El plan y las ejecuciones directas son mutuamente excluyentes:
       una Task o tiene plan o tiene ejecuciones directas (no ambos).
    """

    def __init__(
        self,
        task_id: TaskId,
        session_id: SessionId,
        origin_turn_id: TurnId,
        intent: Intent,
    ) -> None:
        """
        Inicializa una Task.

        Args:
            task_id:        ID único de la task.
            session_id:     ID de la sesión propietaria.
            origin_turn_id: Turn del que surge esta task.
            intent:         Intent del usuario que originó la task.
        """
        self._task_id = task_id
        self._session_id = session_id
        self._origin_turn_id = origin_turn_id
        self._intent = intent
        self._status = TaskStatus.CREATED
        self._plan: Plan | None = None
        self._executions: list[StepExecution] = []
        self._evidence: list[Evidence] = []
        self._artifact_ids: list[ArtifactId] = []
        self._created_at = datetime.now(tz=timezone.utc)
        self._completed_at: datetime | None = None
        self._failure_reason: str | None = None
        self._failure_error_code: str | None = None
        self._domain_events: list[TaskEvent] = []

        self._emit(TaskCreated(
            task_id=self._task_id,
            session_id=self._session_id,
            occurred_at=self._created_at,
            intent_type=self._intent.type.name,
        ))

    # =========================================================================
    # FACTORIES
    # =========================================================================

    @classmethod
    def create(
        cls,
        session_id: SessionId,
        origin_turn_id: TurnId,
        intent: Intent,
    ) -> Task:
        """
        Factory: crea una nueva Task.

        Args:
            session_id:     ID de la sesión propietaria.
            origin_turn_id: Turn del que surge esta task.
            intent:         Intent del usuario.

        Returns:
            Nueva Task en estado CREATED.
        """
        return cls(
            task_id=TaskId.generate(),
            session_id=session_id,
            origin_turn_id=origin_turn_id,
            intent=intent,
        )

    # =========================================================================
    # PROPIEDADES
    # =========================================================================

    @property
    def task_id(self) -> TaskId:
        return self._task_id

    @property
    def session_id(self) -> SessionId:
        return self._session_id

    @property
    def origin_turn_id(self) -> TurnId:
        return self._origin_turn_id

    @property
    def intent(self) -> Intent:
        return self._intent

    @property
    def status(self) -> TaskStatus:
        return self._status

    @property
    def plan(self) -> Plan | None:
        return self._plan

    @property
    def executions(self) -> list[StepExecution]:
        return list(self._executions)

    @property
    def evidence(self) -> list[Evidence]:
        return list(self._evidence)

    @property
    def artifact_ids(self) -> list[ArtifactId]:
        return list(self._artifact_ids)

    @property
    def created_at(self) -> datetime:
        return self._created_at

    @property
    def completed_at(self) -> datetime | None:
        return self._completed_at

    @property
    def failure_reason(self) -> str | None:
        return self._failure_reason

    @property
    def is_terminal(self) -> bool:
        """True si la task está en un estado terminal."""
        return self._status in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }

    @property
    def has_plan(self) -> bool:
        return self._plan is not None

    @property
    def execution_count(self) -> int:
        return len(self._executions)

    @property
    def successful_executions(self) -> list[StepExecution]:
        return [e for e in self._executions if e.was_successful]

    @property
    def domain_events(self) -> list[TaskEvent]:
        return list(self._domain_events)

    @property
    def duration_seconds(self) -> float | None:
        """Duración de la task en segundos. None si aún no completó."""
        if self._completed_at is None:
            return None
        return (self._completed_at - self._created_at).total_seconds()

    # =========================================================================
    # COMANDOS
    # =========================================================================

    def begin_planning(self) -> None:
        """
        Transiciona la task a PLANNING.

        Raises:
            InvalidStateTransitionError: Si la task no está en CREATED.
        """
        _validate_task_transition(self._task_id, self._status, TaskStatus.PLANNING)
        self._status = TaskStatus.PLANNING

    def attach_plan(self, plan: Plan) -> None:
        """
        Adjunta un Plan al aggregate y transiciona a EXECUTING.

        El plan debe estar en estado VALIDATED antes de adjuntarse.

        Args:
            plan: Plan validado por el PolicyEngine.

        Raises:
            ForgeDomainError: Si ya existe un plan en esta task.
            ForgeDomainError: Si el plan no está VALIDATED.
            InvalidStateTransitionError: Si la task no está en PLANNING o CREATED.
        """
        if self._plan is not None:
            raise ForgeDomainError(
                "La task ya tiene un plan. No se puede reemplazar un plan existente.",
                context={"task_id": self._task_id.to_str()},
            )

        if plan.status != PlanStatus.VALIDATED:
            raise ForgeDomainError(
                f"El plan debe estar VALIDATED antes de adjuntarse a la task. "
                f"Estado actual: {plan.status.value}.",
                context={
                    "task_id": self._task_id.to_str(),
                    "plan_id": plan.plan_id.to_str(),
                    "plan_status": plan.status.value,
                },
            )

        _validate_task_transition(self._task_id, self._status, TaskStatus.EXECUTING)
        self._plan = plan
        self._status = TaskStatus.EXECUTING

    def begin_executing(self) -> None:
        """
        Transiciona la task a EXECUTING (para ejecución directa sin plan).

        Raises:
            InvalidStateTransitionError: Si la transición no es válida.
        """
        _validate_task_transition(self._task_id, self._status, TaskStatus.EXECUTING)
        self._status = TaskStatus.EXECUTING

    def record_execution(self, execution: StepExecution) -> None:
        """
        Registra una StepExecution en el aggregate.

        Args:
            execution: StepExecution a registrar.

        Raises:
            ForgeDomainError: Si la task no está en EXECUTING.
        """
        if self._status != TaskStatus.EXECUTING:
            raise ForgeDomainError(
                f"No se pueden registrar ejecuciones en una task '{self._status.value}'.",
                context={"task_id": self._task_id.to_str()},
            )
        self._executions.append(execution)

    def add_evidence(self, evidence: Evidence) -> None:
        """
        Añade evidencia al aggregate.

        La evidencia es append-only. Los artifacts referenciados se almacenan
        en el ArtifactStore; el aggregate solo mantiene la referencia.

        Args:
            evidence: Evidencia a añadir.
        """
        self._evidence.append(evidence)
        if evidence.artifact_id is not None:
            self._artifact_ids.append(evidence.artifact_id)

    def request_approval(self) -> None:
        """
        Transiciona la task a AWAITING_APPROVAL.

        Raises:
            InvalidStateTransitionError: Si la task no está EXECUTING.
        """
        _validate_task_transition(
            self._task_id, self._status, TaskStatus.AWAITING_APPROVAL
        )
        self._status = TaskStatus.AWAITING_APPROVAL

    def resume_after_approval(self) -> None:
        """
        Resume la ejecución después de una aprobación del usuario.

        Raises:
            InvalidStateTransitionError: Si la task no está AWAITING_APPROVAL.
        """
        _validate_task_transition(
            self._task_id, self._status, TaskStatus.EXECUTING
        )
        self._status = TaskStatus.EXECUTING

    def complete(self) -> None:
        """
        Marca la task como COMPLETED.

        Raises:
            InvalidStateTransitionError: Si la task no está EXECUTING.
        """
        _validate_task_transition(self._task_id, self._status, TaskStatus.COMPLETED)
        self._status = TaskStatus.COMPLETED
        self._completed_at = datetime.now(tz=timezone.utc)

        self._emit(TaskCompleted(
            task_id=self._task_id,
            session_id=self._session_id,
            occurred_at=self._completed_at,
            executions_count=len(self._executions),
            evidence_count=len(self._evidence),
            duration_seconds=self.duration_seconds or 0.0,
        ))

    def fail(self, *, error_code: str, reason: str) -> None:
        """
        Marca la task como FAILED.

        Args:
            error_code: Código de error ForgeError que causó el fallo.
            reason:     Razón legible del fallo para el audit log.

        Raises:
            InvalidStateTransitionError: Si la transición no es válida.
        """
        _validate_task_transition(self._task_id, self._status, TaskStatus.FAILED)
        self._status = TaskStatus.FAILED
        self._failure_reason = reason
        self._failure_error_code = error_code
        self._completed_at = datetime.now(tz=timezone.utc)

        self._emit(TaskFailed(
            task_id=self._task_id,
            session_id=self._session_id,
            occurred_at=self._completed_at,
            error_code=error_code,
            reason=reason,
        ))

    def cancel(self) -> None:
        """
        Cancela la task.

        Raises:
            InvalidStateTransitionError: Si la task ya está en estado terminal.
        """
        _validate_task_transition(self._task_id, self._status, TaskStatus.CANCELLED)
        self._status = TaskStatus.CANCELLED
        self._completed_at = datetime.now(tz=timezone.utc)

    def clear_domain_events(self) -> list[TaskEvent]:
        """
        Retorna y vacía los eventos de dominio acumulados.

        Returns:
            Lista de eventos de dominio desde la última llamada.
        """
        events = list(self._domain_events)
        self._domain_events.clear()
        return events

    # =========================================================================
    # CONSULTAS
    # =========================================================================

    def get_highest_risk_level(self) -> RiskLevel:
        """
        Retorna el nivel de riesgo más alto de todas las ejecuciones.

        Returns:
            El RiskLevel máximo entre todas las StepExecution.
            NONE si no hay ejecuciones.
        """
        if not self._executions:
            return RiskLevel.NONE
        return max(
            e.risk_assessment.effective_risk
            for e in self._executions
        )

    def to_summary_dict(self) -> dict[str, Any]:
        """Serializa la task para el audit log y el debug CLI."""
        return {
            "task_id": self._task_id.to_str(),
            "session_id": self._session_id.to_str(),
            "origin_turn_id": self._origin_turn_id.to_str(),
            "intent_type": self._intent.type.name,
            "status": self._status.value,
            "has_plan": self._plan is not None,
            "plan_steps": self._plan.step_count if self._plan else 0,
            "executions_count": len(self._executions),
            "evidence_count": len(self._evidence),
            "artifact_count": len(self._artifact_ids),
            "created_at": self._created_at.isoformat(),
            "completed_at": (
                self._completed_at.isoformat() if self._completed_at else None
            ),
            "duration_seconds": self.duration_seconds,
            "failure_reason": self._failure_reason,
            "highest_risk_level": self.get_highest_risk_level().name,
        }

    # =========================================================================
    # HELPERS INTERNOS
    # =========================================================================

    def _emit(self, event: TaskEvent) -> None:
        """Acumula un evento de dominio para publicación posterior."""
        self._domain_events.append(event)

    def __repr__(self) -> str:
        return (
            f"Task("
            f"id={self._task_id.to_str()!r}, "
            f"status={self._status.value!r}, "
            f"intent={self._intent.type.name!r}, "
            f"executions={len(self._executions)}"
            f")"
        )