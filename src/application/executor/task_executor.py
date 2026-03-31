"""
TaskExecutor - Ejecuta tareas (llamadas a herramientas) en secuencia.
Responsabilidades:
  1. Validar la tarea contra políticas de seguridad
  2. Ejecutar la herramienta mediante el ToolDispatch
  3. Procesar el resultado
  4. Manejar errores y reintentos
  5. Registrar la ejecución en audit log
"""
from __future__ import annotations
from typing import Any
from forge_core.observability.logging import get_logger
from src.domain.entities.task import Task, TaskStatus
from src.domain.value_objects.identifiers import SessionId, UserId
from src.ports.outbound.policy_port import PolicyEnginePort
from src.ports.outbound.tool_port import ToolDispatchPort
logger = get_logger(__name__, component="task_executor")
class TaskExecutor:
    """
    Ejecuta tareas (tool calls) con validación de seguridad.
    Orquesta:
      1. Validación contra PolicyEngine
      2. Ejecución mediante ToolDispatch
      3. Manejo de errores
    """
    def __init__(
        self,
        policy_engine: PolicyEnginePort,
        tool_dispatch: ToolDispatchPort,
    ) -> None:
        """
        Args:
            policy_engine: Motor de políticas para validar permisos.
            tool_dispatch: Despachador de herramientas.
        """
        self.policy_engine = policy_engine
        self.tool_dispatch = tool_dispatch
    async def execute(
        self,
        task: Task,
        user_id: UserId,
        session_id: SessionId,
    ) -> Task:
        """
        Ejecuta una tarea.
        Args:
            task: Tarea a ejecutar.
            user_id: ID del usuario.
            session_id: ID de la sesión.
        Returns:
            Task actualizada con resultados o errores.
        """
        logger.info(
            "task_execution_started",
            task_id=task.task_id.to_str(),
            tool_id=task.tool_id,
            session_id=session_id.to_str(),
        )
        try:
            # 1. Validar contra políticas
            is_allowed = await self.policy_engine.evaluate_tool_call(
                user_id=user_id,
                session_id=session_id,
                tool_id=task.tool_id,
                parameters=task.parameters,
            )
            if not is_allowed:
                logger.warning(
                    "task_denied_by_policy",
                    task_id=task.task_id.to_str(),
                    tool_id=task.tool_id,
                )
                return task.with_error(reason="policy_denied", status=TaskStatus.FAILED)
            # 2. Ejecutar la herramienta
            result = await self.tool_dispatch.execute(
                tool_id=task.tool_id,
                parameters=task.parameters,
                timeout_seconds=30.0,
            )
            if result.is_error:
                logger.error(
                    "task_execution_failed",
                    task_id=task.task_id.to_str(),
                    error=result.error_message,
                )
                return task.with_error(
                    reason=result.error_message,
                    status=TaskStatus.FAILED,
                )
            # 3. Éxito
            logger.info(
                "task_execution_completed",
                task_id=task.task_id.to_str(),
                output_size=len(result.output) if result.output else 0,
            )
            return task.with_result(output=result.output, status=TaskStatus.COMPLETED)
        except Exception as exc:
            logger.error(
                "task_execution_error",
                task_id=task.task_id.to_str(),
                error=str(exc),
            )
            return task.with_error(reason=str(exc), status=TaskStatus.FAILED)
    async def execute_sequence(
        self,
        tasks: list[Task],
        user_id: UserId,
        session_id: SessionId,
        stop_on_error: bool = False,
    ) -> list[Task]:
        """
        Ejecuta una secuencia de tareas en orden.
        Args:
            tasks: Lista de tareas a ejecutar.
            user_id: ID del usuario.
            session_id: ID de la sesión.
            stop_on_error: Si True, detiene si hay error en alguna tarea.
        Returns:
            Lista de tareas ejecutadas.
        """
        results = []
        for task in tasks:
            executed_task = await self.execute(task, user_id, session_id)
            results.append(executed_task)
            if stop_on_error and executed_task.status == TaskStatus.FAILED:
                logger.warning(
                    "task_sequence_stopped_on_error",
                    task_index=len(results),
                    total_tasks=len(tasks),
                )
                break
        return results
