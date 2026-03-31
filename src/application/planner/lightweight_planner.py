"""
LightweightPlanner - Genera planes de ejecución simples pero efectivos.
Responsabilidades:
  1. Analizar la intención del usuario
  2. Generar una secuencia de pasos (tareas)
  3. Validar que los pasos sean ejecutables
  4. Retornar el plan para ejecutar
Este es un "planner ligero" — no usa constraint solving ni formalismos
complejos de planificación. Usa LLM para generar pasos razonables.
"""
from __future__ import annotations
import json
from typing import Any
from forge_core.observability.logging import get_logger
from src.domain.entities.task import Task
from src.domain.value_objects.identifiers import SessionId, UserId
from src.domain.value_objects.intent import IntentClassification
from src.ports.outbound.llm_port import LLMPort, LLMRequest
from src.ports.outbound.tool_port import ToolRegistryPort
logger = get_logger(__name__, component="lightweight_planner")
class LightweightPlanner:
    """
    Genera planes de ejecución usando el LLM.
    Dado una intención del usuario, genera una secuencia de tareas
    que pueden ejecutarse para cumplir con lo solicitado.
    """
    def __init__(
        self,
        llm_adapter: LLMPort,
        tool_registry: ToolRegistryPort,
    ) -> None:
        """
        Args:
            llm_adapter: Adaptador LLM para generar planes.
            tool_registry: Registro de herramientas disponibles.
        """
        self.llm_adapter = llm_adapter
        self.tool_registry = tool_registry
    async def plan(
        self,
        intent: IntentClassification,
        context: str = "",
    ) -> list[Task]:
        """
        Genera un plan (lista de tareas) para cumplir la intención.
        Args:
            intent: Intención clasificada del usuario.
            context: Contexto adicional (conversación anterior, etc).
        Returns:
            Lista de tareas ordenadas para ejecutar.
        """
        # Si es direct_reply, no necesita plan
        if intent.intent_type.name == "DIRECT_REPLY":
            logger.debug("no_plan_needed_for_direct_reply")
            return []
        # Obtener lista de herramientas disponibles
        available_tools = self.tool_registry.list_tools()
        tools_description = "\n".join([
            f"- {tool.tool_id}: {tool.description}"
            for tool in available_tools[:20]  # Limitar para no saturar al LLM
        ])
        system_prompt = f"""Eres un planificador de tareas para un asistente IA.
Dada una intención del usuario, genera un plan (lista de pasos/tareas) para cumplirla.
Herramientas disponibles:
{tools_description}
Retorna SOLO un JSON válido (sin markdown) con:
{{
  "plan": [
    {{"task_id": "0", "tool_id": "...", "description": "...", "parameters": {{}}}},
    {{"task_id": "1", "tool_id": "...", "description": "...", "parameters": {{}}}}
  ],
  "reasoning": "explicación del plan"
}}
Reglas:
- Máximo 5 tareas por plan
- Cada tarea debe usar una herramienta disponible
- Las tareas deben ser ordenadas secuencialmente
- Parámetros deben ser compatibles con la herramienta
"""
        user_prompt = f"""Intención:
- Tipo: {intent.intent_type.name}
- Confianza: {intent.confidence}
- Entidades: {", ".join(intent.entities) if intent.entities else "ninguna"}
Contexto:
{context if context else "(no hay contexto)"}
Genera un plan para cumplir esta intención."""
        llm_request = LLMRequest(
            system_prompt=system_prompt,
            user_message=user_prompt,
            temperature=0.3,
            max_tokens=500,
        )
        try:
            response = await self.llm_adapter.generate_completion(llm_request)
            # Parsear JSON
            response_text = response.text.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1].strip("json").strip()
            plan_dict = json.loads(response_text)
            # Convertir plan JSON a objetos Task
            tasks = []
            for step in plan_dict.get("plan", []):
                try:
                    task = Task.create(
                        tool_id=step.get("tool_id", ""),
                        parameters=step.get("parameters", {}),
                        description=step.get("description", ""),
                    )
                    tasks.append(task)
                except Exception as exc:
                    logger.warning(
                        "plan_task_creation_failed",
                        step=step,
                        error=str(exc),
                    )
            logger.info(
                "plan_generated",
                task_count=len(tasks),
                reasoning=plan_dict.get("reasoning", ""),
            )
            return tasks
        except Exception as exc:
            logger.warning(
                "plan_generation_fallback",
                error=str(exc),
                intent_type=intent.intent_type.name,
            )
            # Fallback: retornar lista vacía (no se ejecutará nada si no hay plan)
            return []
