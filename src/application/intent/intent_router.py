"""
IntentRouter - Analiza mensajes del usuario y determina la intención.
Responsabilidades:
  1. Analizar el mensaje del usuario usando el LLM
  2. Extraer la intención clasificada (DirectReply, ToolCall, Plan)
  3. Extraer entidades relevantes (archivos, comandos, etc)
  4. Construir el payload para la siguiente etapa del pipeline
"""
from __future__ import annotations
import json
from typing import Any
from forge_core.observability.logging import get_logger
from src.domain.value_objects.intent import IntentClassification, IntentType
from src.ports.outbound.llm_port import LLMPort, LLMRequest
logger = get_logger(__name__, component="intent_router")
class IntentRouter:
    """
    Enruta análisis de intención usando el LLM.
    Usa prompts estructurados para obtener clasificaciones JSON del LLM.
    """
    def __init__(self, llm_adapter: LLMPort) -> None:
        """
        Args:
            llm_adapter: Adaptador LLM para generar clasificaciones.
        """
        self.llm_adapter = llm_adapter
    async def classify(
        self,
        user_message: str,
        context: str = "",
    ) -> IntentClassification:
        """
        Clasifica la intención del usuario.
        Args:
            user_message: Mensaje del usuario a analizar.
            context: Contexto previo (sesión anterior, etc).
        Returns:
            IntentClassification con el tipo y confianza.
        """
        system_prompt = """Analiza el mensaje del usuario y clasifica su intención.
Retorna SOLO un JSON válido (sin markdown, sin explicaciones) con:
{
  "intent_type": "direct_reply" | "tool_call" | "plan",
  "confidence": 0.0-1.0,
  "entities": ["lista", "de", "entidades"],
  "reasoning": "explicación breve"
}
Reglas:
- direct_reply: Preguntas o conversación general (no requiere herramientas)
- tool_call: Pedir una acción específica (abrir archivo, escribir, calcular)
- plan: Solicitar una secuencia de pasos complejos (proyecto, análisis, etc)
"""
        user_prompt = f"""Mensaje del usuario:
{user_message}
Contexto:
{context if context else "(no hay contexto previo)"}"""
        llm_request = LLMRequest(
            system_prompt=system_prompt,
            user_message=user_prompt,
            temperature=0.3,
            max_tokens=200,
        )
        try:
            response = await self.llm_adapter.generate_completion(llm_request)
            # Parsear JSON
            response_text = response.text.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1].strip("json").strip()
            classification_dict = json.loads(response_text)
            intent_type_str = classification_dict.get("intent_type", "direct_reply")
            intent_type = IntentType[intent_type_str.upper()] if hasattr(IntentType, intent_type_str.upper()) else IntentType.DIRECT_REPLY
            return IntentClassification(
                intent_type=intent_type,
                confidence=float(classification_dict.get("confidence", 0.5)),
                entities=classification_dict.get("entities", []),
                reasoning=classification_dict.get("reasoning", ""),
            )
        except Exception as exc:
            logger.warning(
                "intent_classification_fallback",
                error=str(exc),
                message=user_message[:100],
            )
            # Fallback a direct_reply si hay error
            return IntentClassification(
                intent_type=IntentType.DIRECT_REPLY,
                confidence=0.5,
                entities=[],
                reasoning="Fallback classification due to error",
            )
