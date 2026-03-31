"""
AnthropicAdapter — adaptador de la API de Anthropic para HiperForge User.

Implementa UserLLMPort usando el SDK oficial de Anthropic (anthropic-sdk-python).
Traduce entre el protocolo interno de forge_core y el protocolo de la API de
Anthropic (Messages API con tool use).

Responsabilidades:
  1. Traducir ConversationContext → anthropic.MessageParam[]
  2. Traducir ToolDefinition[] → anthropic.ToolParam[]
  3. Llamar a client.messages.create() con los parámetros correctos
  4. Parsear la respuesta y producir LLMResponse del dominio
  5. Manejar streaming con client.messages.stream()
  6. Parsear PlanningResult desde JSON en la respuesta del planning
  7. Convertir errores de Anthropic → ForgeError con is_retryable correcto
  8. Implementar retry con exponential backoff para rate limit errors

Configuración:
  - api_key: desde FORGE_USER__LLM__API_KEY (via config schema)
  - model: claude-sonnet-4-5 por defecto (configurable)
  - max_tokens: 8192 por defecto
  - timeout: 120s por defecto
  - max_retries: 3

Protocolo Anthropic Messages API:
  - Los mensajes van en el campo 'messages' como lista de dicts
  - El system prompt va en el campo 'system' separado
  - Las tools van en el campo 'tools' como lista de ToolParam
  - Los tool_use blocks en la respuesta tienen tipo 'tool_use'
  - Los tool_result blocks se envían como user messages
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from forge_core.errors.types import (
    ForgeLLMError,
    LLMConnectionError,
    LLMContentFilterError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMTokenLimitError,
)
from forge_core.llm.protocol import (
    FinishReason,
    LLMMessage,
    LLMResponse,
    MessageRole,
    StreamChunk,
    ToolCall,
    TokenUsage,
)
from forge_core.observability.logging import get_logger

from src.domain.value_objects.intent import LLMResponseAnalyzer, RoutingDecision
from src.ports.outbound.llm_port import (
    CompactionContext,
    ConversationContext,
    PlannedStepSpec,
    PlanningContext,
    PlanningResult,
    SynthesisContext,
    SystemPromptBuilder,
    UserLLMPort,
)


logger = get_logger(__name__, component="anthropic_adapter")

# Modelos disponibles de Anthropic para HiperForge User
_DEFAULT_MODEL = "claude-sonnet-4-5"
_FALLBACK_MODEL = "claude-haiku-4-5-20251001"

# Parámetros por defecto
_DEFAULT_MAX_TOKENS = 8_192
_DEFAULT_TIMEOUT_SECONDS = 120.0
_DEFAULT_MAX_RETRIES = 3
_RETRY_BASE_DELAY_SECONDS = 1.0
_RETRY_MAX_DELAY_SECONDS = 30.0

# Temperatura por operación
_TEMP_CONVERSATION = 0.7
_TEMP_SYNTHESIS = 0.5
_TEMP_PLANNING = 0.1
_TEMP_COMPACTION = 0.1


class AnthropicAdapter(UserLLMPort):
    """
    Adaptador de la API de Anthropic para HiperForge User.

    Implementa UserLLMPort usando anthropic-sdk-python de forma asíncrona.
    Maneja la traducción completa entre el protocolo interno del sistema
    y el protocolo de la Messages API de Anthropic.

    Uso típico:
        adapter = AnthropicAdapter(api_key="sk-ant-...", model="claude-sonnet-4-5")
        response = await adapter.generate_conversation(context)

    El adapter es stateless — puede manejar múltiples sesiones concurrentes.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        base_url: str | None = None,
    ) -> None:
        """
        Inicializa el adapter de Anthropic.

        Args:
            api_key:          Clave de API de Anthropic.
            model:            Modelo a usar ('claude-sonnet-4-5', etc.).
            max_tokens:       Máximo de tokens en la respuesta.
            timeout_seconds:  Timeout de la request en segundos.
            max_retries:      Máximo de reintentos en caso de rate limit.
            base_url:         URL base alternativa (para proxies o tests).
        """
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "El paquete 'anthropic' no está instalado. "
                "Instálalo con: pip install anthropic"
            ) from e

        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._system_prompt_builder = SystemPromptBuilder()
        self._response_analyzer = LLMResponseAnalyzer()

        # Crear cliente asíncrono de Anthropic
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout_seconds,
            "max_retries": 0,  # manejamos retries manualmente para más control
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = anthropic.AsyncAnthropic(**client_kwargs)

        logger.info(
            "anthropic_adapter_inicializado",
            model=model,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )

    # =========================================================================
    # UserLLMPort — operaciones principales
    # =========================================================================

    async def generate_conversation(
        self,
        context: ConversationContext,
    ) -> LLMResponse:
        """
        Genera una respuesta para un turno conversacional usando la Messages API.

        Construye el request con el historial, system prompt, tools, y
        llama a client.messages.create() con retry automático.

        Args:
            context: ConversationContext completo del turno.

        Returns:
            LLMResponse con la respuesta del modelo.

        Raises:
            ForgeLLMError: Si la API falla o se supera el timeout.
        """
        system_prompt, messages = self._build_messages_from_context(context)
        tools = self._build_tool_params(context.available_tools)

        request_params: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system_prompt,
            "messages": messages,
            "temperature": _TEMP_CONVERSATION,
        }

        if tools:
            request_params["tools"] = tools

        return await self._call_with_retry(request_params)

    async def generate_conversation_streaming(
        self,
        context: ConversationContext,
    ) -> AsyncIterator[StreamChunk]:
        """
        Genera una respuesta en streaming usando la Messages API de Anthropic.

        Usa client.messages.stream() para emitir chunks progresivos.
        El último StreamChunk tiene is_final=True con la respuesta completa.

        Args:
            context: ConversationContext completo del turno.

        Yields:
            StreamChunk con fragmentos progresivos.
        """
        system_prompt, messages = self._build_messages_from_context(context)
        tools = self._build_tool_params(context.available_tools)

        request_params: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system_prompt,
            "messages": messages,
            "temperature": _TEMP_CONVERSATION,
        }

        if tools:
            request_params["tools"] = tools

        try:
            accumulated_text = ""
            final_response: LLMResponse | None = None

            async with self._client.messages.stream(**request_params) as stream:
                async for text_delta in stream.text_stream:
                    accumulated_text += text_delta
                    yield StreamChunk(
                        chunk_type="text_delta",
                        text_delta=text_delta,
                    )

                # Obtener el mensaje final con toda la metadata
                final_message = await stream.get_final_message()
                final_response = self._parse_response(final_message)

            if final_response:
                yield StreamChunk(
                    chunk_type="final",
                    is_final=True,
                    final_response=final_response,
                )

        except Exception as exc:
            forge_error = self._convert_error(exc)
            logger.error("streaming_fallo", error=forge_error)
            # Emitir chunk de error
            from src.ports.inbound.conversation_port import AssistantResponse, AssistantResponseType
            from src.domain.value_objects.identifiers import SessionId
            dummy_response = LLMResponse(
                content="Error durante el streaming.",
                finish_reason=FinishReason.ERROR,
                usage=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
                model=self._model,
            )
            yield StreamChunk(
                chunk_type="error",
                error_message=str(forge_error),
                is_final=True,
                final_response=dummy_response,
            )
            raise forge_error

    async def generate_plan(
        self,
        context: PlanningContext,
    ) -> PlanningResult:
        """
        Genera un plan multi-step usando el LLM con modo JSON estricto.

        Construye el system prompt de planificación con el formato JSON
        esperado e instrucciones estrictas. Parsea la respuesta JSON y
        la convierte en un PlanningResult tipado.

        Args:
            context: PlanningContext con la descripción de la tarea.

        Returns:
            PlanningResult con la lista de PlannedStepSpec.

        Raises:
            ForgeLLMError: Si la API falla.
            ValueError:    Si el LLM retorna JSON malformado.
        """
        # System prompt específico para planning con formato JSON estricto
        planning_system = self._system_prompt_builder.build_planning_prompt(
            max_steps=context.max_steps,
            available_tool_ids=[t.name for t in context.available_tools],
        )

        # Mensaje de planning como usuario
        planning_user_message = (
            f"Genera un plan para completar esta tarea:\n\n{context.task_description}"
        )
        if context.conversation_summary:
            planning_user_message += (
                f"\n\nContexto de conversación:\n{context.conversation_summary}"
            )

        tools = self._build_tool_params(context.available_tools)

        request_params: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 2_048,  # el plan no debe ser muy largo
            "system": planning_system,
            "messages": [{"role": "user", "content": planning_user_message}],
            "temperature": _TEMP_PLANNING,
        }

        if tools:
            request_params["tools"] = tools

        raw_response = await self._call_with_retry(request_params)

        # Parsear el JSON del plan
        return self._parse_planning_response(raw_response)

    async def generate_synthesis(
        self,
        context: SynthesisContext,
    ) -> LLMResponse:
        """
        Genera una síntesis de resultados de tools.

        Sin tool schemas en el request — las tools ya se ejecutaron.
        El LLM solo necesita sintetizar los resultados y responder al usuario.

        Args:
            context: SynthesisContext con los resultados de tools.

        Returns:
            LLMResponse con la síntesis para el usuario.
        """
        profile_summary = context.profile_summary
        system_prompt = self._system_prompt_builder.build(
            profile_summary,
            available_tool_count=0,  # sin tools en síntesis
            include_tool_instructions=False,
        )

        # Construir el historial de mensajes para la síntesis
        messages = self._build_history_from_list(context.conversation_history)

        # Añadir los resultados de tools como mensajes de la conversación
        tool_results_messages = self._build_tool_results_messages(context.tool_results)
        messages.extend(tool_results_messages)

        # Instrucción de síntesis como último mensaje del usuario
        synthesis_instruction = (
            f"Basándote en los resultados anteriores, responde al usuario: "
            f"'{context.original_user_message}'"
        )
        if context.task_description:
            synthesis_instruction += f"\n\nContexto de la tarea: {context.task_description}"

        messages.append({"role": "user", "content": synthesis_instruction})

        request_params: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system_prompt,
            "messages": messages,
            "temperature": _TEMP_SYNTHESIS,
        }

        return await self._call_with_retry(request_params)

    async def generate_compaction_summary(
        self,
        context: CompactionContext,
    ) -> str:
        """
        Genera un resumen de compactación del historial.

        Temperatura muy baja (0.1) para un resumen fiel y determinista.

        Args:
            context: CompactionContext con los turns a compactar.

        Returns:
            String con el resumen generado.
        """
        compaction_system = self._system_prompt_builder.build_compaction_prompt(
            max_tokens=context.max_summary_tokens,
            language=context.language,
        )

        # Convertir los turns a compactar en mensajes
        messages = self._build_history_from_list(context.turns_to_compact)

        # Instrucción de resumen
        messages.append({
            "role": "user",
            "content": (
                f"Por favor, genera un resumen conciso de estos {len(context.turns_to_compact)} "
                f"mensajes de conversación. Máximo {context.max_summary_tokens} tokens."
            ),
        })

        request_params: dict[str, Any] = {
            "model": self._model,
            "max_tokens": context.max_summary_tokens * 2,  # margen para el resumen
            "system": compaction_system,
            "messages": messages,
            "temperature": _TEMP_COMPACTION,
        }

        response = await self._call_with_retry(request_params)
        return response.content.strip()

    async def count_tokens(self, text: str) -> int:
        """
        Estima el número de tokens en un texto.

        En V1 usa heurística rápida (4 chars/token).
        En V2 usará la API de tokenización de Anthropic si está disponible.

        Args:
            text: Texto a tokenizar.

        Returns:
            Estimación de tokens.
        """
        from src.domain.value_objects.token_budget import TokenEstimator
        return TokenEstimator.estimate(text)

    async def health_check(self) -> bool:
        """
        Verifica que la API de Anthropic esté disponible.

        Hace una request mínima con max_tokens=1 para verificar conectividad.

        Returns:
            True si la API responde correctamente.
        """
        try:
            await self._client.messages.create(
                model=self._model,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except Exception as exc:
            logger.warning("health_check_fallo", error=str(exc))
            return False

    def get_routing_decision(
        self,
        llm_response: LLMResponse,
        *,
        mutation_tool_ids: frozenset[str],
        planning_threshold: int,
    ) -> RoutingDecision:
        """
        Interpreta la LLMResponse y retorna la RoutingDecision apropiada.

        Delega al LLMResponseAnalyzer del dominio que contiene la lógica
        de clasificación de respuestas.

        Args:
            llm_response:       Respuesta del LLM a interpretar.
            mutation_tool_ids:  IDs de tools que producen mutación.
            planning_threshold: Número de tool calls para activar planificación.

        Returns:
            RoutingDecision con el flujo de ejecución recomendado.
        """
        return self._response_analyzer.analyze(
            llm_response,
            mutation_tool_ids=mutation_tool_ids,
            planning_threshold=planning_threshold,
        )

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def max_context_tokens(self) -> int:
        """Retorna el context window del modelo activo."""
        context_windows = {
            "claude-opus-4-6": 200_000,
            "claude-sonnet-4-6": 200_000,
            "claude-sonnet-4-5": 200_000,
            "claude-haiku-4-5-20251001": 200_000,
        }
        return context_windows.get(self._model, 200_000)

    # =========================================================================
    # CONSTRUCCIÓN DE MENSAJES (protocolo Anthropic)
    # =========================================================================

    def _build_messages_from_context(
        self,
        context: ConversationContext,
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Convierte el ConversationContext en (system_prompt, messages) para Anthropic.

        La API de Anthropic requiere:
        - system: string con el system prompt
        - messages: lista de dicts con role (user|assistant) y content

        El primer mensaje en 'messages' debe ser de role 'user'.
        No puede haber dos mensajes consecutivos del mismo role.

        Args:
            context: ConversationContext completo.

        Returns:
            Tuple de (system_prompt, messages_list).
        """
        system_prompt = ""
        messages: list[dict[str, Any]] = []

        for msg in context.conversation_history:
            if msg.role == MessageRole.SYSTEM:
                # El sistema de Anthropic tiene el system prompt separado
                system_prompt += msg.get_text() + "\n"
            elif msg.role == MessageRole.USER:
                content = self._build_user_content(msg)
                messages.append({"role": "user", "content": content})
            elif msg.role == MessageRole.ASSISTANT:
                content = self._build_assistant_content(msg)
                messages.append({"role": "assistant", "content": content})
            elif msg.role == MessageRole.TOOL:
                # Los resultados de tools se añaden como mensajes de usuario en Anthropic
                tool_result_content = self._build_tool_result_content(msg)
                if messages and messages[-1]["role"] == "user":
                    # Añadir al mensaje de usuario existente
                    if isinstance(messages[-1]["content"], list):
                        messages[-1]["content"].extend(tool_result_content)
                    else:
                        messages[-1]["content"] = [
                            {"type": "text", "text": messages[-1]["content"]}
                        ] + tool_result_content
                else:
                    messages.append({"role": "user", "content": tool_result_content})

        # Garantizar que el primer mensaje sea de usuario (requerimiento de Anthropic)
        if messages and messages[0]["role"] != "user":
            messages.insert(0, {
                "role": "user",
                "content": "[Inicio de conversación]",
            })

        # Garantizar alternancia user/assistant (requerimiento de Anthropic)
        messages = self._ensure_alternating_roles(messages)

        return system_prompt.strip(), messages

    def _build_user_content(
        self,
        msg: LLMMessage,
    ) -> str | list[dict[str, Any]]:
        """
        Construye el content de un mensaje de usuario para Anthropic.

        Si el mensaje tiene solo texto, retorna string simple.
        Si tiene imágenes, retorna lista de bloques de contenido.

        Args:
            msg: LLMMessage de tipo USER.

        Returns:
            String o lista de bloques de contenido multimodal.
        """
        # Verificar si hay contenido multimodal
        if hasattr(msg, "content_blocks") and msg.content_blocks:
            blocks = []
            for block in msg.content_blocks:
                if hasattr(block, "image_base64") and block.image_base64:
                    blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": getattr(block, "media_type", "image/jpeg"),
                            "data": block.image_base64,
                        },
                    })
                elif hasattr(block, "text"):
                    blocks.append({"type": "text", "text": block.text})
            if blocks:
                return blocks

        return msg.get_text() or ""

    def _build_assistant_content(
        self,
        msg: LLMMessage,
    ) -> str | list[dict[str, Any]]:
        """
        Construye el content de un mensaje del asistente para Anthropic.

        Si el mensaje tiene tool calls, incluye los bloques tool_use.

        Args:
            msg: LLMMessage de tipo ASSISTANT.

        Returns:
            String o lista de bloques de contenido con tool_use si aplica.
        """
        text = msg.get_text() or ""
        tool_calls = getattr(msg, "tool_calls", []) or []

        if not tool_calls:
            return text

        # Mensaje con tool calls: combinar texto + bloques tool_use
        blocks: list[dict[str, Any]] = []
        if text:
            blocks.append({"type": "text", "text": text})

        for tc in tool_calls:
            blocks.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            })

        return blocks

    def _build_tool_result_content(
        self,
        msg: LLMMessage,
    ) -> list[dict[str, Any]]:
        """
        Construye el bloque tool_result para Anthropic.

        En la API de Anthropic, los resultados de tools se envían como
        bloques tool_result dentro de un mensaje de usuario.

        Args:
            msg: LLMMessage de tipo TOOL.

        Returns:
            Lista con el bloque tool_result.
        """
        return [{
            "type": "tool_result",
            "tool_use_id": msg.tool_call_id or "",
            "content": msg.get_text() or "",
        }]

    def _build_tool_results_messages(
        self,
        tool_results: list[Any],
    ) -> list[dict[str, Any]]:
        """
        Convierte los ToolResult del dominio en mensajes Anthropic.

        Los tool_results se agrupan en un mensaje de usuario con
        múltiples bloques tool_result.

        Args:
            tool_results: Lista de ToolResult del dominio.

        Returns:
            Lista de mensajes Anthropic (típicamente uno con múltiples bloques).
        """
        if not tool_results:
            return []

        tool_result_blocks = []
        for result in tool_results:
            tool_result_blocks.append({
                "type": "tool_result",
                "tool_use_id": result.tool_call_id,
                "content": result.content,
            })

        return [{"role": "user", "content": tool_result_blocks}]

    def _build_history_from_list(
        self,
        messages: list[LLMMessage],
    ) -> list[dict[str, Any]]:
        """
        Convierte una lista de LLMMessage en mensajes Anthropic.

        Args:
            messages: Lista de LLMMessage del dominio.

        Returns:
            Lista de mensajes en formato Anthropic.
        """
        result = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                continue  # el system va separado
            elif msg.role == MessageRole.USER:
                content = self._build_user_content(msg)
                result.append({"role": "user", "content": content})
            elif msg.role == MessageRole.ASSISTANT:
                content = self._build_assistant_content(msg)
                result.append({"role": "assistant", "content": content})
            elif msg.role == MessageRole.TOOL:
                tool_content = self._build_tool_result_content(msg)
                result.append({"role": "user", "content": tool_content})

        return self._ensure_alternating_roles(result)

    def _build_tool_params(
        self,
        tool_definitions: list[Any],
    ) -> list[dict[str, Any]]:
        """
        Convierte ToolDefinition[] del dominio en parámetros de tools de Anthropic.

        El formato Anthropic requiere:
        {
            "name": str,
            "description": str,
            "input_schema": { "type": "object", "properties": {...}, "required": [...] }
        }

        Args:
            tool_definitions: Lista de ToolDefinition del dominio.

        Returns:
            Lista de dicts en formato ToolParam de Anthropic.
        """
        if not tool_definitions:
            return []

        tool_params = []
        for td in tool_definitions:
            param: dict[str, Any] = {
                "name": td.name,
                "description": td.description,
                "input_schema": td.parameters or {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            }
            tool_params.append(param)

        return tool_params

    @staticmethod
    def _ensure_alternating_roles(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Garantiza que los mensajes alternen entre user y assistant.

        La API de Anthropic requiere estricta alternancia. Si hay dos
        mensajes consecutivos del mismo role, los fusiona en uno.

        Args:
            messages: Lista de mensajes posiblemente con roles repetidos.

        Returns:
            Lista con roles estrictamente alternados.
        """
        if not messages:
            return messages

        merged: list[dict[str, Any]] = []
        for msg in messages:
            if merged and merged[-1]["role"] == msg["role"]:
                # Fusionar con el mensaje anterior
                prev = merged[-1]
                prev_content = prev["content"]
                curr_content = msg["content"]

                if isinstance(prev_content, str) and isinstance(curr_content, str):
                    prev["content"] = prev_content + "\n" + curr_content
                elif isinstance(prev_content, list) and isinstance(curr_content, list):
                    prev["content"] = prev_content + curr_content
                elif isinstance(prev_content, str) and isinstance(curr_content, list):
                    prev["content"] = [{"type": "text", "text": prev_content}] + curr_content
                elif isinstance(prev_content, list) and isinstance(curr_content, str):
                    prev["content"] = prev_content + [{"type": "text", "text": curr_content}]
            else:
                merged.append(dict(msg))  # copia para no mutar el original

        return merged

    # =========================================================================
    # PARSING DE RESPUESTAS
    # =========================================================================

    def _parse_response(self, message: Any) -> LLMResponse:
        """
        Convierte una respuesta de la API de Anthropic en LLMResponse del dominio.

        Procesa el campo 'content' que puede contener bloques de texto y
        bloques tool_use, convirtiéndolos en el formato interno del sistema.

        Args:
            message: Respuesta de la API de Anthropic (anthropic.types.Message).

        Returns:
            LLMResponse del dominio.
        """
        text_content = ""
        tool_calls: list[ToolCall] = []

        for block in message.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        # Mapear stop_reason de Anthropic a FinishReason del dominio
        finish_reason = self._map_finish_reason(
            message.stop_reason,
            has_tool_calls=bool(tool_calls),
        )

        # Construir TokenUsage
        usage = TokenUsage(
            input_tokens=message.usage.input_tokens if message.usage else 0,
            output_tokens=message.usage.output_tokens if message.usage else 0,
            total_tokens=(
                (message.usage.input_tokens + message.usage.output_tokens)
                if message.usage else 0
            ),
        )

        return LLMResponse(
            content=text_content,
            finish_reason=finish_reason,
            usage=usage,
            model=message.model,
            tool_calls=tool_calls if tool_calls else None,
        )

    def _parse_planning_response(self, response: LLMResponse) -> PlanningResult:
        """
        Parsea la respuesta JSON del planning en un PlanningResult tipado.

        El LLM debería retornar JSON puro según el system prompt de planning.
        Maneja casos donde el JSON viene envuelto en markdown code blocks.

        Args:
            response: LLMResponse con el JSON del plan.

        Returns:
            PlanningResult con PlannedStepSpec[] validados.

        Raises:
            ValueError: Si el JSON es malformado o el plan no es válido.
        """
        raw_text = response.content.strip()

        # Limpiar posibles markdown code blocks
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            raw_text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            plan_data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"El LLM retornó JSON malformado en el planning: {e}\n"
                f"Respuesta recibida: {raw_text[:500]}"
            ) from e

        # Validar estructura básica
        steps_data = plan_data.get("steps", [])
        if not steps_data:
            raise ValueError("El plan del LLM no contiene pasos (steps vacío).")

        # Convertir a PlannedStepSpec
        steps = []
        for i, step_data in enumerate(steps_data):
            try:
                step = PlannedStepSpec(
                    description=step_data.get("description", f"Paso {i + 1}"),
                    tool_id=step_data["tool_id"],
                    tool_arguments=step_data.get("tool_arguments", {}),
                    is_optional=step_data.get("is_optional", False),
                    depends_on_steps=step_data.get("depends_on_steps", []),
                    rationale=step_data.get("rationale", ""),
                )
                steps.append(step)
            except KeyError as e:
                raise ValueError(
                    f"El paso {i} del plan no tiene el campo requerido: {e}"
                ) from e

        return PlanningResult(
            steps=steps,
            plan_rationale=plan_data.get("plan_rationale", ""),
            estimated_duration_seconds=plan_data.get("estimated_duration_seconds"),
            raw_response=response,
        )

    @staticmethod
    def _map_finish_reason(
        stop_reason: str | None,
        *,
        has_tool_calls: bool,
    ) -> FinishReason:
        """
        Mapea el stop_reason de Anthropic a FinishReason del dominio.

        Valores de Anthropic:
        - "end_turn": respuesta completa
        - "tool_use": el modelo quiere usar una tool
        - "max_tokens": se alcanzó el límite de tokens
        - "stop_sequence": se encontró la secuencia de stop
        - None: desconocido

        Args:
            stop_reason:   stop_reason de la API de Anthropic.
            has_tool_calls: True si la respuesta incluye tool calls.

        Returns:
            FinishReason del dominio.
        """
        if has_tool_calls or stop_reason == "tool_use":
            return FinishReason.TOOL_CALLS
        if stop_reason == "max_tokens":
            return FinishReason.LENGTH
        if stop_reason == "end_turn":
            return FinishReason.STOP
        return FinishReason.STOP

    # =========================================================================
    # LLAMADAS A LA API CON RETRY
    # =========================================================================

    async def _call_with_retry(
        self,
        request_params: dict[str, Any],
    ) -> LLMResponse:
        """
        Llama a la API de Anthropic con retry automático.

        Implementa exponential backoff para rate limit errors.
        Convierte todos los errores de Anthropic en ForgeError.

        Args:
            request_params: Parámetros del request para messages.create().

        Returns:
            LLMResponse parseada.

        Raises:
            ForgeLLMError: En caso de error no recuperable o agotados los retries.
        """
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                start_time = time.perf_counter()
                message = await self._client.messages.create(**request_params)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                logger.debug(
                    "anthropic_api_ok",
                    model=self._model,
                    attempt=attempt + 1,
                    elapsed_ms=round(elapsed_ms, 2),
                    input_tokens=message.usage.input_tokens if message.usage else 0,
                    output_tokens=message.usage.output_tokens if message.usage else 0,
                )

                return self._parse_response(message)

            except Exception as exc:
                forge_error = self._convert_error(exc)
                last_error = forge_error

                # Solo reintentar en rate limit y errores de servicio
                if not isinstance(forge_error, (LLMRateLimitError,)) or attempt >= self._max_retries:
                    raise forge_error

                # Calcular delay con exponential backoff
                delay = min(
                    _RETRY_BASE_DELAY_SECONDS * (2 ** attempt),
                    _RETRY_MAX_DELAY_SECONDS,
                )

                logger.warning(
                    "anthropic_api_retry",
                    attempt=attempt + 1,
                    max_retries=self._max_retries,
                    delay_seconds=delay,
                    error_type=type(forge_error).__name__,
                )

                await asyncio.sleep(delay)

        # Si llegamos aquí, agotamos todos los retries
        raise last_error or ForgeLLMError("Se agotaron los reintentos de la API de Anthropic.")

    def _convert_error(self, exc: Exception) -> ForgeLLMError:
        """
        Convierte una excepción de la librería Anthropic en ForgeLLMError.

        Mapea los tipos de error específicos a las subclases de ForgeLLMError
        del dominio para que los callers puedan manejarlos apropiadamente.

        Args:
            exc: Excepción de la librería Anthropic o del runtime.

        Returns:
            ForgeLLMError tipado correspondiente.
        """
        exc_type = type(exc).__name__
        exc_str = str(exc)

        try:
            import anthropic

            if isinstance(exc, anthropic.RateLimitError):
                return LLMRateLimitError(
                    f"Límite de velocidad de la API de Anthropic alcanzado. {exc_str}",
                    provider="anthropic",
                    retry_after_seconds=60.0,
                )

            if isinstance(exc, anthropic.APITimeoutError):
                return LLMTimeoutError(
                    f"Timeout de la API de Anthropic ({self._timeout}s). {exc_str}",
                    provider="anthropic",
                    timeout_seconds=self._timeout,
                )

            if isinstance(exc, anthropic.APIConnectionError):
                return LLMConnectionError(
                    f"No se pudo conectar a la API de Anthropic. {exc_str}",
                    provider="anthropic",
                )

            if isinstance(exc, anthropic.BadRequestError):
                # Puede ser context length exceeded o content filter
                if "maximum context length" in exc_str.lower() or "too many tokens" in exc_str.lower():
                    return LLMTokenLimitError(
                        f"El contexto excede el límite del modelo. {exc_str}",
                        provider="anthropic",
                        context_tokens=0,
                        model_limit=self.max_context_tokens,
                    )
                return LLMContentFilterError(
                    f"La solicitud fue rechazada por la API. {exc_str}",
                    provider="anthropic",
                )

            if isinstance(exc, anthropic.AuthenticationError):
                return LLMConnectionError(
                    f"Error de autenticación con la API de Anthropic. Verifica tu API key. {exc_str}",
                    provider="anthropic",
                )

        except ImportError:
            pass

        # Errores genéricos de red/timeout
        if "timeout" in exc_str.lower():
            return LLMTimeoutError(
                f"Timeout al llamar a la API. {exc_str}",
                provider="anthropic",
                timeout_seconds=self._timeout,
            )

        if isinstance(exc, ForgeLLMError):
            return exc

        return ForgeLLMError(
            f"Error inesperado al llamar a la API de Anthropic: {exc_type}: {exc_str}",
            provider="anthropic",
        )