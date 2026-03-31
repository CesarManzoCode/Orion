from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError

from forge_core.errors.types import (
    ForgeLLMError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from forge_core.llm.protocol import (
    FinishReason,
    LLMMessage,
    LLMResponse,
    StreamChunk,
    TokenUsage,
    ToolCall,
)
from forge_core.observability.logging import get_logger

from src.domain.value_objects.intent import LLMResponseAnalyzer, RoutingDecision
from src.domain.value_objects.token_budget import TokenEstimator
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


logger = get_logger(__name__, component="openai_adapter")


class OpenAIAdapter(UserLLMPort):
    """Adapter OpenAI (MVP) para UserLLMPort."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str,
        max_tokens: int,
        timeout_seconds: float,
        max_retries: int,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._system_prompt_builder = SystemPromptBuilder()

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout_seconds,
            "max_retries": max_retries,
        }
        if base_url:
            kwargs["base_url"] = base_url

        self._client = AsyncOpenAI(**kwargs)

        logger.info(
            "openai_adapter_inicializado",
            model=model,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )

    async def generate_conversation(self, context: ConversationContext) -> LLMResponse:
        system_prompt = self._system_prompt_builder.build(
            context.profile_summary,
            available_tool_count=len(context.available_tools),
            include_tool_instructions=True,
        )
        messages = self._build_messages_for_conversation(context)
        tools = self._build_tools(context.available_tools)
        return await self._chat_completion(
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            temperature=0.3,
        )

    async def generate_conversation_streaming(
        self,
        context: ConversationContext,
    ) -> AsyncIterator[StreamChunk]:
        # MVP: streaming degradado a respuesta final única.
        response = await self.generate_conversation(context)
        yield StreamChunk(
            is_final=True,
            final_response=response,
            finish_reason=response.finish_reason,
        )

    async def generate_plan(self, context: PlanningContext) -> PlanningResult:
        planning_system = self._system_prompt_builder.build_planning_prompt(
            max_steps=context.max_steps,
            available_tool_ids=[t.name for t in context.available_tools],
        )
        user_prompt = f"Tarea:\n{context.task_description}\n\nContexto:\n{context.conversation_summary}"
        response = await self._chat_completion(
            system_prompt=planning_system,
            messages=[{"role": "user", "content": user_prompt}],
            tools=None,
            temperature=0.1,
            max_tokens=min(self._max_tokens, 2048),
        )

        try:
            parsed = json.loads(response.content)
            steps: list[PlannedStepSpec] = [
                PlannedStepSpec(
                    description=s.get("description", "Paso"),
                    tool_id=s.get("tool_id", ""),
                    tool_arguments=s.get("tool_arguments", {}),
                    is_optional=bool(s.get("is_optional", False)),
                    depends_on_steps=s.get("depends_on_steps", []),
                    rationale=s.get("rationale", ""),
                )
                for s in parsed.get("steps", [])
            ]
            return PlanningResult(
                steps=steps,
                plan_rationale=parsed.get("plan_rationale", ""),
                estimated_duration_seconds=parsed.get("estimated_duration_seconds"),
                raw_response=response,
            )
        except Exception as exc:
            raise ValueError(f"Plan JSON inválido: {exc}") from exc

    async def generate_synthesis(self, context: SynthesisContext) -> LLMResponse:
        system_prompt = self._system_prompt_builder.build(
            context.profile_summary,
            available_tool_count=0,
            include_tool_instructions=False,
        )
        messages = self._build_history(context.conversation_history)
        for result in context.tool_results:
            messages.append(
                {
                    "role": "tool",
                    "content": result.content,
                    "tool_call_id": result.tool_call_id,
                }
            )
        messages.append(
            {
                "role": "user",
                "content": f"Sintetiza estos resultados para responder: {context.original_user_message}",
            }
        )
        return await self._chat_completion(
            system_prompt=system_prompt,
            messages=messages,
            tools=None,
            temperature=0.2,
        )

    async def generate_compaction_summary(self, context: CompactionContext) -> str:
        system_prompt = self._system_prompt_builder.build_compaction_prompt(
            max_tokens=context.max_summary_tokens,
            language=context.language,
        )
        messages = self._build_history(context.turns_to_compact)
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Resume esta conversación en <= {context.max_summary_tokens} tokens."
                ),
            }
        )
        response = await self._chat_completion(
            system_prompt=system_prompt,
            messages=messages,
            tools=None,
            temperature=0.1,
            max_tokens=min(self._max_tokens, context.max_summary_tokens * 2),
        )
        return response.content.strip()

    async def count_tokens(self, text: str) -> int:
        return TokenEstimator.estimate(text)

    async def health_check(self) -> bool:
        try:
            await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0,
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
        analyzer = LLMResponseAnalyzer(
            planning_threshold=planning_threshold,
            mutation_tool_ids=mutation_tool_ids,
        )
        return analyzer.analyze(llm_response)

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def max_context_tokens(self) -> int:
        # Conservador para la mayoría de modelos modernos de OpenAI.
        return 128_000

    async def _chat_completion(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        req_messages = [{"role": "system", "content": system_prompt}, *messages]
        try:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": req_messages,
                "max_tokens": max_tokens or self._max_tokens,
                "temperature": temperature,
            }
            if tools:
                kwargs["tools"] = tools

            raw = await self._client.chat.completions.create(**kwargs)
        except RateLimitError as exc:
            raise LLMRateLimitError(str(exc), provider="openai", model=self._model) from exc
        except APITimeoutError as exc:
            raise LLMTimeoutError(
                str(exc),
                provider="openai",
                model=self._model,
                timeout_seconds=self._timeout_seconds,
            ) from exc
        except APIConnectionError as exc:
            raise LLMConnectionError(str(exc), provider="openai", model=self._model) from exc
        except ForgeLLMError:
            raise
        except Exception as exc:
            raise LLMConnectionError(str(exc), provider="openai", model=self._model) from exc

        choice = raw.choices[0]
        message = choice.message

        finish = self._map_finish_reason(choice.finish_reason)
        tool_calls = self._parse_tool_calls(message.tool_calls)
        text = message.content or ""

        usage = raw.usage
        token_usage = TokenUsage(
            prompt_tokens=(usage.prompt_tokens if usage else 0),
            completion_tokens=(usage.completion_tokens if usage else 0),
            total_tokens=(usage.total_tokens if usage else 0),
        )

        return LLMResponse(
            content=text,
            tool_calls=tool_calls,
            finish_reason=finish,
            usage=token_usage,
            model=raw.model or self._model,
            provider="openai",
            request_id=getattr(raw, "id", None),
        )

    def _build_messages_for_conversation(self, context: ConversationContext) -> list[dict[str, Any]]:
        messages = self._build_history(context.conversation_history)

        if context.tool_results:
            for result in context.tool_results:
                messages.append(
                    {
                        "role": "tool",
                        "content": result.content,
                        "tool_call_id": result.tool_call_id,
                    }
                )

        parts = [context.user_message.strip()]
        if context.attachments_text:
            parts.append(f"\n\nAdjuntos:\n{context.attachments_text}")
        if context.compaction_summary:
            parts.append(f"\n\nResumen previo:\n{context.compaction_summary}")

        messages.append({"role": "user", "content": "".join(parts).strip()})
        return messages

    def _build_history(self, history: list[LLMMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in history:
            if msg.role.value == "system":
                continue
            content = self._content_to_text(msg.content)
            payload: dict[str, Any] = {"role": msg.role.value, "content": content}
            if msg.role.value == "tool" and msg.tool_call_id:
                payload["tool_call_id"] = msg.tool_call_id
            out.append(payload)
        return out

    def _build_tools(self, tools: list[Any]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for tool in tools:
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters.model_dump(),
                    },
                }
            )
        return converted

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                text = getattr(item, "text", None)
                if text:
                    chunks.append(text)
            return "\n".join(chunks)
        return str(content)

    @staticmethod
    def _parse_tool_calls(raw_tool_calls: Any) -> list[ToolCall]:
        if not raw_tool_calls:
            return []

        parsed: list[ToolCall] = []
        for call in raw_tool_calls:
            raw_args = call.function.arguments if call.function else "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception:
                args = {}
            parsed.append(
                ToolCall(
                    id=call.id,
                    name=call.function.name if call.function else "unknown_tool",
                    arguments=args,
                )
            )
        return parsed

    @staticmethod
    def _map_finish_reason(reason: str | None) -> FinishReason:
        mapping = {
            "stop": FinishReason.STOP,
            "length": FinishReason.LENGTH,
            "tool_calls": FinishReason.TOOL_CALLS,
            "content_filter": FinishReason.CONTENT_FILTER,
        }
        return mapping.get(reason or "", FinishReason.ERROR)

