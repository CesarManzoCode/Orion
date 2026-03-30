"""
Value objects de presupuesto de tokens del dominio HiperForge User.

Este módulo define la lógica de gestión del context window del LLM como
objetos del dominio puro. El context window es un recurso finito y crítico:
si el contexto excede el límite del modelo, la invocación falla. Si es
demasiado corto, el agente pierde contexto conversacional y produce respuestas
de menor calidad.

El ContextBuilder (capa de aplicación) usa estos value objects para:
  1. Calcular cuántos tokens tiene disponibles para el historial.
  2. Decidir cuántos turns del historial caben sin compactar.
  3. Determinar si se necesita compactación antes de la siguiente invocación.
  4. Distribuir el budget entre los diferentes componentes del contexto.

Relación con forge_core/llm/protocol.py:
  - forge_core define TokenBudget genérico (protocolo de red con el LLM)
  - este módulo define ContextTokenBudget (lógica de dominio del producto)
  La diferencia: el protocolo define la interfaz; el dominio define las reglas
  de negocio de distribución y compactación específicas de HiperForge User.

Estrategia de distribución del context window:

  ┌─────────────────────────────────────────────────────────┐
  │  CONTEXT WINDOW TOTAL (ej: 128,000 tokens)              │
  ├─────────────────────────────────────────────────────────┤
  │  System prompt         ~2,000 tokens  (fijo)            │
  │  Tool schemas          ~500-3,000     (varía por tools) │
  │  User profile summary  ~300           (fijo)            │
  │  Response reserved     ~4,096         (fijo)            │
  ├─────────────────────────────────────────────────────────┤
  │  BUDGET PARA HISTORIAL  = total - todas las reservas    │
  │  (lo que queda para los turns de conversación)          │
  └─────────────────────────────────────────────────────────┘

  Trigger de compactación: cuando historial actual > 85% del budget.
  Al compactar: se resumen los N turns más antiguos en ~500 tokens.

Principios de diseño:
  1. Todos los cálculos son deterministas — dado el mismo estado, siempre
     producen el mismo resultado.
  2. Las estimaciones de tokens usan la heurística de 4 chars/token para
     velocidad. La tokenización exacta se usa solo cuando el modelo lo requiere.
  3. El budget es conservador — se reserva un margen de seguridad del 5%
     sobre los límites declarados para absorber variaciones de tokenización.
  4. La compactación es progresiva — solo se compactan los turns necesarios
     para liberar espacio, nunca toda la conversación.

Jerarquía de tipos:

  TokenEstimator          — estimador rápido de tokens por heurística
  BudgetAllocation        — distribución calculada del context window
  TokenUsageSnapshot      — snapshot del uso actual para una sesión
  CompactionDecision      — decisión de si compactar y cuántos turns
  ContextTokenBudget      — objeto central de gestión del budget
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from forge_core.llm.protocol import TokenBudget


# =============================================================================
# ESTIMADOR DE TOKENS
# =============================================================================


class TokenEstimationMethod(Enum):
    """
    Método usado para estimar el número de tokens en un texto.

    La elección del método impacta la velocidad y precisión del ContextBuilder.
    En producción se usa CHAR_HEURISTIC por velocidad; para casos críticos
    donde la precisión importa (verificación de límites) se usa WORD_HEURISTIC.
    """

    CHAR_HEURISTIC = auto()
    """
    4 caracteres = 1 token (aproximación rápida).
    Error típico: ±15% para texto en español/inglés.
    Latencia: O(n) con factor constante muy bajo.
    Uso: estimación rutinaria en el ContextBuilder.
    """

    WORD_HEURISTIC = auto()
    """
    0.75 tokens por carácter en texto típico, o 1.3 tokens por palabra.
    Más preciso que CHAR_HEURISTIC para texto con muchas palabras largas.
    Error típico: ±8% para texto técnico mezclado.
    Uso: verificación de límites antes de enviar al LLM.
    """

    TIKTOKEN_EXACT = auto()
    """
    Tokenización exacta vía tiktoken (OpenAI) o tokenizador nativo del proveedor.
    Error: 0% (exacto).
    Latencia: requiere carga del tokenizador (primera vez) + O(n).
    Uso: cuando se necesita precisión absoluta (raramente necesario).
    """


class TokenEstimator:
    """
    Estimador de tokens para texto en lenguaje natural y código.

    Implementa múltiples métodos de estimación con diferentes trade-offs
    entre velocidad y precisión. El método por defecto (CHAR_HEURISTIC)
    es suficientemente preciso para la gestión del context window dado
    que el sistema incluye un margen de seguridad del 5%.

    La clase es stateless — puede usarse como singleton.
    """

    # Constantes de estimación calibradas para texto técnico mezclado
    # (español + inglés + código)
    _CHARS_PER_TOKEN_DEFAULT: float = 4.0
    _TOKENS_PER_WORD_TECHNICAL: float = 1.3
    _OVERHEAD_PER_MESSAGE: int = 10
    """Overhead aproximado en tokens por cada mensaje (rol, separadores, etc.)."""

    @classmethod
    def estimate(
        cls,
        text: str,
        method: TokenEstimationMethod = TokenEstimationMethod.CHAR_HEURISTIC,
    ) -> int:
        """
        Estima el número de tokens en un texto.

        Args:
            text:   Texto a estimar.
            method: Método de estimación a usar.

        Returns:
            Estimación del número de tokens (siempre >= 1 si el texto no está vacío).
        """
        if not text:
            return 0

        if method == TokenEstimationMethod.CHAR_HEURISTIC:
            return max(1, math.ceil(len(text) / cls._CHARS_PER_TOKEN_DEFAULT))

        elif method == TokenEstimationMethod.WORD_HEURISTIC:
            word_count = len(text.split())
            return max(1, math.ceil(word_count * cls._TOKENS_PER_WORD_TECHNICAL))

        # TIKTOKEN_EXACT: si no hay tokenizador disponible, fallback a WORD
        return max(1, math.ceil(len(text.split()) * cls._TOKENS_PER_WORD_TECHNICAL))

    @classmethod
    def estimate_message(cls, role: str, content: str) -> int:
        """
        Estima los tokens de un mensaje completo (rol + contenido + overhead).

        Args:
            role:    Rol del mensaje ('user', 'assistant', 'system', 'tool').
            content: Contenido textual del mensaje.

        Returns:
            Estimación de tokens incluyendo el overhead del mensaje.
        """
        content_tokens = cls.estimate(content)
        role_tokens = cls.estimate(role)
        return content_tokens + role_tokens + cls._OVERHEAD_PER_MESSAGE

    @classmethod
    def estimate_tool_schema(cls, schema_json: str) -> int:
        """
        Estima los tokens de un schema JSON de tool.

        Los schemas JSON tienen mayor densidad de tokens que el texto normal
        debido a las llaves, comillas y estructura. Se aplica un factor de
        corrección de 0.85x (más tokens por carácter que texto normal).

        Args:
            schema_json: Schema de la tool serializado como JSON string.

        Returns:
            Estimación de tokens del schema.
        """
        base_estimate = cls.estimate(schema_json)
        # JSON tiene estructura más densa → más tokens por carácter
        corrected = math.ceil(base_estimate * 1.15)
        return corrected

    @classmethod
    def estimate_messages_batch(cls, messages: list[dict[str, str]]) -> int:
        """
        Estima los tokens totales de una lista de mensajes.

        Args:
            messages: Lista de dicts con 'role' y 'content'.

        Returns:
            Estimación total de tokens de todos los mensajes.
        """
        total = 0
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            total += cls.estimate_message(role, content)
        return total


# =============================================================================
# ASIGNACIÓN DE BUDGET
# =============================================================================


@dataclass(frozen=True)
class BudgetAllocation:
    """
    Distribución calculada del context window entre los componentes del contexto.

    El ContextBuilder calcula un BudgetAllocation al inicio de cada invocación
    al LLM para saber exactamente cuántos tokens puede usar en cada componente.
    Es el resultado de aplicar la estrategia de distribución al context window
    total del modelo y la configuración de memoria.

    Todos los valores son en tokens y son inmutables. Si la configuración cambia
    (ej: se añaden más tools activas), se recalcula el BudgetAllocation.
    """

    total_context_window: int
    """Tamaño total del context window del modelo, en tokens."""

    system_prompt: int
    """Tokens asignados al system prompt (personalidad + instrucciones + tools policy)."""

    tool_schemas: int
    """
    Tokens asignados a los schemas de tools disponibles para function calling.
    Varía según cuántas tools estén activas en la sesión actual.
    """

    user_profile: int
    """Tokens asignados al resumen del perfil y preferencias del usuario."""

    response_reserve: int
    """
    Tokens reservados para la respuesta del LLM.
    El LLM NO puede generar más tokens que este valor.
    """

    safety_margin: int
    """
    Margen de seguridad (5% del total) para absorber variaciones de tokenización.
    Se descuenta del available_for_history pero no se usa explícitamente.
    """

    @property
    def fixed_overhead(self) -> int:
        """
        Total de tokens consumidos por componentes fijos (no el historial).

        Returns:
            Suma de system_prompt + tool_schemas + user_profile + response_reserve + safety_margin.
        """
        return (
            self.system_prompt
            + self.tool_schemas
            + self.user_profile
            + self.response_reserve
            + self.safety_margin
        )

    @property
    def available_for_history(self) -> int:
        """
        Tokens disponibles para el historial de conversación.

        Este es el budget real que el ContextBuilder puede usar para incluir
        turns del historial. Es el context window total menos todos los
        componentes fijos.

        Returns:
            Tokens disponibles para el historial. Siempre >= 0.
        """
        return max(0, self.total_context_window - self.fixed_overhead)

    @property
    def is_viable(self) -> bool:
        """
        True si el budget es viable para una conversación real.

        Un budget no viable (available_for_history < 1000) indica que la
        configuración es incorrecta — hay demasiados componentes fijos para
        el context window del modelo seleccionado.

        Returns:
            True si hay suficiente budget para al menos 1000 tokens de historial.
        """
        return self.available_for_history >= 1_000

    def usage_ratio(self, current_history_tokens: int) -> float:
        """
        Ratio de uso del budget de historial [0.0, 1.0+].

        Valores > 1.0 indican que el historial excede el budget — esto no
        debe ocurrir en operación normal (el ContextBuilder lo previene).

        Args:
            current_history_tokens: Tokens actualmente usados por el historial.

        Returns:
            Ratio de uso. 0.85 = 85% del budget usado.
        """
        if self.available_for_history == 0:
            return float("inf")
        return current_history_tokens / self.available_for_history

    def to_forge_token_budget(self) -> TokenBudget:
        """
        Convierte el BudgetAllocation al TokenBudget del protocolo LLM.

        El TokenBudget (forge_core) es el objeto que se incluye en el LLMContext
        para que el adapter del LLM pueda verificar los límites antes de enviar.

        Returns:
            TokenBudget compatible con el protocolo LLM de forge_core.
        """
        return TokenBudget(
            total_context_window=self.total_context_window,
            reserved_for_response=self.response_reserve,
            reserved_for_system=self.system_prompt,
            reserved_for_tools=self.tool_schemas,
            reserved_for_user_profile=self.user_profile,
        )

    def describe(self) -> dict[str, Any]:
        """
        Retorna un diccionario descriptivo del budget para logging y debugging.

        Returns:
            Dict con todos los valores del budget y los ratios calculados.
        """
        return {
            "total_context_window": self.total_context_window,
            "fixed_overhead": self.fixed_overhead,
            "available_for_history": self.available_for_history,
            "breakdown": {
                "system_prompt": self.system_prompt,
                "tool_schemas": self.tool_schemas,
                "user_profile": self.user_profile,
                "response_reserve": self.response_reserve,
                "safety_margin": self.safety_margin,
            },
            "utilization_pct": round(
                (self.fixed_overhead / self.total_context_window) * 100, 1
            ),
        }


# =============================================================================
# SNAPSHOT DE USO DE TOKENS
# =============================================================================


@dataclass(frozen=True)
class TokenUsageSnapshot:
    """
    Snapshot del uso actual de tokens en una sesión conversacional.

    Captura el estado actual del consumo de tokens para que el ContextBuilder
    pueda decidir cuántos turns del historial incluir y si necesita compactar.

    Se crea en cada turno de la conversación al construir el contexto.
    """

    history_tokens_used: int
    """
    Tokens actualmente usados por el historial de conversación incluido
    en el contexto. No incluye el system prompt ni las herramientas.
    """

    turns_included: int
    """
    Número de turns del historial incluidos en el contexto actual.
    Puede ser menor al total de turns de la sesión si se truncó el historial.
    """

    total_turns_in_session: int
    """
    Número total de turns en la sesión (incluyendo los no incluidos).
    Si turns_included < total_turns_in_session, hay turns excluidos.
    """

    has_compacted_summary: bool
    """
    True si el historial incluye un resumen compactado de turns antiguos.
    Indica que parte del contexto histórico fue resumido para liberar tokens.
    """

    compacted_turns_count: int = 0
    """
    Número de turns que fueron compactados en el resumen (si has_compacted_summary=True).
    """

    @property
    def turns_excluded(self) -> int:
        """Número de turns del historial que no están en el contexto actual."""
        return max(0, self.total_turns_in_session - self.turns_included)

    @property
    def has_excluded_turns(self) -> bool:
        """True si hay turns del historial que no están en el contexto."""
        return self.turns_excluded > 0

    def describe(self) -> dict[str, Any]:
        """Serializa el snapshot para logging."""
        return {
            "history_tokens_used": self.history_tokens_used,
            "turns_included": self.turns_included,
            "turns_excluded": self.turns_excluded,
            "total_turns": self.total_turns_in_session,
            "has_compacted_summary": self.has_compacted_summary,
            "compacted_turns": self.compacted_turns_count,
        }


# =============================================================================
# DECISIÓN DE COMPACTACIÓN
# =============================================================================


class CompactionStrategy(Enum):
    """
    Estrategia de compactación del historial conversacional.

    Determina qué turns se compactan cuando el historial excede el budget.
    La estrategia afecta la calidad de la memoria conversacional del agente.
    """

    OLDEST_FIRST = auto()
    """
    Compactar los turns más antiguos primero.
    Preserva el contexto reciente completo — el más relevante para el turno actual.
    Es la estrategia por defecto y la más natural para una conversación lineal.
    """

    SLIDING_WINDOW = auto()
    """
    Mantener siempre los N turns más recientes completos.
    Los turns fuera de la ventana se compactan en un resumen único.
    Buena para sesiones muy largas donde solo el contexto reciente importa.
    """

    IMPORTANCE_BASED = auto()
    """
    Compactar primero los turns de menor importancia semántica.
    Preserva turns con tool_calls, artifacts producidos, y decisiones importantes.
    Requiere un paso adicional de análisis — más costoso pero mejor calidad.
    Reservado para V2.
    """


@dataclass(frozen=True)
class CompactionDecision:
    """
    Decisión del sistema sobre si compactar el historial y cómo.

    El ContextTokenBudget produce un CompactionDecision al evaluar el
    TokenUsageSnapshot contra el BudgetAllocation. Si needs_compaction=True,
    el ContextBuilder llama al MemoryManager para compactar.

    Es immutable — es el resultado de una evaluación puntual del estado.
    """

    needs_compaction: bool
    """True si se requiere compactación antes de la siguiente invocación al LLM."""

    turns_to_compact: int
    """
    Número de turns a compactar.
    0 si needs_compaction=False o si no hay turns elegibles.
    """

    strategy: CompactionStrategy
    """Estrategia de compactación recomendada."""

    urgency: float
    """
    Urgencia de la compactación [0.0, 1.0].
    0.0 = no necesaria, 1.0 = crítica (historial excede el budget).
    """

    reason: str
    """Razón legible de la decisión para logging."""

    tokens_to_free: int = 0
    """Tokens estimados que se liberarán con la compactación."""

    @property
    def is_urgent(self) -> bool:
        """True si la compactación es urgente (historial > 95% del budget)."""
        return self.urgency >= 0.95

    @classmethod
    def no_compaction_needed(cls) -> CompactionDecision:
        """Factory: crea una decisión de no compactación."""
        return cls(
            needs_compaction=False,
            turns_to_compact=0,
            strategy=CompactionStrategy.OLDEST_FIRST,
            urgency=0.0,
            reason="El historial está dentro del budget disponible.",
        )

    @classmethod
    def compact_oldest(
        cls,
        turns_to_compact: int,
        urgency: float,
        tokens_to_free: int,
    ) -> CompactionDecision:
        """
        Factory: crea una decisión de compactación de turns más antiguos.

        Args:
            turns_to_compact: Número de turns a compactar.
            urgency:          Urgencia de la compactación [0.0, 1.0].
            tokens_to_free:   Tokens estimados a liberar.

        Returns:
            CompactionDecision con OLDEST_FIRST strategy.
        """
        return cls(
            needs_compaction=True,
            turns_to_compact=turns_to_compact,
            strategy=CompactionStrategy.OLDEST_FIRST,
            urgency=urgency,
            reason=(
                f"El historial usa el {urgency * 100:.1f}% del budget. "
                f"Compactar los {turns_to_compact} turns más antiguos para liberar "
                f"~{tokens_to_free} tokens."
            ),
            tokens_to_free=tokens_to_free,
        )


# =============================================================================
# CONTEXT TOKEN BUDGET — OBJETO CENTRAL
# =============================================================================


@dataclass(frozen=True)
class ContextTokenBudget:
    """
    Gestor del presupuesto de tokens para el context window del LLM.

    Es el objeto central de la lógica de gestión de tokens del dominio.
    Encapsula la configuración del model (context window total), la configuración
    del sistema (reservas fijas) y produce las decisiones de distribución y
    compactación que el ContextBuilder necesita.

    Diseño:
    - Immutable: todos los cálculos son pure functions del estado inicial.
    - Un ContextTokenBudget se crea por sesión con la configuración del modelo
      y no cambia (salvo cambio de modelo, que requeriría crear uno nuevo).
    - El ContextBuilder lo consulta en cada turno para distribuir el budget
      y decidir compactación.
    """

    # --- Configuración del modelo ---
    model_context_window: int
    """
    Tamaño total del context window del modelo LLM activo, en tokens.
    Ej: 128_000 para GPT-4o, 200_000 para Claude 3.5 Sonnet.
    """

    # --- Reservas fijas (configuradas en MemoryConfig) ---
    system_prompt_reserve: int
    """Tokens reservados para el system prompt."""

    tools_schema_reserve: int
    """
    Tokens reservados para los schemas de tools.
    Debe actualizarse cuando cambian las tools activas en la sesión.
    """

    response_reserve: int
    """Tokens reservados para la respuesta del LLM."""

    user_profile_reserve: int
    """Tokens reservados para el resumen del perfil de usuario."""

    # --- Parámetros de compactación ---
    compaction_threshold_ratio: float = 0.85
    """
    Ratio del budget de historial a partir del cual se activa compactación.
    0.85 = compactar cuando el historial usa el 85% del budget disponible.
    """

    compaction_summary_tokens: int = 500
    """
    Tokens máximos del resumen generado durante compactación.
    El resumen reemplaza los turns compactados en el contexto.
    """

    min_turns_before_compaction: int = 5
    """
    Número mínimo de turns que deben existir antes de considerar compactación.
    Evita compactar en sesiones muy cortas donde el costo no vale la pena.
    """

    safety_margin_ratio: float = 0.05
    """
    Fracción del context window total reservada como margen de seguridad.
    Absorbe variaciones entre la estimación heurística y la tokenización real.
    0.05 = 5% del total.
    """

    def compute_allocation(
        self,
        active_tools_token_estimate: int | None = None,
    ) -> BudgetAllocation:
        """
        Calcula la distribución actual del context window.

        Toma en cuenta los tokens actuales de tools activas para dar un
        BudgetAllocation preciso. Si no se provee la estimación de tools,
        usa tools_schema_reserve como estimación por defecto.

        Args:
            active_tools_token_estimate: Estimación de tokens para los schemas
                                         de tools actualmente activas.
                                         None usa la reserva configurada.

        Returns:
            BudgetAllocation con la distribución completa del context window.
        """
        tools_tokens = active_tools_token_estimate or self.tools_schema_reserve
        safety_tokens = math.ceil(self.model_context_window * self.safety_margin_ratio)

        return BudgetAllocation(
            total_context_window=self.model_context_window,
            system_prompt=self.system_prompt_reserve,
            tool_schemas=tools_tokens,
            user_profile=self.user_profile_reserve,
            response_reserve=self.response_reserve,
            safety_margin=safety_tokens,
        )

    def should_compact(
        self,
        current_history_tokens: int,
        total_turns: int,
        allocation: BudgetAllocation | None = None,
    ) -> CompactionDecision:
        """
        Determina si el historial necesita compactación y cómo.

        Evalúa el uso actual del historial contra el budget disponible.
        Si el ratio supera compaction_threshold_ratio y hay suficientes
        turns para justificar la compactación, produce una CompactionDecision
        con los parámetros recomendados.

        Args:
            current_history_tokens: Tokens actualmente usados por el historial.
            total_turns:            Total de turns en la sesión.
            allocation:             BudgetAllocation pre-calculado.
                                    None → lo calcula internamente.

        Returns:
            CompactionDecision con la recomendación de compactación.
        """
        if allocation is None:
            allocation = self.compute_allocation()

        # Verificar si hay suficientes turns para justificar compactación
        if total_turns < self.min_turns_before_compaction:
            return CompactionDecision.no_compaction_needed()

        # Calcular el ratio de uso actual
        usage_ratio = allocation.usage_ratio(current_history_tokens)

        # Sin necesidad de compactación
        if usage_ratio < self.compaction_threshold_ratio:
            return CompactionDecision.no_compaction_needed()

        # Calcular cuántos turns compactar para liberar suficiente espacio
        # Objetivo: reducir el uso al 70% del budget (margen confortable)
        target_tokens = math.floor(allocation.available_for_history * 0.70)
        tokens_to_free = current_history_tokens - target_tokens

        if tokens_to_free <= 0:
            return CompactionDecision.no_compaction_needed()

        # Estimar cuántos turns representan los tokens a liberar
        # Heurística: tokens_to_free / tokens_por_turn_promedio
        if total_turns > 0 and current_history_tokens > 0:
            avg_tokens_per_turn = current_history_tokens / total_turns
            turns_to_compact = max(
                1,
                math.ceil(tokens_to_free / avg_tokens_per_turn),
            )
        else:
            turns_to_compact = 1

        # No compactar más de la mitad de los turns totales en un solo paso
        turns_to_compact = min(turns_to_compact, total_turns // 2)

        # Urgencia: qué tan lejos estamos del límite
        urgency = min(1.0, usage_ratio)

        return CompactionDecision.compact_oldest(
            turns_to_compact=turns_to_compact,
            urgency=urgency,
            tokens_to_free=tokens_to_free,
        )

    def can_fit_turn(
        self,
        turn_tokens: int,
        current_history_tokens: int,
        allocation: BudgetAllocation | None = None,
    ) -> bool:
        """
        Verifica si un nuevo turn cabe en el budget disponible sin compactación.

        Args:
            turn_tokens:            Tokens del turn a añadir.
            current_history_tokens: Tokens actualmente usados por el historial.
            allocation:             BudgetAllocation pre-calculado.

        Returns:
            True si el turn cabe sin necesidad de compactar.
        """
        if allocation is None:
            allocation = self.compute_allocation()

        return (current_history_tokens + turn_tokens) <= allocation.available_for_history

    def max_turns_from_end(
        self,
        turns_with_tokens: list[tuple[int, int]],
        allocation: BudgetAllocation | None = None,
    ) -> int:
        """
        Calcula cuántos turns (desde el más reciente) caben en el budget.

        Itera los turns desde el más reciente (final de la lista) hacia atrás,
        acumulando tokens hasta llegar al límite del budget. Esto garantiza
        que siempre se incluyen los turns más recientes — los más relevantes
        para el turno actual.

        Args:
            turns_with_tokens: Lista de (turn_index, token_count) ordenada
                               cronológicamente (el más reciente al final).
            allocation:        BudgetAllocation pre-calculado.

        Returns:
            Número de turns (desde el final de la lista) que caben en el budget.
            0 si ninguno cabe. len(turns_with_tokens) si todos caben.
        """
        if allocation is None:
            allocation = self.compute_allocation()

        budget = allocation.available_for_history
        accumulated = 0
        turns_fitting = 0

        # Iterar desde el más reciente hacia atrás
        for _, token_count in reversed(turns_with_tokens):
            if accumulated + token_count > budget:
                break
            accumulated += token_count
            turns_fitting += 1

        return turns_fitting

    def estimate_compaction_summary_fits(
        self,
        current_history_tokens: int,
        allocation: BudgetAllocation | None = None,
    ) -> bool:
        """
        Verifica si un resumen de compactación cabrá en el budget.

        Después de compactar, el resumen reemplaza los turns compactados.
        Este método verifica que el historial resultante (historial original
        - tokens liberados + resumen) cabe en el budget.

        Args:
            current_history_tokens: Tokens actuales del historial.
            allocation:             BudgetAllocation pre-calculado.

        Returns:
            True si el resumen de compactación cabrá en el budget resultante.
        """
        if allocation is None:
            allocation = self.compute_allocation()

        # Después de compactar, el historial resultante tendrá aproximadamente:
        # tokens_actuales * (1 - compaction_ratio) + compaction_summary_tokens
        # donde compaction_ratio es la fracción del historial que se compacta
        estimated_post_compaction = (
            current_history_tokens * 0.6  # asumiendo compactar ~40% del historial
            + self.compaction_summary_tokens
        )

        return estimated_post_compaction <= allocation.available_for_history

    def with_updated_tools_reserve(self, new_tools_tokens: int) -> ContextTokenBudget:
        """
        Crea un nuevo ContextTokenBudget con una estimación de tools actualizada.

        Se usa cuando las tools activas en la sesión cambian (ej: se habilita
        una nueva tool en medio de una sesión).

        Args:
            new_tools_tokens: Nueva estimación de tokens para los schemas de tools.

        Returns:
            Nuevo ContextTokenBudget con la reserva de tools actualizada.
        """
        return ContextTokenBudget(
            model_context_window=self.model_context_window,
            system_prompt_reserve=self.system_prompt_reserve,
            tools_schema_reserve=new_tools_tokens,
            response_reserve=self.response_reserve,
            user_profile_reserve=self.user_profile_reserve,
            compaction_threshold_ratio=self.compaction_threshold_ratio,
            compaction_summary_tokens=self.compaction_summary_tokens,
            min_turns_before_compaction=self.min_turns_before_compaction,
            safety_margin_ratio=self.safety_margin_ratio,
        )

    def describe(self) -> dict[str, Any]:
        """
        Serializa la configuración del budget para logging y debugging.

        Returns:
            Dict con la configuración completa y el allocation calculado.
        """
        allocation = self.compute_allocation()
        return {
            "model_context_window": self.model_context_window,
            "allocation": allocation.describe(),
            "compaction": {
                "threshold_ratio": self.compaction_threshold_ratio,
                "summary_tokens": self.compaction_summary_tokens,
                "min_turns": self.min_turns_before_compaction,
            },
            "safety_margin_ratio": self.safety_margin_ratio,
        }


# =============================================================================
# FACTORIES DE PRESUPUESTO ESTÁNDAR
# =============================================================================


def build_context_budget(
    model_context_window: int,
    *,
    system_prompt_reserve: int = 2_000,
    tools_schema_reserve: int = 3_000,
    response_reserve: int = 4_096,
    user_profile_reserve: int = 300,
    compaction_threshold_ratio: float = 0.85,
    compaction_summary_tokens: int = 500,
    min_turns_before_compaction: int = 5,
    safety_margin_ratio: float = 0.05,
) -> ContextTokenBudget:
    """
    Construye un ContextTokenBudget con la configuración especificada.

    Factory de conveniencia que valida los parámetros antes de crear el objeto.
    Se usa en el bootstrap del sistema para crear el budget de la sesión.

    Args:
        model_context_window:      Tamaño del context window del modelo.
        system_prompt_reserve:     Reserva para el system prompt (tokens).
        tools_schema_reserve:      Reserva para schemas de tools (tokens).
        response_reserve:          Reserva para la respuesta del LLM (tokens).
        user_profile_reserve:      Reserva para el perfil de usuario (tokens).
        compaction_threshold_ratio: Ratio de activación de compactación.
        compaction_summary_tokens:  Longitud máxima del resumen de compactación.
        min_turns_before_compaction: Mínimo de turns antes de compactar.
        safety_margin_ratio:        Margen de seguridad como fracción del total.

    Returns:
        ContextTokenBudget configurado y validado.

    Raises:
        ValueError: Si los parámetros producen un budget no viable
                    (available_for_history < 1000 tokens).
    """
    budget = ContextTokenBudget(
        model_context_window=model_context_window,
        system_prompt_reserve=system_prompt_reserve,
        tools_schema_reserve=tools_schema_reserve,
        response_reserve=response_reserve,
        user_profile_reserve=user_profile_reserve,
        compaction_threshold_ratio=compaction_threshold_ratio,
        compaction_summary_tokens=compaction_summary_tokens,
        min_turns_before_compaction=min_turns_before_compaction,
        safety_margin_ratio=safety_margin_ratio,
    )

    allocation = budget.compute_allocation()
    if not allocation.is_viable:
        total_reserved = allocation.fixed_overhead
        raise ValueError(
            f"La configuración del budget no es viable: "
            f"las reservas fijas ({total_reserved} tokens) dejan solo "
            f"{allocation.available_for_history} tokens para el historial, "
            f"insuficiente para una conversación útil (mínimo 1000). "
            f"Reduce las reservas o usa un modelo con mayor context window "
            f"(actual: {model_context_window} tokens)."
        )

    return budget


# Presupuestos predefinidos para modelos comunes
# Disponibles para uso en el bootstrap sin necesidad de configuración manual

BUDGET_128K = build_context_budget(
    model_context_window=128_000,
    system_prompt_reserve=2_000,
    tools_schema_reserve=3_000,
    response_reserve=4_096,
    user_profile_reserve=300,
)
"""Budget para modelos de 128K tokens (GPT-4o, Claude Sonnet 3.5, etc.)."""

BUDGET_200K = build_context_budget(
    model_context_window=200_000,
    system_prompt_reserve=2_000,
    tools_schema_reserve=3_000,
    response_reserve=4_096,
    user_profile_reserve=300,
)
"""Budget para modelos de 200K tokens (Claude 3.5 Sonnet extended, etc.)."""

BUDGET_32K = build_context_budget(
    model_context_window=32_000,
    system_prompt_reserve=1_500,
    tools_schema_reserve=2_000,
    response_reserve=3_000,
    user_profile_reserve=200,
    compaction_threshold_ratio=0.75,  # compactar antes en ventanas pequeñas
    min_turns_before_compaction=3,
)
"""Budget para modelos de 32K tokens (modelos locales como Llama 3.2, etc.)."""

BUDGET_8K = build_context_budget(
    model_context_window=8_000,
    system_prompt_reserve=800,
    tools_schema_reserve=1_000,
    response_reserve=2_000,
    user_profile_reserve=100,
    compaction_threshold_ratio=0.65,  # compactar agresivamente
    compaction_summary_tokens=200,
    min_turns_before_compaction=2,
    safety_margin_ratio=0.03,
)
"""Budget para modelos de 8K tokens (modelos muy pequeños o limitados)."""