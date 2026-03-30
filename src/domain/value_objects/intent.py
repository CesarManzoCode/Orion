"""
Value objects de intent y routing del dominio HiperForge User.

Este módulo define los tipos que representan la intención del usuario y la
decisión de routing del agente. Son el puente entre la respuesta del LLM y
el camino de ejecución que el ConversationCoordinator elige.

Arquitectura de decisión de routing:

  Usuario escribe un mensaje
        │
        ▼
  LLM genera LLMResponse
        │
        ▼
  IntentRouter analiza la estructura de LLMResponse
        │
        ├── ¿Tiene tool_calls?
        │     └── Sí → ¿Cuántos?
        │           ├── Uno  → RoutingDecision.SingleToolCall
        │           └── Varios → RoutingDecision.MultipleToolCalls
        │                  └── ¿Encadenables? → RoutingDecision.PlanningRequired
        │
        └── No tool_calls → RoutingDecision.DirectReply

El IntentRouter NO clasifica intents usando heurísticas ni ML propio.
El LLM ya decidió si usar tools mediante function calling nativo.
El router interpreta la ESTRUCTURA de la respuesta, no el contenido.

Esta es una decisión de arquitectura fundamental documentada en el ADR-005:
'LLM decides tool use'. El router es un intérprete estructural, no un
clasificador semántico.

Principios de diseño:
  1. RoutingDecision es un tipo algebraico sellado — como PolicyDecision.
     Cada variante lleva exactamente los datos que necesita.
  2. Intent es un value object inmutable. Se construye una vez por turno
     y no se muta durante el procesamiento.
  3. La confidence es un float [0.0, 1.0] que permite al sistema tomar
     decisiones diferentes según la certeza del routing.
  4. Los signals son evidencia auditeable de por qué se tomó la decisión.

Jerarquía de tipos:

  IntentType              — enum de tipos de intent
  Intent                  — value object con tipo, confidence y metadata
  IntentSignal            — señal que contribuyó a la clasificación del intent
  RoutingDecision (base)  — decisión de routing (tipo algebraico)
  ├── DirectReply         — respuesta conversacional pura
  ├── SingleToolCall      — ejecutar una sola tool
  ├── MultipleToolCalls   — ejecutar múltiples tools secuencialmente
  └── PlanningRequired    — generar plan multi-step antes de ejecutar
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from forge_core.llm.protocol import LLMResponse, ToolCall


# =============================================================================
# TIPO DE INTENT
# =============================================================================


class IntentType(Enum):
    """
    Tipo de intención del usuario en un turno de conversación.

    Refleja la naturaleza de lo que el usuario quiere lograr, no el mecanismo
    técnico de cómo se logra. El IntentRouter mapea los tipos de intent a
    RoutingDecisions de ejecución.

    La clasificación es mutua y exhaustiva — todo turno cae en exactamente
    uno de estos tipos.
    """

    CONVERSATIONAL = auto()
    """
    El usuario quiere conversar, preguntar, explorar ideas o recibir
    explicaciones. No requiere tools — el LLM responde directamente con
    su conocimiento.

    Ejemplos:
    - "¿Qué es la fotosíntesis?"
    - "Explícame las diferencias entre Python y Rust"
    - "¿Puedes ayudarme a entender este concepto?"

    Routing: DirectReply siempre.
    LLM behavior: responde sin tool_calls.
    Frecuencia esperada: ~60-70% de los turnos.
    """

    QUERY = auto()
    """
    El usuario quiere información actualizada o específica que requiere
    búsqueda externa o análisis de un documento. Requiere tools de lectura
    (búsqueda web, análisis de documento) pero sin mutación del sistema.

    Ejemplos:
    - "Busca las últimas noticias sobre IA"
    - "Resume este PDF que te adjunté"
    - "¿Qué dice este contrato sobre las cláusulas de terminación?"

    Routing: SingleToolCall o MultipleToolCalls (web_search, document_read).
    LLM behavior: emite tool_calls de solo lectura.
    Frecuencia esperada: ~20-25% de los turnos.
    """

    SINGLE_ACTION = auto()
    """
    El usuario quiere ejecutar una acción concreta en el sistema. Requiere
    una sola tool de mutación o acción.

    Ejemplos:
    - "Abre Firefox"
    - "Copia esto al portapapeles"
    - "Guarda este resumen en un archivo"

    Routing: SingleToolCall (tool de mutación, puede requerir approval).
    LLM behavior: emite exactamente un tool_call.
    Frecuencia esperada: ~5-10% de los turnos.
    """

    MULTI_STEP = auto()
    """
    El usuario quiere lograr un objetivo complejo que requiere planificación
    y ejecución de múltiples pasos en secuencia.

    Ejemplos:
    - "Busca papers sobre RL, léelos y dame un resumen comparativo"
    - "Investiga 3 laptops y compáralas en una tabla"

    Routing: PlanningRequired → plan multi-step → ejecución secuencial.
    LLM behavior: puede emitir múltiples tool_calls o indicar necesidad de plan.
    Frecuencia esperada: ~3-7% de los turnos.
    """

    HYBRID = auto()
    """
    Combina conversación con acción. El usuario quiere que el agente
    haga algo Y explique o comente al respecto.

    Ejemplos:
    - "Busca sobre X y luego explícame qué encontraste"
    - "Resume este documento y dime qué partes debería estudiar más"

    Routing: SingleToolCall o MultipleToolCalls + respuesta textual enriquecida.
    LLM behavior: emite tool_calls Y produce contenido textual rico.
    Frecuencia esperada: ~5-10% de los turnos.
    """

    UNKNOWN = auto()
    """
    Intent no determinado. Ocurre cuando la respuesta del LLM tiene una
    estructura inesperada o ambigua. El sistema hace su mejor esfuerzo
    con la información disponible.

    Routing: depende de la estructura de la respuesta; default a DirectReply.
    """


# =============================================================================
# SEÑALES DE CLASIFICACIÓN
# =============================================================================


class IntentSignalSource(Enum):
    """Fuente de una señal que contribuyó a la clasificación del intent."""

    TOOL_CALLS_PRESENT = auto()
    """El LLM incluyó tool_calls en su respuesta."""

    NO_TOOL_CALLS = auto()
    """El LLM respondió sin tool_calls (respuesta conversacional)."""

    MULTIPLE_TOOL_CALLS = auto()
    """El LLM incluyó múltiples tool_calls en una sola respuesta."""

    MUTATION_TOOL_DETECTED = auto()
    """Al menos uno de los tool_calls invoca una tool de mutación."""

    READ_ONLY_TOOLS_ONLY = auto()
    """Todos los tool_calls son de tools de solo lectura."""

    FINISH_REASON_STOP = auto()
    """La respuesta terminó con FinishReason.STOP (respuesta completa)."""

    FINISH_REASON_TOOL_CALLS = auto()
    """La respuesta terminó con FinishReason.TOOL_CALLS."""

    HAS_TEXTUAL_CONTENT = auto()
    """La respuesta contiene contenido textual además de (o en lugar de) tool_calls."""

    COMPLEXITY_THRESHOLD_EXCEEDED = auto()
    """El número de tool_calls supera el umbral de complejidad para planificación."""

    CHAINED_DEPENDENCIES_DETECTED = auto()
    """Los tool_calls tienen dependencias entre sí (output de uno es input de otro)."""


@dataclass(frozen=True)
class IntentSignal:
    """
    Señal que contribuyó a la determinación del intent y el routing.

    El conjunto de señales provee trazabilidad auditeable de por qué el
    IntentRouter tomó una decisión específica. Crucial para debugging cuando
    el routing produce resultados inesperados.
    """

    source: IntentSignalSource
    """Fuente de la señal."""

    weight: float
    """
    Peso de esta señal en la decisión final (0.0 - 1.0).
    Señales de mayor peso tienen más influencia en la clasificación.
    """

    description: str
    """Descripción legible de qué se detectó y por qué es relevante."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Datos adicionales de la señal para debugging."""


# =============================================================================
# INTENT VALUE OBJECT
# =============================================================================


@dataclass(frozen=True)
class Intent:
    """
    Intención del usuario en un turno de conversación.

    Value object inmutable que encapsula el tipo de intent, la confianza
    del sistema en esa clasificación, y las señales que llevaron a ella.

    El Intent se construye una vez por turno, en el IntentRouter, y se
    propaga a través del flujo de ejecución como contexto de auditoría.
    No se muta durante el procesamiento — es evidencia del estado inicial.
    """

    type: IntentType
    """
    Tipo de intent clasificado para este turno.
    """

    confidence: float
    """
    Confianza del sistema en la clasificación [0.0, 1.0].

    1.0 = certeza total (ej: LLM retornó tool_calls → QUERY o SINGLE_ACTION).
    0.5 = moderada certeza (ej: comportamiento ambiguo del LLM).
    0.0 = sin certeza (fallback).

    La confianza se usa para decisiones de degradación graceful:
    con confianza < 0.5 en MULTI_STEP, el sistema puede ejecutar
    directamente en lugar de planificar.
    """

    signals: tuple[IntentSignal, ...]
    """
    Señales que contribuyeron a la clasificación, en orden de peso descendente.
    Proporciona trazabilidad completa de la decisión.
    """

    raw_description: str
    """
    Descripción humana del intent detectado para logging y debugging.
    No se muestra al usuario — es contexto interno del sistema.
    """

    tool_call_count: int = 0
    """
    Número de tool_calls presentes en la respuesta del LLM que originó este intent.
    Zero para intents conversacionales puros.
    """

    has_textual_content: bool = True
    """
    True si la respuesta del LLM contiene contenido textual además de tool_calls.
    Relevante para el HYBRID intent donde coexisten texto y acciones.
    """

    def is_action_required(self) -> bool:
        """
        Indica si este intent requiere ejecución de alguna tool.

        Returns:
            True si el intent es QUERY, SINGLE_ACTION, MULTI_STEP, o HYBRID.
            False solo para CONVERSATIONAL y UNKNOWN sin tool_calls.
        """
        return self.type in {
            IntentType.QUERY,
            IntentType.SINGLE_ACTION,
            IntentType.MULTI_STEP,
            IntentType.HYBRID,
        }

    def requires_planning(self) -> bool:
        """
        Indica si este intent requiere planificación multi-step.

        Returns:
            True solo para MULTI_STEP con confianza suficiente.
        """
        return self.type == IntentType.MULTI_STEP and self.confidence >= 0.6

    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """
        Indica si la clasificación tiene confianza alta.

        Args:
            threshold: Umbral de confianza (por defecto 0.8).

        Returns:
            True si confidence >= threshold.
        """
        return self.confidence >= threshold

    def to_log_dict(self) -> dict[str, Any]:
        """
        Serializa el intent para logging estructurado.

        Returns:
            Diccionario con los campos relevantes del intent para el audit log.
        """
        return {
            "intent_type": self.type.name,
            "intent_confidence": round(self.confidence, 3),
            "intent_tool_calls": self.tool_call_count,
            "intent_has_text": self.has_textual_content,
            "intent_description": self.raw_description,
            "intent_signals": [
                {
                    "source": s.source.name,
                    "weight": s.weight,
                    "description": s.description,
                }
                for s in self.signals
            ],
        }

    @classmethod
    def conversational(cls) -> Intent:
        """
        Factory: crea un Intent conversacional de alta confianza.

        Usado cuando el LLM responde sin tool_calls — es la clasificación
        más directa y con mayor confianza del sistema.
        """
        return cls(
            type=IntentType.CONVERSATIONAL,
            confidence=1.0,
            signals=(
                IntentSignal(
                    source=IntentSignalSource.NO_TOOL_CALLS,
                    weight=1.0,
                    description="LLM respondió sin invocar tools — respuesta conversacional pura.",
                ),
            ),
            raw_description="Respuesta conversacional directa del LLM sin uso de tools.",
            tool_call_count=0,
            has_textual_content=True,
        )

    @classmethod
    def single_tool(
        cls,
        tool_call: ToolCall,
        *,
        is_mutation: bool = False,
    ) -> Intent:
        """
        Factory: crea un Intent para ejecución de una sola tool.

        Args:
            tool_call:   El ToolCall que el LLM emitió.
            is_mutation: True si la tool produce mutación del sistema.
        """
        intent_type = IntentType.SINGLE_ACTION if is_mutation else IntentType.QUERY
        description = (
            f"LLM solicitó una {'acción' if is_mutation else 'consulta'} "
            f"via tool '{tool_call.name}'."
        )

        signals = [
            IntentSignal(
                source=IntentSignalSource.TOOL_CALLS_PRESENT,
                weight=1.0,
                description=f"LLM emitió exactamente 1 tool_call: '{tool_call.name}'.",
                metadata={"tool_name": tool_call.name},
            ),
        ]

        if is_mutation:
            signals.append(IntentSignal(
                source=IntentSignalSource.MUTATION_TOOL_DETECTED,
                weight=0.9,
                description=f"La tool '{tool_call.name}' produce mutación del sistema.",
                metadata={"tool_name": tool_call.name},
            ))
        else:
            signals.append(IntentSignal(
                source=IntentSignalSource.READ_ONLY_TOOLS_ONLY,
                weight=0.9,
                description=f"La tool '{tool_call.name}' es de solo lectura.",
            ))

        return cls(
            type=intent_type,
            confidence=1.0,
            signals=tuple(signals),
            raw_description=description,
            tool_call_count=1,
            has_textual_content=False,
        )

    @classmethod
    def multiple_tools(
        cls,
        tool_calls: list[ToolCall],
        *,
        needs_planning: bool = False,
        has_textual_content: bool = False,
    ) -> Intent:
        """
        Factory: crea un Intent para múltiples tool_calls.

        Args:
            tool_calls:         Lista de ToolCalls emitidos por el LLM.
            needs_planning:     True si la complejidad requiere planificación.
            has_textual_content: True si el LLM también produjo contenido textual.
        """
        tool_names = [tc.name for tc in tool_calls]
        intent_type = IntentType.MULTI_STEP if needs_planning else IntentType.QUERY

        if has_textual_content:
            intent_type = IntentType.HYBRID

        signals: list[IntentSignal] = [
            IntentSignal(
                source=IntentSignalSource.TOOL_CALLS_PRESENT,
                weight=1.0,
                description=f"LLM emitió {len(tool_calls)} tool_calls.",
                metadata={"tool_names": tool_names, "count": len(tool_calls)},
            ),
            IntentSignal(
                source=IntentSignalSource.MULTIPLE_TOOL_CALLS,
                weight=0.95,
                description=f"Múltiples tool_calls detectados: {', '.join(tool_names)}.",
            ),
        ]

        if needs_planning:
            signals.append(IntentSignal(
                source=IntentSignalSource.COMPLEXITY_THRESHOLD_EXCEEDED,
                weight=0.8,
                description=(
                    f"El número de tool_calls ({len(tool_calls)}) supera el umbral "
                    f"de complejidad para planificación directa."
                ),
                metadata={"tool_count": len(tool_calls)},
            ))

        if has_textual_content:
            signals.append(IntentSignal(
                source=IntentSignalSource.HAS_TEXTUAL_CONTENT,
                weight=0.7,
                description="LLM produjo contenido textual además de tool_calls (HYBRID).",
            ))

        description = (
            f"LLM solicitó {len(tool_calls)} tools "
            f"({'con planificación' if needs_planning else 'ejecución directa'}): "
            f"{', '.join(tool_names)}."
        )

        return cls(
            type=intent_type,
            confidence=0.9 if needs_planning else 1.0,
            signals=tuple(signals),
            raw_description=description,
            tool_call_count=len(tool_calls),
            has_textual_content=has_textual_content,
        )


# =============================================================================
# ROUTING DECISION — TIPO ALGEBRAICO
# =============================================================================


@dataclass(frozen=True)
class RoutingDecision:
    """
    Clase base de la decisión de routing del IntentRouter.

    Como PolicyDecision, es un tipo algebraico sellado implementado
    como frozen dataclasses. Cada variante lleva exactamente los datos
    que el flujo de ejecución necesita.

    El ConversationCoordinator hace pattern matching sobre RoutingDecision
    para elegir el camino de ejecución:

        decision = intent_router.route(llm_response)
        match decision:
            case DirectReply(content=c):
                return format_response(c)
            case SingleToolCall(tool_call=tc):
                result = await task_executor.execute_single(tc)
                ...
            case MultipleToolCalls(tool_calls=tcs):
                results = await task_executor.execute_sequential(tcs)
                ...
            case PlanningRequired(analysis=a):
                plan = await planner.plan(a)
                ...
    """

    intent: Intent
    """Intent clasificado que llevó a esta decisión de routing."""

    def is_direct_reply(self) -> bool:
        """True si la decisión es una respuesta directa sin tools."""
        return isinstance(self, DirectReply)

    def is_tool_execution(self) -> bool:
        """True si la decisión requiere ejecución de una o más tools."""
        return isinstance(self, (SingleToolCall, MultipleToolCalls))

    def is_planning(self) -> bool:
        """True si la decisión requiere planificación antes de ejecutar."""
        return isinstance(self, PlanningRequired)

    def requires_llm_followup(self) -> bool:
        """
        True si después de ejecutar la acción se requiere una segunda llamada
        al LLM para sintetizar la respuesta final para el usuario.

        Las respuestas directas no requieren followup — el contenido es la
        respuesta final. Las tool executions siempre requieren un segundo
        turno del LLM para interpretar los resultados.
        """
        return not isinstance(self, DirectReply)


@dataclass(frozen=True)
class DirectReply(RoutingDecision):
    """
    El LLM respondió directamente sin invocar ninguna tool.

    El contenido de la respuesta es la respuesta final para el usuario.
    No se requiere ninguna acción adicional del executor.

    Flujo:
        DirectReply → formatear content → enviar al usuario → FIN
    """

    content: str
    """
    Contenido textual de la respuesta del LLM, listo para presentar al usuario.
    Puede contener markdown — la UI lo renderiza apropiadamente.
    """

    finish_reason: str = "stop"
    """Finish reason del LLM que produjo esta respuesta."""

    def __str__(self) -> str:
        preview = self.content[:60].replace("\n", " ")
        return f"DirectReply(preview={preview!r}...)"


@dataclass(frozen=True)
class SingleToolCall(RoutingDecision):
    """
    El LLM solicitó la ejecución de exactamente una tool.

    El executor ejecuta el tool_call (pasando por PolicyEngine y
    ApprovalWorkflow si es necesario) y luego invoca al LLM nuevamente
    con el resultado para generar la respuesta final.

    Flujo:
        SingleToolCall → PolicyEngine → (Approval?) → ToolDispatch
                      → resultado → LLM (síntesis) → usuario → FIN
    """

    tool_call: ToolCall
    """El tool_call específico que el LLM quiere ejecutar."""

    def __str__(self) -> str:
        return f"SingleToolCall(tool={self.tool_call.name!r})"


@dataclass(frozen=True)
class MultipleToolCalls(RoutingDecision):
    """
    El LLM solicitó la ejecución de múltiples tools.

    Las tools se ejecutan secuencialmente (no en paralelo en V1, aunque
    el ToolDispatch está diseñado para soportar paralelismo en V2).
    Después de todas las ejecuciones, el LLM sintetiza la respuesta final.

    Este routing se usa cuando el LLM emite 2-N tool_calls en una sola
    respuesta, lo que es posible en providers que soportan parallel tool use
    (OpenAI gpt-4o, Claude con tool batching).

    Flujo:
        MultipleToolCalls → para cada tool_call:
                              PolicyEngine → (Approval?) → ToolDispatch
                          → todos los resultados → LLM (síntesis) → usuario
    """

    tool_calls: tuple[ToolCall, ...]
    """
    Tupla de tool_calls a ejecutar, en el orden en que el LLM los emitió.
    Inmutable para garantizar que el orden no cambie durante el procesamiento.
    """

    def __str__(self) -> str:
        names = [tc.name for tc in self.tool_calls]
        return f"MultipleToolCalls(tools={names!r})"

    @property
    def count(self) -> int:
        """Número de tool_calls en esta decisión."""
        return len(self.tool_calls)


@dataclass(frozen=True)
class PlanningRequired(RoutingDecision):
    """
    La tarea es suficientemente compleja como para requerir planificación
    antes de ejecutar.

    El LightweightPlanner genera un Plan multi-step, y el TaskExecutor
    lo ejecuta paso a paso. Este es el routing menos frecuente y más costoso.

    Se activa cuando:
    - El número de tool_calls supera el umbral de complejidad configurado.
    - El intent es MULTI_STEP con confianza suficiente.
    - El LLM explícitamente indica que necesita un plan (rare).

    Flujo:
        PlanningRequired → LightweightPlanner.plan() → Plan
                        → TaskExecutor.execute_plan(plan)
                        → para cada step: PolicyEngine → (Approval?) → ToolDispatch
                        → síntesis LLM → usuario → FIN
    """

    initial_tool_calls: tuple[ToolCall, ...]
    """
    Tool_calls iniciales emitidos por el LLM, si los hay.
    El Planner los usa como semilla para generar el plan completo.
    Puede ser vacío si el intent se detectó por análisis del mensaje.
    """

    complexity_hint: str = ""
    """
    Descripción de por qué se requiere planificación.
    Usada por el LightweightPlanner como contexto adicional.
    """

    def __str__(self) -> str:
        count = len(self.initial_tool_calls)
        return f"PlanningRequired(seed_tool_calls={count}, hint={self.complexity_hint!r})"


# =============================================================================
# ANALIZADOR DE RESPUESTA LLM
# =============================================================================


class LLMResponseAnalyzer:
    """
    Analizador de la estructura de LLMResponse para extraer el RoutingDecision.

    Esta clase implementa la lógica central del IntentRouter: interpreta la
    estructura de la respuesta del LLM (tiene tool_calls, cuántos, qué tipo)
    y produce el RoutingDecision apropiado.

    NO es un clasificador semántico — no analiza el texto de la respuesta.
    Solo examina la estructura: tool_calls presentes, finish_reason, y la
    presencia/ausencia de contenido textual.

    La instancia es stateless — puede usarse como singleton de forma segura
    en entornos async con múltiples sesiones concurrentes.
    """

    def __init__(
        self,
        planning_threshold: int = 2,
        mutation_tool_ids: frozenset[str] | None = None,
    ) -> None:
        """
        Inicializa el analizador con parámetros de configuración.

        Args:
            planning_threshold:  Número mínimo de tool_calls para activar
                                 planificación. Por defecto 2: si el LLM emite
                                 3+ tool_calls, se considera multi-step.
                                 Con 1-2 se ejecuta directamente.
            mutation_tool_ids:   Conjunto de IDs de tools que producen mutación.
                                 Usado para clasificar SingleToolCall como
                                 SINGLE_ACTION vs QUERY.
                                 None = tratar todas como no-mutación (conservador).
        """
        self._planning_threshold = planning_threshold
        self._mutation_tool_ids: frozenset[str] = mutation_tool_ids or frozenset()

    def analyze(self, llm_response: LLMResponse) -> RoutingDecision:
        """
        Analiza una LLMResponse y produce el RoutingDecision apropiado.

        Algoritmo de decisión:
          1. ¿Tiene tool_calls?
             No → DirectReply
             Sí → continuar
          2. ¿Cuántos tool_calls?
             0 → DirectReply (redundante con 1, por seguridad)
             1 → SingleToolCall
             2..N → evaluar umbral de planificación
          3. Si N > planning_threshold → PlanningRequired
             Si N <= planning_threshold → MultipleToolCalls
          4. Si hay contenido textual ADEMÁS de tool_calls → ajustar a HYBRID

        Args:
            llm_response: Respuesta completa del LLM a analizar.

        Returns:
            RoutingDecision apropiado para esta respuesta.
        """
        # Caso 1: Sin tool_calls → respuesta conversacional directa
        if not llm_response.has_tool_calls:
            intent = Intent.conversational()
            return DirectReply(
                intent=intent,
                content=llm_response.content,
                finish_reason=llm_response.finish_reason.value,
            )

        tool_calls = llm_response.tool_calls
        has_text = bool(llm_response.content.strip())

        # Caso 2: Exactamente un tool_call
        if len(tool_calls) == 1:
            tool_call = tool_calls[0]
            is_mutation = tool_call.name in self._mutation_tool_ids
            intent = Intent.single_tool(tool_call, is_mutation=is_mutation)
            return SingleToolCall(
                intent=intent,
                tool_call=tool_call,
            )

        # Caso 3: Múltiples tool_calls
        needs_planning = len(tool_calls) > self._planning_threshold

        if needs_planning:
            intent = Intent.multiple_tools(
                tool_calls,
                needs_planning=True,
                has_textual_content=has_text,
            )
            return PlanningRequired(
                intent=intent,
                initial_tool_calls=tuple(tool_calls),
                complexity_hint=(
                    f"LLM emitió {len(tool_calls)} tool_calls, "
                    f"superando el umbral de planificación de {self._planning_threshold}."
                ),
            )

        # Múltiples tool_calls dentro del umbral → ejecutar directamente
        intent = Intent.multiple_tools(
            tool_calls,
            needs_planning=False,
            has_textual_content=has_text,
        )
        return MultipleToolCalls(
            intent=intent,
            tool_calls=tuple(tool_calls),
        )

    def extract_intent_only(self, llm_response: LLMResponse) -> Intent:
        """
        Extrae solo el Intent de una LLMResponse sin construir el RoutingDecision.

        Útil para logging y métricas cuando solo se necesita el intent
        sin el overhead de construir el routing completo.

        Args:
            llm_response: Respuesta del LLM.

        Returns:
            Intent clasificado.
        """
        decision = self.analyze(llm_response)
        return decision.intent

    def update_mutation_tools(self, mutation_tool_ids: frozenset[str]) -> LLMResponseAnalyzer:
        """
        Crea un nuevo analizador con el conjunto de mutation tools actualizado.

        Como el analizador es stateless, este método retorna una nueva instancia
        en lugar de mutar la existente. Útil cuando el ToolRegistry se actualiza
        en runtime (añadiendo nuevas tools).

        Args:
            mutation_tool_ids: Nuevo conjunto de IDs de tools de mutación.

        Returns:
            Nueva instancia de LLMResponseAnalyzer con la configuración actualizada.
        """
        return LLMResponseAnalyzer(
            planning_threshold=self._planning_threshold,
            mutation_tool_ids=mutation_tool_ids,
        )