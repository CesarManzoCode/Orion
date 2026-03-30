"""
Puerto de salida LLM de HiperForge User.

Este módulo define el contrato específico de integración con el LLM para el
producto HiperForge User. Extiende el protocolo base de forge_core con
operaciones de alto nivel que corresponden a los flujos de la aplicación:

  - generate_conversation: para turnos conversacionales normales
  - generate_plan: para que el LLM genere un plan multi-step
  - generate_synthesis: para que el LLM sintetice resultados de tools
  - generate_compaction_summary: para compactar el historial

La distinción respecto a forge_core/llm/protocol.py:
  - forge_core define LLMPort genérico (protocolo de red con cualquier LLM)
  - este módulo define UserLLMPort (operaciones de negocio de HiperForge User)

Cada operación de UserLLMPort:
  1. Construye el LLMContext apropiado para la operación
  2. Selecciona los parámetros correctos (temperatura, max_tokens, tools)
  3. Maneja los errores con semántica del dominio del producto
  4. Retorna tipos del dominio, no tipos del protocolo LLM

Tipos de contexto enriquecidos:

  ConversationContext    — contexto para un turno normal de conversación
  PlanningContext        — contexto para generar un plan multi-step
  SynthesisContext       — contexto para sintetizar resultados de tools
  CompactionContext      — contexto para generar resumen de compactación

Principios:
  1. Los adapters (OpenAI, Anthropic) implementan forge_core.LLMPort.
  2. Este UserLLMPort usa un forge_core.LLMPort internamente — no hereda de él.
  3. El ConversationCoordinator depende de UserLLMPort, no del protocolo base.
  4. El system prompt de personalidad se gestiona aquí, no en los adapters.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from forge_core.llm.protocol import (
    LLMMessage,
    LLMResponse,
    StreamChunk,
    ToolDefinition,
    ToolResult,
)

from src.domain.entities.artifact import ArtifactSummary
from src.domain.entities.user_profile import ProfileSummary
from src.domain.value_objects.identifiers import RequestId, SessionId, TaskId
from src.domain.value_objects.intent import RoutingDecision
from src.domain.value_objects.token_budget import BudgetAllocation


# =============================================================================
# TIPOS DE CONTEXTO ENRIQUECIDOS
# =============================================================================


@dataclass(frozen=True)
class ConversationContext:
    """
    Contexto completo para un turno conversacional normal.

    Encapsula todo lo que el ConversationCoordinator recopila antes de
    llamar al LLM: el mensaje del usuario, el historial, el perfil, los
    artifacts activos, y las tools disponibles.

    El UserLLMPort usa este contexto para construir el LLMContext final
    que se envía al adapter del proveedor.
    """

    session_id: SessionId
    """ID de la sesión actual."""

    user_message: str
    """
    Mensaje del usuario en este turno.
    Puede incluir texto de archivos adjuntos preprocesados.
    """

    conversation_history: list[LLMMessage]
    """
    Historial de conversación incluyendo turns anteriores.
    El ContextBuilder ya aplicó la estrategia de truncación/compactación.
    La lista está en orden cronológico (el más antiguo primero).
    """

    profile_summary: ProfileSummary
    """Resumen del perfil del usuario para personalización."""

    available_tools: list[ToolDefinition]
    """
    Schemas de las tools disponibles para este turno.
    Lista vacía si no hay tools activas (ej: modo conversacional puro).
    """

    budget_allocation: BudgetAllocation
    """Distribución del context window para este turno."""

    active_artifacts: list[ArtifactSummary] = field(default_factory=list)
    """
    Summaries de artifacts activos de la sesión actual.
    El LLM los recibe como contexto adicional para referencias del usuario.
    """

    tool_results: list[ToolResult] = field(default_factory=list)
    """
    Resultados de tools ejecutadas en este turno (si aplica).
    Se incluyen en la conversación para que el LLM genere la respuesta final.
    """

    compaction_summary: str | None = None
    """
    Resumen compactado del historial antiguo (si el historial fue compactado).
    Se incluye antes del historial reciente para dar contexto continuo.
    """

    request_id: RequestId | None = None
    """ID único de esta request para trazabilidad."""

    task_id: TaskId | None = None
    """ID de la task activa si la hay (para trazabilidad)."""

    attachments_text: str | None = None
    """
    Contenido de archivos adjuntos del usuario preprocesados como texto.
    Los PDFs, DOCXs y textos adjuntos se extraen y se incluyen aquí.
    """


@dataclass(frozen=True)
class PlanningContext:
    """
    Contexto para que el LLM genere un plan multi-step.

    El LightweightPlanner construye este contexto cuando determina que
    una tarea requiere planificación. El LLM recibe la tarea y debe
    retornar un plan estructurado con steps y tools a usar.

    La temperatura se baja para planning (0.2) — se necesita determinismo,
    no creatividad. El formato de respuesta debe ser JSON estricto.
    """

    session_id: SessionId
    """ID de la sesión."""

    task_description: str
    """
    Descripción clara de la tarea que se debe planificar.
    Incluye el mensaje original del usuario y contexto adicional.
    """

    available_tools: list[ToolDefinition]
    """Tools disponibles que el plan puede usar."""

    max_steps: int
    """
    Número máximo de steps en el plan.
    Corresponde a Plan.MAX_STEPS (default: 10).
    """

    conversation_summary: str
    """
    Resumen de la conversación relevante como contexto del planificador.
    Los últimos N turns en texto comprimido.
    """

    profile_summary: ProfileSummary
    """Perfil del usuario para personalizar el plan."""

    request_id: RequestId | None = None
    """ID de la request para trazabilidad."""

    task_id: TaskId | None = None
    """ID de la task para trazabilidad."""


@dataclass(frozen=True)
class SynthesisContext:
    """
    Contexto para que el LLM sintetice los resultados de tools ejecutadas.

    Después de ejecutar una o más tools, el TaskExecutor construye este
    contexto con los resultados para que el LLM genere la respuesta final
    que verá el usuario.

    La síntesis puede incluir resultados de múltiples tools (ej: varias
    búsquedas web + fetch de páginas) y el LLM debe integrarlos coherentemente.
    """

    session_id: SessionId
    """ID de la sesión."""

    original_user_message: str
    """El mensaje original del usuario que originó la ejecución de tools."""

    tool_results: list[ToolResult]
    """
    Resultados de todas las tools ejecutadas.
    El LLM debe sintetizar estos resultados en una respuesta cohesiva.
    """

    conversation_history: list[LLMMessage]
    """Historial relevante de la conversación para contexto."""

    profile_summary: ProfileSummary
    """Perfil del usuario para personalizar la síntesis."""

    budget_allocation: BudgetAllocation
    """Budget del context window."""

    task_description: str = ""
    """
    Descripción de la tarea completada (para que el LLM entienda el contexto).
    Puede ser vacío para ejecuciones directas de una sola tool.
    """

    request_id: RequestId | None = None
    """ID de la request para trazabilidad."""

    task_id: TaskId | None = None
    """ID de la task para trazabilidad."""


@dataclass(frozen=True)
class CompactionContext:
    """
    Contexto para generar un resumen de compactación del historial.

    El MemoryManager construye este contexto cuando necesita compactar
    turns antiguos del historial. El LLM genera un resumen conciso que
    reemplaza esos turns en el contexto futuro.

    La temperatura se baja a 0.1 — el resumen debe ser fiel y determinista.
    No se incluyen tools — solo se necesita texto.
    """

    session_id: SessionId
    """ID de la sesión."""

    turns_to_compact: list[LLMMessage]
    """
    Turns del historial que se van a compactar.
    El LLM debe generar un resumen que preserve la información esencial.
    """

    max_summary_tokens: int
    """
    Longitud máxima del resumen en tokens.
    Corresponde a MemoryConfig.compaction_summary_tokens (default: 500).
    """

    language: str = "es"
    """Idioma del resumen (debe coincidir con el idioma de la conversación)."""

    request_id: RequestId | None = None
    """ID de la request para trazabilidad."""


# =============================================================================
# RESULTADO DE PLANNING
# =============================================================================


@dataclass(frozen=True)
class PlannedStepSpec:
    """
    Especificación de un step del plan retornado por el LLM.

    El LLM retorna el plan como JSON estructurado. Este dataclass representa
    un step individual del plan después de ser parseado y validado.
    """

    description: str
    """Descripción legible del paso para el usuario."""

    tool_id: str
    """ID de la tool a usar en este paso."""

    tool_arguments: dict[str, Any]
    """
    Argumentos sugeridos por el LLM para la tool.
    El TaskExecutor los refina con el PolicyEngine antes de usar.
    """

    is_optional: bool = False
    """True si el paso puede omitirse sin fallar el plan completo."""

    depends_on_steps: list[int] = field(default_factory=list)
    """
    Índices (0-based) de los steps que deben completar antes que este.
    Vacío si no hay dependencias.
    """

    rationale: str = ""
    """
    Razón del LLM para incluir este paso.
    Útil para debugging y para el audit log.
    """


@dataclass(frozen=True)
class PlanningResult:
    """
    Resultado de una operación de planning retornada por UserLLMPort.

    El LightweightPlanner recibe este objeto y lo transforma en una
    entidad Plan del dominio con PlannedSteps.
    """

    steps: list[PlannedStepSpec]
    """Especificaciones de los pasos del plan."""

    plan_rationale: str
    """Razón del LLM para estructurar el plan de esta forma."""

    estimated_duration_seconds: float | None
    """Estimación del tiempo total del plan (opcional, para la UI)."""

    raw_response: LLMResponse
    """Respuesta completa del LLM para audit y debugging."""

    @property
    def step_count(self) -> int:
        """Número de steps en el plan."""
        return len(self.steps)

    @property
    def has_optional_steps(self) -> bool:
        """True si el plan tiene al menos un step opcional."""
        return any(s.is_optional for s in self.steps)


# =============================================================================
# PUERTO ABSTRACTO USER LLM PORT
# =============================================================================


class UserLLMPort(abc.ABC):
    """
    Puerto de salida LLM del producto HiperForge User.

    Define las operaciones de alto nivel que el sistema necesita del LLM,
    mapeadas a los flujos de negocio del producto. Es una capa de abstracción
    sobre forge_core.LLMPort que maneja la construcción de system prompts,
    la serialización del contexto, y el parsing de respuestas estructuradas.

    El ConversationCoordinator, el LightweightPlanner y el MemoryManager
    dependen de este puerto — nunca de forge_core.LLMPort directamente.

    Implementado por: UserLLMAdapter en src/infrastructure/llm/

    Contratos invariantes:
      1. Todas las operaciones son async — nunca bloquean el event loop.
      2. Los errores de infraestructura (red, timeout) se convierten en
         ForgeLLMError antes de propagarse al caller.
      3. El system prompt de personalidad se incluye SIEMPRE en todas las
         operaciones — el LLM siempre conoce su rol y las preferencias del usuario.
      4. El request_id se propaga al adapter para trazabilidad end-to-end.
      5. generate_conversation y generate_conversation_streaming son equivalentes
         en resultado — solo difieren en cómo se entrega (batch vs stream).
    """

    @abc.abstractmethod
    async def generate_conversation(
        self,
        context: ConversationContext,
    ) -> LLMResponse:
        """
        Genera una respuesta para un turno conversacional normal.

        Este es el método principal del sistema. El ConversationCoordinator
        lo llama para cada turno del usuario, incluyendo los turnos que
        resultan en tool calls.

        El LLM recibe:
          - System prompt con personalidad + preferencias del usuario
          - Historial de conversación (compactado si necesario)
          - Contexto de artifacts activos
          - Schemas de tools disponibles
          - Resultados de tools ejecutadas (si es un turno de síntesis)

        Args:
            context: ConversationContext completo para este turno.

        Returns:
            LLMResponse con la respuesta del LLM.
            Si finish_reason=TOOL_CALLS, el LLM quiere ejecutar tools.
            Si finish_reason=STOP, la respuesta es conversacional directa.

        Raises:
            LLMConnectionError:    Sin conexión al provider.
            LLMRateLimitError:     Rate limit alcanzado.
            LLMTokenLimitError:    Contexto excede el context window.
            LLMContentFilterError: Contenido rechazado por el provider.
            LLMTimeoutError:       Timeout de la request.
        """

    @abc.abstractmethod
    async def generate_conversation_streaming(
        self,
        context: ConversationContext,
    ) -> AsyncIterator[StreamChunk]:
        """
        Genera una respuesta en streaming para un turno conversacional.

        Versión streaming de generate_conversation. Emite StreamChunks
        progresivamente para que la UI pueda mostrar la respuesta mientras
        se genera.

        El último chunk tiene is_final=True y final_response con la
        LLMResponse completa acumulada.

        Args:
            context: ConversationContext completo para este turno.

        Yields:
            StreamChunk con fragmentos progresivos de la respuesta.

        Raises:
            Las mismas excepciones que generate_conversation.
        """

    @abc.abstractmethod
    async def generate_plan(
        self,
        context: PlanningContext,
    ) -> PlanningResult:
        """
        Genera un plan multi-step para una tarea compleja.

        El LLM recibe la descripción de la tarea y las tools disponibles,
        y retorna un plan estructurado con los pasos necesarios para completarla.

        La temperatura se usa baja (0.1-0.2) para garantizar que el plan
        sea determinista y estructurado. La respuesta del LLM debe ser JSON
        estricto con el formato del plan.

        Args:
            context: PlanningContext con la descripción de la tarea.

        Returns:
            PlanningResult con la lista de PlannedStepSpecs.
            El LightweightPlanner lo transforma en entidades Plan del dominio.

        Raises:
            Las mismas excepciones que generate_conversation, más:
            ValueError: Si el LLM retorna un plan malformado o vacío.
        """

    @abc.abstractmethod
    async def generate_synthesis(
        self,
        context: SynthesisContext,
    ) -> LLMResponse:
        """
        Genera una síntesis de los resultados de tools ejecutadas.

        Después de ejecutar una o más tools, el TaskExecutor llama a este
        método para que el LLM procese los resultados y genere la respuesta
        final para el usuario.

        A diferencia de generate_conversation, este método no incluye
        tool schemas (las tools ya se ejecutaron) y el foco está en la
        síntesis y presentación de resultados.

        Args:
            context: SynthesisContext con los resultados de tools.

        Returns:
            LLMResponse con la síntesis para el usuario.
            finish_reason siempre debe ser STOP (sin nuevos tool calls).

        Raises:
            Las mismas excepciones que generate_conversation.
        """

    @abc.abstractmethod
    async def generate_compaction_summary(
        self,
        context: CompactionContext,
    ) -> str:
        """
        Genera un resumen compactado de turns del historial.

        El MemoryManager llama a este método cuando el historial excede
        el umbral de compactación. El LLM genera un resumen conciso que
        captura los puntos esenciales de los turns a compactar.

        El resumen debe:
          - Ser en el idioma de la conversación
          - Preservar hechos importantes (nombres, decisiones, resultados)
          - Ser suficientemente conciso para caber en max_summary_tokens
          - NO incluir información de bajo nivel (IDs, timestamps, etc.)

        Args:
            context: CompactionContext con los turns a compactar.

        Returns:
            String con el resumen generado.
            Longitud garantizada <= max_summary_tokens tokens estimados.

        Raises:
            Las mismas excepciones que generate_conversation.
        """

    @abc.abstractmethod
    async def count_tokens(self, text: str) -> int:
        """
        Estima el número de tokens en un texto.

        El ContextBuilder lo usa para calcular el budget antes de enviar.
        La estimación puede ser heurística (rápida) o exacta (con tokenizador).

        Args:
            text: Texto a tokenizar.

        Returns:
            Número estimado de tokens.
        """

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """
        Verifica que el provider LLM esté disponible.

        Returns:
            True si el provider está operativo.
        """

    @abc.abstractmethod
    def get_routing_decision(
        self,
        llm_response: LLMResponse,
        *,
        mutation_tool_ids: frozenset[str],
        planning_threshold: int,
    ) -> RoutingDecision:
        """
        Interpreta una LLMResponse y retorna la RoutingDecision apropiada.

        El ConversationCoordinator llama a este método después de recibir
        la respuesta del LLM para determinar el flujo de ejecución.

        Este método encapsula el LLMResponseAnalyzer del dominio dentro
        del puerto para que el Coordinator no necesite instanciarlo directamente.

        Args:
            llm_response:       La respuesta del LLM a interpretar.
            mutation_tool_ids:  IDs de tools que producen mutación.
            planning_threshold: Número de tool calls que activa planificación.

        Returns:
            RoutingDecision con el flujo de ejecución recomendado.
        """

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Nombre del provider LLM activo ('openai', 'anthropic', 'local')."""

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Nombre del modelo LLM activo."""

    @property
    @abc.abstractmethod
    def max_context_tokens(self) -> int:
        """Tamaño máximo del context window del modelo activo."""


# =============================================================================
# SYSTEM PROMPT BUILDER
# =============================================================================


class SystemPromptBuilder:
    """
    Construye el system prompt de personalidad del agente HiperForge User.

    El system prompt es el componente más importante del contexto del LLM.
    Define quién es el agente, qué puede hacer, cómo debe comportarse, y
    qué restricciones tiene. Se construye una vez por sesión y se incluye
    en todas las invocaciones al LLM.

    Este builder es una clase utilitaria usada por el UserLLMAdapter.
    No es un puerto — es lógica de negocio del adaptador.
    """

    # Personalidad base del agente — estable, no cambia con el usuario
    _AGENT_IDENTITY: str = """Eres Forge, un asistente personal inteligente diseñado para ayudar a las personas con sus tareas cotidianas: buscar información, analizar documentos, organizar ideas y ejecutar acciones en su computadora.

Tu forma de ser:
- Directo y útil — vas al grano sin preámbulos innecesarios
- Honesto — si no sabes algo, lo dices; si necesitas más información, la pides
- Respetuoso — tratas al usuario como un adulto capaz
- Proactivo — anticipas lo que el usuario puede necesitar a continuación
- Natural — conversas como una persona real, no como un bot corporativo"""

    # Instrucciones de comportamiento con tools
    _TOOL_BEHAVIOR: str = """Capacidades operativas:
Cuando el usuario pide algo que requiere buscar información, analizar un archivo, o hacer algo en su computadora, usa las herramientas disponibles en lugar de inventar respuestas.

Reglas para el uso de herramientas:
- Usa herramientas cuando claramente aportan valor (información actualizada, archivos reales, acciones concretas)
- No uses herramientas para cosas que ya sabes con certeza
- Si necesitas usar varias herramientas, hazlo de forma eficiente
- Siempre explica al usuario qué estás haciendo y por qué"""

    # Restricciones de seguridad
    _SAFETY_CONSTRAINTS: str = """Restricciones de seguridad:
- Nunca ejecutes acciones irreversibles sin confirmación explícita del usuario
- No accedas a archivos sensibles (contraseñas, claves SSH, etc.) bajo ninguna circunstancia
- Si tienes dudas sobre si algo es seguro, no lo hagas y explica por qué
- Las acciones en el sistema del usuario requieren su aprobación cuando están marcadas como tal"""

    def build(
        self,
        profile_summary: ProfileSummary,
        *,
        available_tool_count: int = 0,
        include_tool_instructions: bool = True,
    ) -> str:
        """
        Construye el system prompt completo para una sesión.

        Combina la identidad del agente con las instrucciones de
        personalización del usuario y las instrucciones de tools.

        Args:
            profile_summary:          Resumen del perfil del usuario.
            available_tool_count:     Número de tools disponibles en este turno.
            include_tool_instructions: Si True, incluye las instrucciones de tools.

        Returns:
            System prompt completo listo para incluir en el LLMContext.
        """
        parts: list[str] = [self._AGENT_IDENTITY]

        # Instrucciones de personalización del usuario
        user_context = profile_summary.to_context_string()
        if user_context.strip():
            parts.append(user_context)

        # Instrucciones de tools (solo si hay tools disponibles)
        if include_tool_instructions and available_tool_count > 0:
            parts.append(self._TOOL_BEHAVIOR)

        # Restricciones de seguridad — siempre presentes
        parts.append(self._SAFETY_CONSTRAINTS)

        return "\n\n".join(parts)

    def build_planning_prompt(
        self,
        max_steps: int,
        available_tool_ids: list[str],
    ) -> str:
        """
        Construye el system prompt específico para el modo de planning.

        Instruye al LLM para que genere un plan estructurado en JSON.

        Args:
            max_steps:          Número máximo de pasos permitidos.
            available_tool_ids: IDs de las tools disponibles para el plan.

        Returns:
            System prompt para modo planning.
        """
        tools_str = ", ".join(available_tool_ids) if available_tool_ids else "ninguna"

        return f"""Eres un planificador de tareas. Tu objetivo es generar un plan estructurado para completar la tarea del usuario.

Reglas del plan:
- El plan debe tener COMO MÁXIMO {max_steps} pasos
- Cada paso debe usar exactamente UNA de las herramientas disponibles: {tools_str}
- Los pasos deben ser lo más paralelos posible (minimizar dependencias)
- Solo incluye pasos que son realmente necesarios

Responde ÚNICAMENTE con un JSON válido con este formato exacto:
{{
  "plan_rationale": "Explicación breve de por qué este plan es el más apropiado",
  "steps": [
    {{
      "description": "Descripción clara del paso para el usuario",
      "tool_id": "id_de_la_herramienta",
      "tool_arguments": {{"arg1": "valor1"}},
      "is_optional": false,
      "depends_on_steps": [],
      "rationale": "Por qué este paso es necesario"
    }}
  ],
  "estimated_duration_seconds": null
}}

No incluyas texto antes ni después del JSON."""

    def build_compaction_prompt(
        self,
        max_tokens: int,
        language: str,
    ) -> str:
        """
        Construye el system prompt para generación de resúmenes de compactación.

        Args:
            max_tokens: Longitud máxima del resumen en tokens.
            language:   Idioma del resumen.

        Returns:
            System prompt para modo compactación.
        """
        lang_name = {
            "es": "español",
            "en": "inglés",
            "pt": "portugués",
            "fr": "francés",
            "de": "alemán",
        }.get(language, language)

        return f"""Eres un asistente especializado en resumir conversaciones de forma precisa y concisa.

Tu tarea: resumir los turnos de conversación que te proporcione, preservando:
- Decisiones importantes tomadas
- Hechos clave mencionados (nombres, fechas, conceptos)
- Acciones ejecutadas y sus resultados principales
- Contexto esencial para entender la conversación futura

Restricciones:
- El resumen debe estar en {lang_name}
- Longitud máxima: aproximadamente {max_tokens} tokens
- No incluyas detalles de bajo nivel (IDs técnicos, timestamps exactos)
- Usa un tono neutro y factual
- Comienza directamente con el contenido, sin preámbulos

Formato: prosa continua, sin listas ni headers."""