"""
Protocolo abstracto LLM del ecosistema Forge Platform.

Este módulo define el contrato completo de integración con cualquier proveedor
de LLM. Es la pieza central de la arquitectura hexagonal en lo que respecta a
inteligencia: todos los adapters concretos (OpenAI, Anthropic, Local) implementan
este protocolo; ningún componente de la aplicación conoce al proveedor real.

Principios de diseño:
  1. El protocolo es el único conocimiento que la capa de aplicación tiene sobre
     el LLM. Si el contrato no cambia, se puede cambiar el proveedor sin tocar
     ningún otro archivo.
  2. Todos los tipos son inmutables (frozen Pydantic models). El estado del LLM
     no se muta — se crea una nueva instancia por cada interacción.
  3. El streaming es un ciudadano de primera clase, no un afterthought. El puerto
     expone tanto generate() síncrono (acumula respuesta completa) como
     generate_streaming() (emite chunks).
  4. El contexto (LLMContext) es un valor completo y autocontenido. El adapter
     recibe todo lo que necesita saber en un único objeto — sin estado implícito.
  5. Los tool calls son tipados y validados. El LLM no devuelve JSON libre;
     devuelve ToolCall objetos con schemas conocidos.
  6. El manejo de tokens es explícito. Cada LLMContext tiene un TokenBudget
     y cada LLMResponse reporta TokenUsage. El sistema nunca supera los límites
     sin saberlo.

Jerarquía de tipos:

  LLMMessage               — un mensaje individual en la conversación
  ToolDefinition           — schema de una tool para function calling
  ToolCall                 — invocación de tool solicitada por el LLM
  ToolResult               — resultado de una tool para devolver al LLM
  TokenBudget              — presupuesto de tokens para el context window
  LLMContext               — contexto completo para una invocación al LLM
  LLMResponse              — respuesta completa del LLM
  StreamChunk              — fragmento de respuesta en streaming
  FinishReason             — razón de terminación de la generación
  LLMPort (ABC)            — puerto abstracto que implementan los adapters
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# ENUMERACIONES
# =============================================================================


class MessageRole(str, Enum):
    """
    Rol del emisor de un mensaje en la conversación.

    Mapea directamente a los roles del estándar OpenAI Chat Completions,
    que Anthropic y la mayoría de providers modernos también adoptan.
    """

    SYSTEM = "system"
    """
    Mensaje de sistema. Define la personalidad, capacidades y restricciones
    del asistente. Solo debe haber uno por conversación y debe ir primero.
    """

    USER = "user"
    """
    Mensaje del usuario. Puede contener texto, imágenes o documentos adjuntos.
    """

    ASSISTANT = "assistant"
    """
    Mensaje del asistente (LLM). Puede contener texto y/o tool calls.
    """

    TOOL = "tool"
    """
    Resultado de una tool. Responde a un tool call previo del assistant.
    Se alterna con mensajes de assistant en flujos multi-turn con tools.
    """


class FinishReason(str, Enum):
    """
    Razón por la que el LLM detuvo la generación de tokens.

    Determina qué acción debe tomar el caller después de recibir la respuesta.
    """

    STOP = "stop"
    """
    El modelo generó una respuesta completa de forma natural.
    El contenido textual está listo para ser presentado al usuario.
    """

    TOOL_CALLS = "tool_calls"
    """
    El modelo solicitó ejecutar una o más tools.
    El caller debe ejecutar los tool calls y continuar la conversación
    con los resultados antes de presentar algo al usuario.
    """

    LENGTH = "length"
    """
    La generación se detuvo porque se alcanzó max_tokens.
    La respuesta puede estar incompleta — el caller debe manejarlo.
    """

    CONTENT_FILTER = "content_filter"
    """
    El proveedor bloqueó la respuesta por sus content policies.
    El caller debe notificar al usuario que la solicitud no puede procesarse.
    """

    ERROR = "error"
    """
    La generación falló por un error interno del proveedor.
    El caller debe manejar el error y posiblemente reintentar.
    """


class ContentType(str, Enum):
    """Tipo de contenido dentro de un mensaje multimodal."""

    TEXT = "text"
    """Contenido de texto plano o markdown."""

    IMAGE_URL = "image_url"
    """Imagen referenciada por URL (para providers que la soportan)."""

    IMAGE_BASE64 = "image_base64"
    """Imagen codificada en base64 (para adjuntos locales)."""

    DOCUMENT = "document"
    """Documento de texto (PDF extraído, DOCX, etc.) para análisis."""


# =============================================================================
# TIPOS DE CONTENIDO MULTIMODAL
# =============================================================================


class TextContent(BaseModel):
    """Bloque de contenido de texto dentro de un mensaje."""

    model_config = {"frozen": True}

    type: Literal[ContentType.TEXT] = ContentType.TEXT
    text: str = Field(
        min_length=0,
        description="Contenido textual del bloque.",
    )


class ImageContent(BaseModel):
    """
    Bloque de contenido de imagen dentro de un mensaje.

    Soporta tanto imágenes por URL como imágenes en base64. Los providers
    difieren en qué formato soportan; el adapter es responsable de la conversión.
    """

    model_config = {"frozen": True}

    type: Literal[ContentType.IMAGE_URL, ContentType.IMAGE_BASE64]
    url: str | None = Field(
        default=None,
        description="URL de la imagen (cuando type=IMAGE_URL).",
    )
    data: str | None = Field(
        default=None,
        description="Datos base64 de la imagen (cuando type=IMAGE_BASE64).",
    )
    media_type: str = Field(
        default="image/jpeg",
        pattern=r"^image/(jpeg|png|gif|webp)$",
        description="MIME type de la imagen.",
    )
    detail: Literal["low", "high", "auto"] = Field(
        default="auto",
        description=(
            "Nivel de detalle para análisis de imagen. 'auto' deja la decisión "
            "al provider. 'low' es más rápido y barato; 'high' más preciso."
        ),
    )

    @model_validator(mode="after")
    def validate_content_present(self) -> ImageContent:
        """Valida que haya URL o datos base64, no ambos ni ninguno."""
        if self.type == ContentType.IMAGE_URL and not self.url:
            raise ValueError("url es requerida cuando type=IMAGE_URL")
        if self.type == ContentType.IMAGE_BASE64 and not self.data:
            raise ValueError("data es requerida cuando type=IMAGE_BASE64")
        return self


class DocumentContent(BaseModel):
    """
    Bloque de contenido de documento (texto extraído) dentro de un mensaje.

    Se usa para pasar el contenido de un PDF, DOCX u otro documento al LLM
    para análisis. El texto ya está extraído — no se pasa el binario.
    """

    model_config = {"frozen": True}

    type: Literal[ContentType.DOCUMENT] = ContentType.DOCUMENT
    title: str = Field(
        description="Nombre o título del documento para referencia en la respuesta.",
    )
    text: str = Field(
        description="Texto extraído del documento.",
    )
    source_filename: str | None = Field(
        default=None,
        description="Nombre del archivo original para referencia.",
    )
    page_count: int | None = Field(
        default=None,
        ge=1,
        description="Número de páginas del documento original.",
    )


# Tipo union para el contenido de un mensaje
MessageContent = TextContent | ImageContent | DocumentContent


# =============================================================================
# MENSAJES
# =============================================================================


class LLMMessage(BaseModel):
    """
    Un mensaje individual en la conversación con el LLM.

    Soporta el protocolo multi-turn estándar con roles system/user/assistant/tool.
    El contenido puede ser simple texto o una lista de bloques multimodales.

    Diseño:
    - Si content es un string, representa un mensaje de texto simple.
    - Si content es una lista, representa un mensaje multimodal con múltiples bloques.
    - Los tool_calls solo están presentes en mensajes de role=ASSISTANT.
    - El tool_call_id solo está presente en mensajes de role=TOOL.
    """

    model_config = {"frozen": True}

    role: MessageRole = Field(
        description="Rol del emisor del mensaje.",
    )
    content: str | list[MessageContent] = Field(
        description=(
            "Contenido del mensaje. String para texto simple, "
            "lista de bloques para contenido multimodal."
        ),
    )
    tool_calls: list[ToolCall] | None = Field(
        default=None,
        description=(
            "Tool calls solicitados por el LLM. Solo presente en mensajes "
            "de role=ASSISTANT cuando el modelo decide usar tools."
        ),
    )
    tool_call_id: str | None = Field(
        default=None,
        description=(
            "ID del tool call al que responde este mensaje. "
            "Solo presente en mensajes de role=TOOL."
        ),
    )
    name: str | None = Field(
        default=None,
        description=(
            "Nombre opcional del participante. En mensajes TOOL, "
            "identifica qué tool produjo el resultado."
        ),
    )

    @classmethod
    def system(cls, text: str) -> LLMMessage:
        """Factory: crea un mensaje de sistema con texto simple."""
        return cls(role=MessageRole.SYSTEM, content=text)

    @classmethod
    def user(cls, content: str | list[MessageContent]) -> LLMMessage:
        """Factory: crea un mensaje de usuario con texto o contenido multimodal."""
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant(
        cls,
        text: str,
        *,
        tool_calls: list[ToolCall] | None = None,
    ) -> LLMMessage:
        """Factory: crea un mensaje de asistente con texto opcional y tool calls."""
        return cls(role=MessageRole.ASSISTANT, content=text, tool_calls=tool_calls)

    @classmethod
    def tool_result(
        cls,
        tool_call_id: str,
        content: str,
        *,
        tool_name: str | None = None,
    ) -> LLMMessage:
        """
        Factory: crea un mensaje con el resultado de una tool.

        Args:
            tool_call_id: ID del ToolCall al que responde este resultado.
            content:      Resultado de la tool serializado como string.
            tool_name:    Nombre de la tool para referencia.
        """
        return cls(
            role=MessageRole.TOOL,
            content=content,
            tool_call_id=tool_call_id,
            name=tool_name,
        )

    def get_text(self) -> str:
        """
        Extrae el texto del mensaje independientemente del formato del contenido.

        Returns:
            El texto del mensaje. Para contenido multimodal, concatena solo
            los bloques de tipo TEXT separados por espacio.
        """
        if isinstance(self.content, str):
            return self.content

        parts: list[str] = []
        for block in self.content:
            if isinstance(block, TextContent):
                parts.append(block.text)
            elif isinstance(block, DocumentContent):
                parts.append(f"[Documento: {block.title}]\n{block.text}")
        return " ".join(parts)

    def estimated_tokens(self) -> int:
        """
        Estima el número de tokens de este mensaje para presupuesto de contexto.

        La estimación usa la heurística de 4 caracteres por token, que es
        razonablemente precisa para texto en inglés y español. Para una
        estimación exacta, usar el tokenizer del provider (más costoso).

        Returns:
            Estimación del número de tokens del mensaje.
        """
        text = self.get_text()
        # Overhead por estructura del mensaje (~10 tokens por mensaje)
        overhead = 10
        return len(text) // 4 + overhead


# =============================================================================
# TOOL DEFINITIONS (para function calling)
# =============================================================================


class ToolParameterSchema(BaseModel):
    """
    Schema JSON de los parámetros de una tool para function calling.

    Sigue el estándar JSON Schema (draft 7) que usan OpenAI, Anthropic y
    la mayoría de providers modernos para definir los parámetros de tools.
    """

    model_config = {"frozen": True}

    type: str = Field(default="object", description="Tipo JSON Schema del parámetro.")
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Propiedades del objeto de parámetros con sus schemas individuales.",
    )
    required: list[str] = Field(
        default_factory=list,
        description="Lista de nombres de propiedades requeridas.",
    )
    additional_properties: bool = Field(
        default=False,
        description=(
            "Si se permiten propiedades adicionales no definidas en el schema. "
            "False por defecto para validación estricta."
        ),
    )
    description: str | None = Field(
        default=None,
        description="Descripción del schema completo (opcional).",
    )


class ToolDefinition(BaseModel):
    """
    Definición de una tool para function calling del LLM.

    Este es el objeto que se envía al LLM como parte del contexto para que
    sepa qué tools puede invocar. El adapter es responsable de transformar
    este formato al formato específico del provider (OpenAI functions,
    Anthropic tools, etc.).
    """

    model_config = {"frozen": True}

    name: str = Field(
        pattern=r"^[a-zA-Z0-9_-]{1,64}$",
        description=(
            "Nombre único de la tool en formato snake_case o kebab-case. "
            "Debe ser reconocible y descriptivo para el LLM."
        ),
    )
    description: str = Field(
        min_length=10,
        max_length=1024,
        description=(
            "Descripción detallada de qué hace la tool, cuándo usarla, "
            "y qué restricciones tiene. Es crítica para que el LLM "
            "tome buenas decisiones de uso."
        ),
    )
    parameters: ToolParameterSchema = Field(
        description="Schema JSON de los parámetros de entrada de la tool.",
    )
    strict: bool = Field(
        default=True,
        description=(
            "Si True, el LLM debe seguir el schema estrictamente. "
            "Algunos providers (OpenAI) soportan esto nativamente."
        ),
    )


# =============================================================================
# TOOL CALLS Y RESULTADOS
# =============================================================================


class ToolCall(BaseModel):
    """
    Invocación de una tool solicitada por el LLM en su respuesta.

    Cuando el LLM decide usar una tool, incluye uno o más ToolCall en su
    respuesta. El sistema debe ejecutar cada tool call y devolver los
    resultados en el siguiente turno de la conversación.
    """

    model_config = {"frozen": True}

    id: str = Field(
        description=(
            "ID único de este tool call, generado por el LLM. "
            "Se usa para correlacionar el tool call con su resultado (ToolResult)."
        ),
    )
    name: str = Field(
        description="Nombre de la tool a invocar (debe coincidir con ToolDefinition.name).",
    )
    arguments: dict[str, Any] = Field(
        description=(
            "Argumentos para la tool como diccionario. El LLM genera estos "
            "argumentos basándose en el schema de ToolDefinition.parameters."
        ),
    )

    def get_argument(self, key: str, default: Any = None) -> Any:
        """
        Obtiene un argumento específico del tool call con valor por defecto.

        Args:
            key:     Nombre del argumento.
            default: Valor por defecto si el argumento no está presente.

        Returns:
            El valor del argumento o el default.
        """
        return self.arguments.get(key, default)


class ToolResult(BaseModel):
    """
    Resultado de la ejecución de un tool call para devolver al LLM.

    Después de ejecutar una tool, el sistema crea un ToolResult y lo añade
    a la conversación como un mensaje de role=TOOL para que el LLM pueda
    usar el resultado en su siguiente respuesta.
    """

    model_config = {"frozen": True}

    tool_call_id: str = Field(
        description="ID del ToolCall al que responde este resultado.",
    )
    tool_name: str = Field(
        description="Nombre de la tool que produjo este resultado.",
    )
    content: str = Field(
        description=(
            "Resultado de la tool serializado como string. "
            "Para resultados complejos, se serializa a JSON."
        ),
    )
    is_error: bool = Field(
        default=False,
        description=(
            "True si el resultado representa un error de ejecución. "
            "Algunos providers cambian el comportamiento del LLM ante errores."
        ),
    )

    @classmethod
    def success(
        cls,
        tool_call_id: str,
        tool_name: str,
        content: str,
    ) -> ToolResult:
        """Factory: crea un resultado de tool exitoso."""
        return cls(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            content=content,
            is_error=False,
        )

    @classmethod
    def error(
        cls,
        tool_call_id: str,
        tool_name: str,
        error_message: str,
    ) -> ToolResult:
        """Factory: crea un resultado de tool que representa un error."""
        return cls(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            content=f"Error: {error_message}",
            is_error=True,
        )

    def to_llm_message(self) -> LLMMessage:
        """
        Convierte el ToolResult en un LLMMessage de role=TOOL.

        Este mensaje se añade al historial de conversación para que el LLM
        pueda ver el resultado de la tool en el siguiente turno.

        Returns:
            LLMMessage listo para añadir al contexto de conversación.
        """
        return LLMMessage.tool_result(
            tool_call_id=self.tool_call_id,
            content=self.content,
            tool_name=self.tool_name,
        )


# =============================================================================
# TOKEN BUDGET
# =============================================================================


class TokenBudget(BaseModel):
    """
    Presupuesto de tokens para una invocación al LLM.

    Gestiona la distribución del context window entre los diferentes
    componentes del contexto (system prompt, tools, historial, respuesta).
    Garantiza que el contexto nunca exceda los límites del modelo.
    """

    model_config = {"frozen": True}

    total_context_window: int = Field(
        ge=1_000,
        description="Tamaño total del context window del modelo en tokens.",
    )
    reserved_for_response: int = Field(
        ge=256,
        description="Tokens reservados para la respuesta del LLM.",
    )
    reserved_for_system: int = Field(
        ge=100,
        description="Tokens reservados para el system prompt.",
    )
    reserved_for_tools: int = Field(
        ge=0,
        description="Tokens reservados para los schemas de tools.",
    )
    reserved_for_user_profile: int = Field(
        default=300,
        ge=0,
        description="Tokens reservados para el resumen del perfil de usuario.",
    )

    @model_validator(mode="after")
    def validate_budget_feasibility(self) -> TokenBudget:
        """
        Valida que el presupuesto total sea alcanzable con las reservas configuradas.

        El budget para el historial de conversación es lo que queda después de
        todas las reservas. Si las reservas superan el context window, la
        configuración es inviable.
        """
        total_reserved = (
            self.reserved_for_response
            + self.reserved_for_system
            + self.reserved_for_tools
            + self.reserved_for_user_profile
        )
        if total_reserved >= self.total_context_window:
            raise ValueError(
                f"Las reservas de tokens ({total_reserved}) superan o igualan "
                f"el context window total ({self.total_context_window}). "
                f"No quedan tokens para el historial de conversación."
            )
        return self

    @property
    def available_for_history(self) -> int:
        """
        Tokens disponibles para el historial de conversación.

        Es el espacio restante después de todas las reservas fijas.
        El ContextBuilder usa este valor para determinar cuántos turns
        del historial pueden incluirse sin necesidad de compactación.
        """
        return (
            self.total_context_window
            - self.reserved_for_response
            - self.reserved_for_system
            - self.reserved_for_tools
            - self.reserved_for_user_profile
        )

    def can_fit_tokens(self, token_count: int) -> bool:
        """
        Indica si una cantidad de tokens cabe en el budget disponible para historial.

        Args:
            token_count: Número de tokens a verificar.

        Returns:
            True si los tokens caben en el budget disponible.
        """
        return token_count <= self.available_for_history

    def usage_ratio(self, current_history_tokens: int) -> float:
        """
        Calcula el ratio de uso del budget de historial.

        Args:
            current_history_tokens: Tokens actualmente usados por el historial.

        Returns:
            Ratio de uso entre 0.0 (vacío) y 1.0+ (excedido).
        """
        if self.available_for_history == 0:
            return 1.0
        return current_history_tokens / self.available_for_history


# =============================================================================
# CONTEXTO DE INVOCACIÓN
# =============================================================================


class LLMContext(BaseModel):
    """
    Contexto completo para una invocación al LLM.

    Este es el objeto central del protocolo LLM. Contiene todo lo que el
    adapter necesita para construir y enviar una request al provider:
    mensajes, tools disponibles, parámetros de generación y presupuesto
    de tokens. El adapter no necesita conocer nada más.

    El LLMContext es inmutable — se crea una nueva instancia por cada
    invocación. No hay estado compartido entre invocaciones.
    """

    model_config = {"frozen": True}

    # --- Conversación ---
    messages: list[LLMMessage] = Field(
        min_length=1,
        description=(
            "Lista de mensajes de la conversación en orden cronológico. "
            "Debe incluir al menos un mensaje (el del usuario actual). "
            "El primer mensaje debe ser de role=SYSTEM si se usa system prompt."
        ),
    )

    # --- Tools disponibles ---
    tools: list[ToolDefinition] = Field(
        default_factory=list,
        description=(
            "Definiciones de las tools disponibles para esta invocación. "
            "Lista vacía deshabilita el function calling."
        ),
    )
    tool_choice: Literal["auto", "none", "required"] = Field(
        default="auto",
        description=(
            "Política de selección de tools: "
            "'auto' = el LLM decide si usa tools o no (comportamiento por defecto); "
            "'none' = deshabilita tools aunque estén definidas; "
            "'required' = el LLM DEBE usar al menos una tool."
        ),
    )

    # --- Parámetros de generación ---
    max_tokens: int = Field(
        ge=1,
        le=32_768,
        description="Número máximo de tokens a generar en la respuesta.",
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Temperatura de muestreo para la generación.",
    )

    # --- Presupuesto de tokens ---
    token_budget: TokenBudget = Field(
        description=(
            "Presupuesto de tokens del context window. Usado por el adapter "
            "para verificar que el contexto no exceda los límites del modelo."
        ),
    )

    # --- Metadata de trazabilidad ---
    request_id: str | None = Field(
        default=None,
        description=(
            "ID único de esta request para correlación en logs y tracing. "
            "Se propaga al provider si lo soporta (OpenAI: X-Request-ID header)."
        ),
    )
    session_id: str | None = Field(
        default=None,
        description="ID de la sesión Forge asociada a esta invocación.",
    )
    task_id: str | None = Field(
        default=None,
        description="ID de la tarea Forge asociada a esta invocación.",
    )

    @field_validator("messages")
    @classmethod
    def validate_message_order(cls, messages: list[LLMMessage]) -> list[LLMMessage]:
        """
        Valida que el orden de los mensajes sea coherente con el protocolo.

        Reglas:
          - Si hay un mensaje SYSTEM, debe ser el primero.
          - No puede haber dos mensajes SYSTEM consecutivos.
          - Un mensaje TOOL debe estar precedido por un mensaje ASSISTANT
            con tool_calls.
        """
        if not messages:
            return messages

        # El system prompt debe ser el primer mensaje
        for i, msg in enumerate(messages):
            if msg.role == MessageRole.SYSTEM and i > 0:
                raise ValueError(
                    f"El mensaje SYSTEM debe ser el primero. "
                    f"Se encontró SYSTEM en posición {i}."
                )

        return messages

    def estimated_input_tokens(self) -> int:
        """
        Estima el total de tokens de input de este contexto.

        Suma las estimaciones individuales de todos los mensajes más el
        overhead de las definiciones de tools.

        Returns:
            Estimación total de tokens de input.
        """
        message_tokens = sum(msg.estimated_tokens() for msg in self.messages)

        # Estimación de tokens para tool definitions (schema JSON)
        tool_tokens = sum(
            len(tool.model_dump_json()) // 4 + 20
            for tool in self.tools
        )

        return message_tokens + tool_tokens

    def get_system_message(self) -> LLMMessage | None:
        """
        Retorna el mensaje de sistema del contexto, si existe.

        Returns:
            El LLMMessage de role=SYSTEM, o None si no hay system prompt.
        """
        for msg in self.messages:
            if msg.role == MessageRole.SYSTEM:
                return msg
        return None

    def get_conversation_messages(self) -> list[LLMMessage]:
        """
        Retorna solo los mensajes de conversación (excluyendo el system prompt).

        Returns:
            Lista de mensajes con roles USER, ASSISTANT y TOOL.
        """
        return [m for m in self.messages if m.role != MessageRole.SYSTEM]

    def with_tool_results(self, results: list[ToolResult]) -> LLMContext:
        """
        Crea un nuevo LLMContext añadiendo los resultados de tools al historial.

        Este es el mecanismo para continuar una conversación después de ejecutar
        tool calls. El adapter llama a este método después de que el sistema
        ejecuta los tools y antes de la siguiente invocación al LLM.

        Args:
            results: Lista de resultados de tools a añadir al contexto.

        Returns:
            Nuevo LLMContext con los resultados de tools añadidos.
        """
        new_messages = list(self.messages) + [r.to_llm_message() for r in results]
        return self.model_copy(update={"messages": new_messages})


# =============================================================================
# RESPUESTA DEL LLM
# =============================================================================


class TokenUsage(BaseModel):
    """
    Uso de tokens reportado por el LLM provider para una invocación.

    Esencial para tracking de costos, presupuesto de tokens y optimización.
    """

    model_config = {"frozen": True}

    prompt_tokens: int = Field(
        ge=0,
        description="Tokens consumidos por el prompt (input) incluyendo historial y tools.",
    )
    completion_tokens: int = Field(
        ge=0,
        description="Tokens generados en la respuesta (output).",
    )
    total_tokens: int = Field(
        ge=0,
        description="Total de tokens (prompt + completion).",
    )
    cached_tokens: int = Field(
        default=0,
        ge=0,
        description=(
            "Tokens del prompt que vinieron del cache del provider "
            "(Anthropic prompt caching, OpenAI prompt caching). "
            "Reducen el costo pero se contabilizan en prompt_tokens."
        ),
    )

    @model_validator(mode="after")
    def validate_total(self) -> TokenUsage:
        """Valida que el total sea coherente con prompt + completion."""
        expected_total = self.prompt_tokens + self.completion_tokens
        # Tolerancia de ±1 para diferencias de redondeo entre providers
        if abs(self.total_tokens - expected_total) > 1:
            # No lanzar error — algunos providers calculan el total diferente
            # Solo corregir silenciosamente para mantener la consistencia interna
            object.__setattr__(self, "total_tokens", expected_total)
        return self

    @property
    def estimated_cost_usd(self) -> float:
        """
        Estimación del costo en USD basada en precios promedio de mercado.

        NOTA: Esta estimación es aproximada y no refleja precios exactos de
        ningún provider. Se usa solo para métricas de monitoring interno.
        Los precios reales dependen del provider, modelo y volumen.

        Estimación basada en: $3.00/1M input tokens, $15.00/1M output tokens
        (aproximadamente GPT-4o y Claude Sonnet como referencia).
        """
        input_cost = (self.prompt_tokens - self.cached_tokens) * 3.0 / 1_000_000
        cached_cost = self.cached_tokens * 0.3 / 1_000_000  # cached discount
        output_cost = self.completion_tokens * 15.0 / 1_000_000
        return input_cost + cached_cost + output_cost


class LLMResponse(BaseModel):
    """
    Respuesta completa del LLM a una invocación.

    Encapsula todo lo que el LLM retorna: el contenido textual (si lo hay),
    los tool calls solicitados (si los hay), la razón de finalización y las
    métricas de uso de tokens.

    El caller examina finish_reason para determinar el siguiente paso:
    - STOP: la respuesta es completa, presentar al usuario.
    - TOOL_CALLS: ejecutar los tool_calls y continuar la conversación.
    - LENGTH: la respuesta está truncada, manejar apropiadamente.
    - CONTENT_FILTER: rechazado por el provider, notificar al usuario.
    """

    model_config = {"frozen": True}

    content: str = Field(
        default="",
        description=(
            "Contenido textual de la respuesta. Puede ser vacío cuando "
            "finish_reason=TOOL_CALLS (el LLM solo solicitó tools, sin texto)."
        ),
    )
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description=(
            "Tool calls solicitados por el LLM. Lista vacía cuando "
            "finish_reason=STOP (respuesta puramente textual)."
        ),
    )
    finish_reason: FinishReason = Field(
        description="Razón por la que el LLM detuvo la generación.",
    )
    usage: TokenUsage = Field(
        description="Uso de tokens de esta invocación.",
    )
    model: str = Field(
        description="Nombre exacto del modelo que generó esta respuesta.",
    )
    provider: str = Field(
        description="Identificador del provider (openai, anthropic, local).",
    )
    request_id: str | None = Field(
        default=None,
        description="ID de la request devuelto por el provider para trazabilidad.",
    )

    @property
    def has_tool_calls(self) -> bool:
        """Indica si la respuesta contiene tool calls a ejecutar."""
        return len(self.tool_calls) > 0

    @property
    def has_content(self) -> bool:
        """Indica si la respuesta contiene texto (puede coexistir con tool calls)."""
        return bool(self.content.strip())

    @property
    def is_complete(self) -> bool:
        """
        Indica si la respuesta está completa (no truncada por límite de tokens).
        """
        return self.finish_reason in {
            FinishReason.STOP,
            FinishReason.TOOL_CALLS,
        }

    def to_assistant_message(self) -> LLMMessage:
        """
        Convierte la respuesta en un LLMMessage de role=ASSISTANT.

        Se usa para añadir la respuesta del LLM al historial de conversación
        antes de continuar con los tool calls o presentar al usuario.

        Returns:
            LLMMessage listo para añadir al historial.
        """
        return LLMMessage.assistant(
            text=self.content,
            tool_calls=self.tool_calls if self.tool_calls else None,
        )


# =============================================================================
# STREAMING
# =============================================================================


class StreamChunk(BaseModel):
    """
    Fragmento de respuesta en modo streaming.

    En streaming, el LLM emite tokens progresivamente. Cada StreamChunk
    contiene un fragmento del texto de respuesta o metadatos sobre tool calls
    que se van construyendo incrementalmente.

    El caller acumula los chunks hasta recibir is_final=True, momento en el
    que la respuesta completa está disponible.
    """

    model_config = {"frozen": True}

    delta_content: str = Field(
        default="",
        description=(
            "Fragmento incremental de texto de la respuesta. "
            "Se acumula concatenando todos los deltas hasta is_final=True."
        ),
    )
    delta_tool_call: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Fragmento incremental de un tool call en construcción. "
            "El formato depende del provider — el adapter lo normaliza."
        ),
    )
    is_final: bool = Field(
        default=False,
        description=(
            "True en el último chunk de la stream. Cuando is_final=True, "
            "el campo final_response contiene la respuesta completa acumulada."
        ),
    )
    final_response: LLMResponse | None = Field(
        default=None,
        description=(
            "Respuesta completa acumulada. Solo presente cuando is_final=True. "
            "Contiene usage, finish_reason y la respuesta completa."
        ),
    )
    finish_reason: FinishReason | None = Field(
        default=None,
        description="Razón de finalización. Solo presente en el chunk final.",
    )


# =============================================================================
# PUERTO ABSTRACTO LLM
# =============================================================================


class LLMPort(abc.ABC):
    """
    Puerto abstracto que define el contrato de integración con un LLM provider.

    Todos los adapters concretos (OpenAIAdapter, AnthropicAdapter, LocalModelAdapter)
    deben implementar este protocolo. La capa de aplicación solo conoce este
    puerto — nunca las implementaciones concretas.

    Contratos invariantes que todos los adapters DEBEN respetar:
      1. generate() SIEMPRE retorna un LLMResponse, nunca lanza excepciones de
         infraestructura al caller — las convierte en subclases de ForgeLLMError.
      2. El retry y backoff es responsabilidad del adapter, no del caller.
      3. El token budget del LLMContext se respeta — si el contexto excede el
         límite, el adapter debe lanzar LLMTokenLimitError, no silenciarlo.
      4. El streaming es non-blocking — generate_streaming() retorna un
         AsyncIterator que el caller puede consumir sin bloquear el event loop.
      5. count_tokens() es una estimación eficiente. Para tokenización exacta
         (si el provider lo requiere), el adapter puede hacer una llamada
         adicional pero debe cachear el resultado.
    """

    @abc.abstractmethod
    async def generate(self, context: LLMContext) -> LLMResponse:
        """
        Genera una respuesta completa del LLM para el contexto dado.

        Envía el contexto al provider, espera la respuesta completa y la
        retorna como LLMResponse. Para conversaciones donde la UX no requiere
        streaming (batch processing, planning, etc.) este método es preferible.

        Args:
            context: Contexto completo de la invocación, incluyendo mensajes,
                     tools disponibles y parámetros de generación.

        Returns:
            Respuesta completa del LLM con contenido, tool calls y métricas.

        Raises:
            LLMConnectionError:    Si no se puede conectar al provider.
            LLMRateLimitError:     Si el provider retorna rate limit (429).
            LLMTokenLimitError:    Si el contexto excede el context window.
            LLMContentFilterError: Si el provider bloquea el contenido.
            LLMTimeoutError:       Si la request excede el timeout configurado.
            LLMProviderError:      Para otros errores del provider (5xx, etc.).
        """

    @abc.abstractmethod
    async def generate_streaming(
        self, context: LLMContext
    ) -> AsyncIterator[StreamChunk]:
        """
        Genera una respuesta del LLM en modo streaming.

        Emite StreamChunk progresivamente a medida que el LLM genera tokens.
        El último chunk tiene is_final=True y contiene la LLMResponse completa
        acumulada en final_response.

        Este método es preferible para conversaciones interactivas donde
        se quiere mostrar la respuesta al usuario mientras se genera.

        Args:
            context: Contexto completo de la invocación.

        Yields:
            StreamChunk con fragmentos progresivos de la respuesta.
            El último chunk tiene is_final=True.

        Raises:
            Las mismas excepciones que generate().
        """

    @abc.abstractmethod
    async def count_tokens(self, text: str) -> int:
        """
        Cuenta o estima el número de tokens en un texto dado.

        Usado por el ContextBuilder para calcular el presupuesto de tokens
        antes de enviar el contexto al LLM. La implementación puede ser:
        - Tokenización exacta usando el tokenizer del provider (preciso pero costoso).
        - Heurística de caracteres/4 (rápido pero menos preciso).
        - Llamada a la API del provider si soporta tokenización.

        Args:
            text: Texto a tokenizar.

        Returns:
            Número estimado de tokens.

        Raises:
            LLMConnectionError: Si la tokenización requiere una llamada de red
                                y no hay conexión disponible.
        """

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """
        Verifica que el provider esté disponible y operativo.

        Se llama durante el startup y periódicamente para detectar outages.
        Debe ser una verificación ligera (ping, modelo list, etc.) — no
        debe generar tokens ni consumir tokens de billing.

        Returns:
            True si el provider está disponible y respondiendo.
            False si hay problemas de conectividad o el provider está degradado.
        """

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """
        Nombre del provider implementado por este adapter.

        Returns:
            Identificador del provider en minúsculas ('openai', 'anthropic', 'local').
        """

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """
        Nombre del modelo actualmente configurado en este adapter.

        Returns:
            Nombre del modelo tal como lo identifica el provider
            ('gpt-4o', 'claude-sonnet-4-20250514', 'llama3.2').
        """

    @property
    @abc.abstractmethod
    def max_context_tokens(self) -> int:
        """
        Tamaño máximo del context window del modelo configurado, en tokens.

        Returns:
            Número máximo de tokens (input + output) que soporta el modelo.
        """


# =============================================================================
# TIPOS DE CONVENIENCIA PARA TYPE HINTS
# =============================================================================

# AsyncIterator tipado para streaming — para uso en type hints de la capa de aplicación
LLMStream = AsyncIterator[StreamChunk]