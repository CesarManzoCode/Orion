"""
Puertos de entrada de HiperForge User.

Este módulo define los contratos que la capa de interfaz (UI, CLI)
usa para interactuar con el core del agente. Son los únicos puntos
de entrada al sistema — ninguna otra capa de la UI puede acceder
directamente a servicios internos.

Principio de aislamiento:
  La UI solo conoce estos puertos. Nunca importa:
  - ConversationCoordinator directamente
  - SessionManager, TaskExecutor, PolicyEngine
  - Ninguna entidad de dominio salvo los tipos de respuesta

Esto garantiza que la UI sea reemplazable (Tauri → Web UI → CLI)
sin tocar nada del core. El puente entre UI y core es exactamente
este módulo.

Puertos definidos:

  ConversationPort — operaciones de conversación y sesión para la UI
  AdminPort        — operaciones de diagnóstico para el debug CLI

Tipos de entrada/salida:

  UserMessageRequest      — mensaje del usuario con adjuntos opcionales
  AttachedFile            — archivo adjunto procesado
  AssistantResponse       — respuesta completa del agente para la UI
  StreamingChunk          — fragmento de respuesta en streaming para UI
  ApprovalResponseRequest — respuesta del usuario a una solicitud de aprobación
  SessionListResponse     — listado de sesiones para el selector de sesión
  SessionInfo             — info resumida de una sesión para la UI
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, AsyncIterator

from src.domain.entities.artifact import ArtifactSummary, ArtifactType
from src.domain.entities.session import SessionStatus, SessionSummary
from src.domain.value_objects.identifiers import (
    ArtifactId,
    ApprovalId,
    SessionId,
    UserId,
)


# =============================================================================
# TIPOS DE ARCHIVO ADJUNTO
# =============================================================================


class AttachedFileType(Enum):
    """Tipo de archivo adjunto enviado por el usuario."""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "markdown"
    IMAGE = "image"
    CSV = "csv"
    JSON = "json"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class AttachedFile:
    """
    Archivo adjunto del usuario preprocesado por la capa de interfaz.

    La UI extrae el contenido del archivo antes de enviarlo al ConversationPort.
    El core recibe texto ya extraído, no el binario del archivo.

    En el caso de imágenes, la UI puede enviarlas como base64 para que
    el LLM las procese directamente (multimodal).
    """

    filename: str
    """Nombre original del archivo."""

    file_type: AttachedFileType
    """Tipo de archivo detectado."""

    extracted_text: str | None
    """
    Texto extraído del archivo (para PDF, DOCX, TXT, MD, CSV, JSON).
    None si el archivo es una imagen o si la extracción falló.
    """

    image_base64: str | None
    """
    Datos de la imagen en base64 (solo para imágenes).
    None para archivos de texto.
    """

    image_media_type: str | None
    """
    MIME type de la imagen ('image/jpeg', 'image/png', etc.).
    None para archivos de texto.
    """

    size_bytes: int = 0
    """Tamaño del archivo original en bytes."""

    extraction_error: str | None = None
    """
    Mensaje de error si la extracción de texto falló.
    La UI puede mostrárselo al usuario.
    """

    @property
    def is_image(self) -> bool:
        """True si el adjunto es una imagen."""
        return self.file_type == AttachedFileType.IMAGE

    @property
    def has_content(self) -> bool:
        """True si el adjunto tiene contenido utilizable."""
        return bool(self.extracted_text) or bool(self.image_base64)

    def get_context_representation(self) -> str:
        """
        Retorna una representación del adjunto para incluir en el contexto del LLM.

        Para archivos de texto, retorna el contenido extraído con encabezado.
        Para imágenes, retorna un placeholder descriptivo.

        Returns:
            String representando el adjunto para el context del LLM.
        """
        if self.is_image:
            return f"[Imagen adjunta: {self.filename}]"

        if self.extracted_text:
            header = f"--- Archivo adjunto: {self.filename} ---\n"
            return header + self.extracted_text

        if self.extraction_error:
            return f"[Archivo adjunto: {self.filename} — Error al leer: {self.extraction_error}]"

        return f"[Archivo adjunto: {self.filename} — Sin contenido]"


# =============================================================================
# TIPOS DE REQUEST
# =============================================================================


@dataclass(frozen=True)
class UserMessageRequest:
    """
    Mensaje del usuario enviado al ConversationPort.

    Encapsula el mensaje de texto, los archivos adjuntos, y el contexto
    de sesión necesario para procesar el turno conversacional.
    """

    session_id: SessionId
    """ID de la sesión activa."""

    message: str
    """
    Texto del mensaje del usuario.
    Puede estar vacío si el usuario solo envió archivos adjuntos.
    """

    attachments: list[AttachedFile] = field(default_factory=list)
    """
    Archivos adjuntos del usuario, ya preprocesados por la UI.
    La UI extrae el texto y los envía aquí — el core no hace I/O de archivos.
    """

    request_id: str | None = None
    """
    ID único de la request generado por la UI para correlación.
    Si no se provee, el ConversationPort genera uno.
    """

    client_timestamp: datetime | None = None
    """
    Timestamp del cliente cuando se envió el mensaje.
    Útil para detectar diferencias de reloj significativas.
    """

    @property
    def has_attachments(self) -> bool:
        """True si el request incluye archivos adjuntos."""
        return len(self.attachments) > 0

    @property
    def is_empty(self) -> bool:
        """True si no hay mensaje de texto ni adjuntos con contenido."""
        return not self.message.strip() and not any(
            a.has_content for a in self.attachments
        )

    def get_full_text_for_context(self) -> str:
        """
        Construye el texto completo del mensaje incluyendo adjuntos.

        Para el LLM, el mensaje del usuario y los adjuntos se presentan
        como un bloque de texto unificado.

        Returns:
            String con el mensaje y el contenido de adjuntos.
        """
        parts: list[str] = []

        if self.message.strip():
            parts.append(self.message.strip())

        for attachment in self.attachments:
            if attachment.has_content and not attachment.is_image:
                parts.append(attachment.get_context_representation())

        return "\n\n".join(parts)


@dataclass(frozen=True)
class ApprovalResponseRequest:
    """
    Respuesta del usuario a una solicitud de aprobación.

    La UI envía este request cuando el usuario hace clic en
    "Aprobar" o "Denegar" en el diálogo de aprobación.
    """

    session_id: SessionId
    """ID de la sesión activa."""

    approval_id: ApprovalId
    """ID de la solicitud de aprobación que se está respondiendo."""

    granted: bool
    """True si el usuario aprobó, False si denegó."""

    remember_decision: bool = False
    """
    True si el usuario eligió recordar la decisión para el futuro.
    El sistema registrará el permiso en el UserProfile.
    """


@dataclass(frozen=True)
class CancelTaskRequest:
    """Solicitud de cancelación de la tarea activa."""

    session_id: SessionId
    """ID de la sesión cuya tarea activa se quiere cancelar."""

    reason: str = "user_requested"
    """Razón de la cancelación para el audit log."""


@dataclass(frozen=True)
class CreateSessionRequest:
    """Solicitud de creación de una nueva sesión."""

    user_id: UserId
    """ID del usuario que crea la sesión."""

    restore_context: bool = False
    """
    Si True, el sistema intenta recuperar contexto relevante de la última
    sesión cerrada para la nueva sesión (resumen, preferencias activas, etc.).
    """


@dataclass(frozen=True)
class SwitchSessionRequest:
    """Solicitud de cambio a una sesión existente."""

    session_id: SessionId
    """ID de la sesión a la que se quiere cambiar."""


# =============================================================================
# TIPOS DE RESPUESTA
# =============================================================================


class AssistantResponseType(Enum):
    """
    Tipo de respuesta del agente.

    Determina cómo la UI renderiza la respuesta.
    """

    TEXT = auto()
    """Respuesta de texto puro o Markdown. El caso más común."""

    TEXT_WITH_ARTIFACTS = auto()
    """Respuesta de texto + uno o más artifacts producidos."""

    APPROVAL_REQUEST = auto()
    """
    El agente necesita aprobación del usuario antes de continuar.
    La UI debe mostrar el diálogo de aprobación.
    """

    THINKING = auto()
    """
    Indicador de progreso — el agente está procesando.
    La UI muestra un spinner o animación de carga.
    """

    ERROR = auto()
    """
    El agente encontró un error que no pudo manejar internamente.
    La UI debe mostrar el mensaje de error al usuario.
    """

    SESSION_CLOSED = auto()
    """La sesión fue cerrada. La UI debe limpiar el estado de conversación."""


@dataclass(frozen=True)
class ArtifactInfo:
    """
    Información de un artifact producido durante el turno.

    La UI usa esta info para mostrar los artifacts en la interfaz
    sin necesitar cargar su contenido completo.
    """

    artifact_id: ArtifactId
    """ID del artifact."""

    artifact_type: ArtifactType
    """Tipo semántico."""

    display_name: str
    """Nombre para mostrar en la UI."""

    type_label: str
    """Etiqueta legible del tipo ('Resumen', 'Flashcards', etc.)."""

    item_count: int | None = None
    """Número de items si aplica (flashcards, resultados, etc.)."""

    has_downloadable_content: bool = False
    """True si el artifact puede exportarse/descargarse."""


@dataclass(frozen=True)
class ThinkingIndicator:
    """
    Indicador de progreso visible para el usuario.

    Describe qué está haciendo el agente en un lenguaje no técnico.
    La UI anima esto mientras espera la respuesta completa.
    """

    message: str
    """Mensaje de progreso. Ej: 'Buscando información...', 'Analizando tu documento...'"""

    tool_id: str | None = None
    """ID de la tool que está ejecutando (solo para logging interno, no para la UI)."""

    step_number: int | None = None
    """Número del paso si es parte de un plan (para mostrar progreso)."""

    total_steps: int | None = None
    """Total de pasos del plan (para la barra de progreso)."""


@dataclass(frozen=True)
class ApprovalRequestInfo:
    """
    Información de una solicitud de aprobación para renderizar en la UI.

    Subset de ApprovalRequest optimizado para la capa de interfaz.
    No incluye IDs técnicos internos salvo approval_id (necesario para la respuesta).
    """

    approval_id: ApprovalId
    """ID de la solicitud — la UI lo envía de vuelta en ApprovalResponseRequest."""

    description: str
    """Descripción legible de qué se quiere hacer."""

    risk_label: str
    """Nivel de riesgo en lenguaje natural ('riesgo moderado', 'bajo riesgo', etc.)."""

    tool_name: str
    """Nombre amigable de la herramienta."""

    can_remember: bool
    """Si la UI debe mostrar el checkbox 'Recordar esta decisión'."""

    context_summary: str = ""
    """Contexto adicional para ayudar al usuario a decidir."""

    risk_explanation: str = ""
    """Explicación del riesgo en lenguaje no técnico."""

    expires_in_seconds: int = 300
    """Segundos hasta que expire la solicitud."""

    is_first_time: bool = False
    """True para mostrar mensaje de onboarding adicional."""


@dataclass(frozen=True)
class AssistantResponse:
    """
    Respuesta completa del agente al turno del usuario.

    Este es el objeto que la UI recibe de ConversationPort después de
    cada turno. Contiene todo lo necesario para renderizar la respuesta:
    el texto, los artifacts producidos, y el estado del sistema.

    La UI no necesita conocer nada interno del sistema — toda la información
    necesaria para renderizar está aquí.
    """

    response_type: AssistantResponseType
    """Tipo de respuesta para determinar el rendering."""

    session_id: SessionId
    """ID de la sesión activa."""

    # --- Contenido principal ---
    text_content: str = ""
    """
    Texto de la respuesta (Markdown o texto plano).
    Vacío para APPROVAL_REQUEST, THINKING, SESSION_CLOSED.
    """

    # --- Artifacts ---
    artifacts_produced: list[ArtifactInfo] = field(default_factory=list)
    """
    Artifacts producidos en este turno.
    La UI los muestra como cards interactivos debajo del texto.
    """

    # --- Control de flujo ---
    approval_request: ApprovalRequestInfo | None = None
    """
    Solicitud de aprobación pendiente.
    Solo presente cuando response_type=APPROVAL_REQUEST.
    La UI debe mostrar el diálogo y esperar la respuesta.
    """

    thinking_indicator: ThinkingIndicator | None = None
    """
    Indicador de progreso actual.
    Solo presente cuando response_type=THINKING.
    """

    # --- Estado de error ---
    error_message: str | None = None
    """
    Mensaje de error legible para el usuario.
    Solo presente cuando response_type=ERROR.
    No expone detalles técnicos internos.
    """

    error_is_retryable: bool = False
    """
    True si la UI puede mostrar un botón "Reintentar".
    Solo relevante cuando response_type=ERROR.
    """

    # --- Metadata del turno ---
    turn_id: str | None = None
    """ID del turn producido en este intercambio (para trazabilidad)."""

    task_id: str | None = None
    """ID de la task ejecutada (para trazabilidad)."""

    duration_ms: float = 0.0
    """Duración total del procesamiento del turno en milisegundos."""

    # --- Acciones sugeridas ---
    suggested_actions: list[str] = field(default_factory=list)
    """
    Acciones sugeridas por el agente para continuar la conversación.
    La UI puede mostrarlas como chips clickeables.
    Ejemplo: ['Generar flashcards', 'Crear quiz', 'Exportar como PDF']
    """

    # --- Estado de la sesión ---
    session_turn_count: int = 0
    """Total de turns en la sesión después de este turno."""

    has_pending_approvals: bool = False
    """True si hay otras aprobaciones pendientes además de esta."""

    @property
    def is_terminal(self) -> bool:
        """True si esta respuesta representa el final de la sesión."""
        return self.response_type == AssistantResponseType.SESSION_CLOSED

    @property
    def requires_user_action(self) -> bool:
        """True si la UI debe mostrar algún control interactivo al usuario."""
        return self.response_type in {
            AssistantResponseType.APPROVAL_REQUEST,
        }

    @property
    def has_artifacts(self) -> bool:
        """True si se produjeron artifacts en este turno."""
        return len(self.artifacts_produced) > 0

    def to_display_dict(self) -> dict[str, Any]:
        """Serializa la respuesta para la UI (formato JSON)."""
        return {
            "type": self.response_type.name.lower(),
            "session_id": self.session_id.to_str(),
            "text": self.text_content,
            "artifacts": [
                {
                    "id": a.artifact_id.to_str(),
                    "type": a.artifact_type.value,
                    "type_label": a.type_label,
                    "name": a.display_name,
                    "item_count": a.item_count,
                    "downloadable": a.has_downloadable_content,
                }
                for a in self.artifacts_produced
            ],
            "approval": self.approval_request.to_ui_dict() if self.approval_request else None,
            "thinking": {
                "message": self.thinking_indicator.message,
                "step": self.thinking_indicator.step_number,
                "total_steps": self.thinking_indicator.total_steps,
            } if self.thinking_indicator else None,
            "error": self.error_message,
            "error_retryable": self.error_is_retryable,
            "suggested_actions": self.suggested_actions,
            "turn_id": self.turn_id,
            "duration_ms": round(self.duration_ms, 1),
            "session_turn_count": self.session_turn_count,
        }


@dataclass(frozen=True)
class StreamingChunk:
    """
    Fragmento de respuesta en modo streaming para la UI.

    El ConversationPort emite StreamingChunks progresivamente mientras
    el LLM genera la respuesta. La UI los acumula y muestra en tiempo real.

    El último chunk tiene is_final=True y contiene la AssistantResponse
    completa acumulada en final_response.
    """

    chunk_type: str
    """
    Tipo del chunk:
    'text_delta'    — fragmento de texto a concatenar
    'thinking'      — indicador de progreso (sin texto)
    'artifact_ready'— un artifact fue producido
    'approval_needed'—solicitud de aprobación lista
    'final'         — último chunk, contiene final_response
    'error'         — error durante el streaming
    """

    text_delta: str = ""
    """Fragmento de texto a concatenar. Solo para chunk_type='text_delta'."""

    thinking_indicator: ThinkingIndicator | None = None
    """Indicador de progreso. Solo para chunk_type='thinking'."""

    artifact: ArtifactInfo | None = None
    """Artifact producido. Solo para chunk_type='artifact_ready'."""

    approval_request: ApprovalRequestInfo | None = None
    """Solicitud de aprobación. Solo para chunk_type='approval_needed'."""

    is_final: bool = False
    """True en el último chunk."""

    final_response: AssistantResponse | None = None
    """Respuesta completa acumulada. Solo presente cuando is_final=True."""

    error_message: str | None = None
    """Mensaje de error. Solo para chunk_type='error'."""


@dataclass(frozen=True)
class SessionInfo:
    """
    Información resumida de una sesión para el selector de sesiones en la UI.
    """

    session_id: SessionId
    status: SessionStatus
    created_at: datetime
    last_activity: datetime
    turn_count: int
    artifact_count: int
    preview_text: str = ""
    """
    Preview del último mensaje del usuario en la sesión.
    Para mostrar en el listado de sesiones sin cargar el historial.
    """

    duration_minutes: float = 0.0
    """Duración de la sesión en minutos."""

    def to_display_dict(self) -> dict[str, Any]:
        """Serializa para la UI del selector de sesiones."""
        return {
            "session_id": self.session_id.to_str(),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "turn_count": self.turn_count,
            "artifact_count": self.artifact_count,
            "preview": self.preview_text,
            "duration_minutes": round(self.duration_minutes, 1),
        }


@dataclass(frozen=True)
class SessionListResponse:
    """Respuesta de listado de sesiones para la UI."""

    sessions: list[SessionInfo]
    total_count: int
    has_more: bool

    def to_display_dict(self) -> dict[str, Any]:
        return {
            "sessions": [s.to_display_dict() for s in self.sessions],
            "total": self.total_count,
            "has_more": self.has_more,
        }


# =============================================================================
# PUERTO INBOUND: CONVERSATION PORT
# =============================================================================


class ConversationPort(abc.ABC):
    """
    Puerto de entrada principal de HiperForge User.

    Define todas las operaciones que la capa de interfaz (Tauri UI, Web UI)
    puede invocar sobre el sistema. Es el único punto de entrada al core
    desde la UI — ningún componente interno es accesible directamente.

    El ConversationCoordinator implementa este puerto. La UI nunca conoce
    la implementación concreta.

    Contratos invariantes:
      1. send_message() es la operación principal — toda interacción pasa por aquí.
      2. Las operaciones de streaming (_streaming) emiten AssistantResponse chunks
         progresivamente. El último chunk tiene is_final=True.
      3. respond_to_approval() debe llamarse dentro del timeout de la solicitud.
         Después del timeout, la respuesta se ignora (ya fue auto-denegada).
      4. Toda operación que involucra una sesión requiere que esta esté ACTIVE.
         Si la sesión no está activa, la operación retorna error con AssistantResponse.
      5. cancel_current_task() es idempotente — cancelar cuando no hay tarea
         activa retorna una respuesta de no-op sin error.
      6. Las operaciones nunca lanzan excepciones al caller de la UI.
         Los errores siempre van en AssistantResponse con response_type=ERROR.
    """

    @abc.abstractmethod
    async def send_message(
        self,
        request: UserMessageRequest,
    ) -> AssistantResponse:
        """
        Procesa un mensaje del usuario y retorna la respuesta completa del agente.

        Esta es la operación más frecuente del sistema. El ConversationCoordinator
        orquesta todo el flujo: construir contexto → llamar al LLM → routing →
        ejecutar tools si aplica → sintetizar → persistir → retornar.

        Para conversaciones donde la UX permite esperar la respuesta completa.
        Para streaming en tiempo real, usar send_message_streaming().

        Args:
            request: UserMessageRequest con el mensaje y adjuntos del usuario.

        Returns:
            AssistantResponse completa con el texto, artifacts y metadata.
            Nunca lanza excepciones — los errores van en response_type=ERROR.
        """

    @abc.abstractmethod
    async def send_message_streaming(
        self,
        request: UserMessageRequest,
    ) -> AsyncIterator[StreamingChunk]:
        """
        Procesa un mensaje del usuario y emite la respuesta en streaming.

        Versión streaming de send_message(). Emite StreamingChunks
        progresivamente para que la UI muestre la respuesta mientras se genera.

        El último chunk tiene is_final=True y final_response con la
        AssistantResponse completa.

        Args:
            request: UserMessageRequest con el mensaje del usuario.

        Yields:
            StreamingChunk con fragmentos progresivos de la respuesta.
        """

    @abc.abstractmethod
    async def respond_to_approval(
        self,
        request: ApprovalResponseRequest,
    ) -> AssistantResponse:
        """
        Registra la respuesta del usuario a una solicitud de aprobación.

        El TaskExecutor estaba esperando esta respuesta (await_decision()).
        Al recibirla, el ConversationCoordinator desbloquea la ejecución
        (si aprobó) o la cancela (si denegó).

        Args:
            request: ApprovalResponseRequest con la decisión del usuario.

        Returns:
            AssistantResponse con el resultado después de procesar la aprobación.
            Si aprobó: continúa la ejecución y retorna el resultado.
            Si denegó: retorna confirmación de cancelación.
        """

    @abc.abstractmethod
    async def cancel_current_task(
        self,
        request: CancelTaskRequest,
    ) -> AssistantResponse:
        """
        Cancela la tarea activa de la sesión.

        Si hay una tarea en ejecución (EXECUTING o AWAITING_APPROVAL),
        la cancela y retorna una confirmación al usuario.
        Si no hay tarea activa, retorna un no-op.

        Args:
            request: CancelTaskRequest con el ID de la sesión.

        Returns:
            AssistantResponse confirmando la cancelación.
        """

    @abc.abstractmethod
    async def create_session(
        self,
        request: CreateSessionRequest,
    ) -> tuple[SessionId, AssistantResponse]:
        """
        Crea una nueva sesión conversacional.

        Genera un nuevo SessionId, crea el aggregate Session, y
        opcionalmente restaura contexto de la sesión anterior.

        Args:
            request: CreateSessionRequest con el user_id.

        Returns:
            Tuple de (SessionId, AssistantResponse) con el mensaje de bienvenida.
        """

    @abc.abstractmethod
    async def switch_session(
        self,
        request: SwitchSessionRequest,
    ) -> AssistantResponse:
        """
        Cambia la sesión activa a una sesión existente.

        Pausa la sesión actual (si hay una), reactiva la sesión solicitada,
        y retorna el estado de esa sesión.

        Args:
            request: SwitchSessionRequest con el ID de la sesión destino.

        Returns:
            AssistantResponse con el estado de la sesión restaurada.
        """

    @abc.abstractmethod
    async def close_session(
        self,
        session_id: SessionId,
    ) -> AssistantResponse:
        """
        Cierra explícitamente la sesión activa.

        Genera el resumen de la sesión, persiste el estado final,
        y retorna una confirmación al usuario.

        Args:
            session_id: ID de la sesión a cerrar.

        Returns:
            AssistantResponse con el resumen de la sesión cerrada.
        """

    @abc.abstractmethod
    async def list_sessions(
        self,
        user_id: UserId,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> SessionListResponse:
        """
        Lista las sesiones del usuario para el selector de sesiones.

        Args:
            user_id: ID del usuario.
            limit:   Máximo de sesiones a retornar.
            offset:  Desplazamiento para paginación.

        Returns:
            SessionListResponse con las sesiones del usuario.
        """

    @abc.abstractmethod
    async def get_session_artifacts(
        self,
        session_id: SessionId,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ArtifactSummary]:
        """
        Retorna los artifacts de una sesión para la vista de artifacts de la UI.

        Args:
            session_id: ID de la sesión.
            limit:      Máximo de artifacts a retornar.
            offset:     Desplazamiento para paginación.

        Returns:
            Lista de ArtifactSummary de la sesión.
        """

    @abc.abstractmethod
    async def get_artifact_content(
        self,
        artifact_id: ArtifactId,
        session_id: SessionId,
    ) -> dict[str, Any]:
        """
        Retorna el contenido completo de un artifact para la vista de detalle.

        La UI llama a este método cuando el usuario hace clic en un artifact
        para verlo en detalle.

        Args:
            artifact_id: ID del artifact.
            session_id:  ID de la sesión propietaria (para autorización).

        Returns:
            Dict con el contenido completo del artifact en el formato de la UI.
        """

    @abc.abstractmethod
    async def export_artifact(
        self,
        artifact_id: ArtifactId,
        session_id: SessionId,
        *,
        format: str = "markdown",
    ) -> bytes:
        """
        Exporta un artifact en el formato especificado.

        La UI llama a este método cuando el usuario elige descargar un artifact.

        Args:
            artifact_id: ID del artifact a exportar.
            session_id:  ID de la sesión propietaria.
            format:      Formato de exportación: 'markdown', 'json', 'txt'.

        Returns:
            Bytes del archivo exportado.
        """

    @abc.abstractmethod
    async def get_current_session_state(
        self,
        session_id: SessionId,
    ) -> dict[str, Any]:
        """
        Retorna el estado actual de la sesión para la UI.

        Incluye: status, turn_count, pending_approvals, task_status, etc.
        La UI lo usa para sincronizar su estado con el backend (ej: al reconectar).

        Args:
            session_id: ID de la sesión.

        Returns:
            Dict con el estado completo de la sesión para la UI.
        """


# =============================================================================
# PUERTO INBOUND: ADMIN PORT (debug CLI)
# =============================================================================


class AdminPort(abc.ABC):
    """
    Puerto de entrada para el debug CLI de administración.

    Solo disponible en modo --admin-cli. No expuesto en producción.
    Proporciona acceso a diagnósticos, métricas y audit log.

    La UI nunca importa este puerto — es exclusivo del debug CLI.
    """

    @abc.abstractmethod
    async def get_session_list(
        self,
        *,
        limit: int = 20,
        include_closed: bool = True,
    ) -> list[dict[str, Any]]:
        """Lista sesiones con metadata técnica completa."""

    @abc.abstractmethod
    async def inspect_session(
        self,
        session_id: SessionId,
    ) -> dict[str, Any]:
        """Retorna toda la información técnica de una sesión."""

    @abc.abstractmethod
    async def query_audit_log(
        self,
        *,
        session_id: str | None = None,
        tool_id: str | None = None,
        last_hours: int = 24,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Consulta el audit log con filtros."""

    @abc.abstractmethod
    async def get_policy_status(self) -> dict[str, Any]:
        """Retorna el estado del PolicyEngine y las políticas activas."""

    @abc.abstractmethod
    async def get_tool_list(
        self,
        *,
        category: str | None = None,
        include_disabled: bool = False,
    ) -> list[dict[str, Any]]:
        """Lista las tools registradas con sus metadatos."""

    @abc.abstractmethod
    async def get_security_report(
        self,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Retorna el reporte de seguridad del PolicyEngine."""

    @abc.abstractmethod
    async def get_metrics_summary(self) -> dict[str, Any]:
        """Retorna un resumen de métricas del sistema."""

    @abc.abstractmethod
    async def get_config(self) -> dict[str, Any]:
        """Retorna la configuración activa del sistema (sin secrets)."""

    @abc.abstractmethod
    async def reset_session_security(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Reinicia el contexto de seguridad de una sesión.

        Solo disponible en el debug CLI. Permite salir del lockdown
        sin reiniciar la aplicación completa.
        """

    @abc.abstractmethod
    async def force_session_close(
        self,
        session_id: SessionId,
    ) -> dict[str, Any]:
        """
        Fuerza el cierre de una sesión.

        Útil cuando una sesión quedó en estado inconsistente.
        """