"""
Protocolo abstracto de tools del ecosistema Forge Platform.

Este módulo define el contrato completo que toda tool del sistema debe cumplir.
Es el segundo pilar de la arquitectura hexagonal de Forge — junto con LLMPort,
define las dos capacidades externas fundamentales del agente: razonar (LLM) y
actuar (Tools).

El protocolo está diseñado con seguridad como restricción arquitectónica, no
como feature opcional. Cada tool declara su risk_classification, mutation_type
y plataformas soportadas de forma obligatoria. El PolicyEngine usa estos metadatos
para tomar decisiones de ejecución sin necesidad de conocer los internals de
cada tool.

Principios de diseño:
  1. Las tools son ejecutores tontos. Toda la inteligencia (cuándo usar una tool,
     con qué argumentos) está en el LLM y la capa de aplicación. Una tool busca,
     lee, escribe — no decide.
  2. Toda tool declara su riesgo honestamente. Una tool que miente en su
     risk_classification compromete todo el modelo de seguridad.
  3. Las tools de lectura (NONE/LOW risk) se ejecutan directamente. Las tools
     de mutación (MEDIUM+) pasan por sandbox o approval según la clasificación.
  4. El schema de input/output es inmutable por versión. Si el schema cambia,
     la versión de la tool incrementa — el registry puede mantener múltiples
     versiones en paralelo (preparado para V3).
  5. Toda invocación produce un ToolExecutionRecord completo para el audit log,
     independientemente de si fue exitosa o falló.

Jerarquía de tipos:

  RiskLevel            — clasificación de riesgo de una acción
  MutationType         — tipo de mutación que produce la tool
  Platform             — plataforma de SO soportada
  ToolCategory         — categoría funcional de la tool
  ToolCapability       — metadatos de seguridad y comportamiento de la tool
  ToolSchema           — definición completa de la tool (extiende LLM ToolDefinition)
  ToolInput            — input validado para una invocación
  ToolOutput           — output estructurado de una invocación exitosa
  ToolExecutionRecord  — registro completo de una invocación para audit
  RiskAssessment       — evaluación de riesgo de una invocación específica
  ToolPort (ABC)       — contrato que toda tool debe implementar
"""

from __future__ import annotations

import abc
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from forge_core.llm.protocol import ToolDefinition, ToolParameterSchema


# =============================================================================
# ENUMERACIONES DE CLASIFICACIÓN
# =============================================================================


class RiskLevel(int, Enum):
    """
    Nivel de riesgo de una acción o tool.

    Hereda de int para permitir comparaciones ordinales directas:
        RiskLevel.LOW < RiskLevel.MEDIUM → True
        max(RiskLevel.NONE, RiskLevel.HIGH) → RiskLevel.HIGH

    La clasificación de riesgo es el mecanismo central por el que el PolicyEngine
    decide qué acciones requieren aprobación explícita y cuáles pueden ejecutarse
    automáticamente. Toda tool debe declarar honestamente su nivel base.
    """

    NONE = 0
    """
    Sin riesgo apreciable. Operaciones de solo lectura o generación de contenido
    que no interactúan con el sistema operativo ni con recursos externos mutables.

    Ejemplos: conversación pura, generación de texto, cálculos, búsqueda web
    (lectura de resultados), análisis de documentos.

    Ejecución: directa, sin sandbox, sin approval.
    Audit: log básico.
    """

    LOW = 1
    """
    Riesgo bajo. Lectura de recursos del sistema dentro de los límites permitidos,
    o interacciones con APIs externas de solo lectura.

    Ejemplos: leer archivos en directorios del usuario, tomar screenshots,
    obtener información del sistema, fetch de páginas web.

    Ejecución: directa, sin sandbox, sin approval (dentro de paths permitidos).
    Audit: log con metadata del recurso accedido.
    """

    MEDIUM = 2
    """
    Riesgo medio. Escritura de datos, modificaciones de estado local, o
    interacciones con el sistema operativo que pueden tener efectos visibles.

    Ejemplos: escribir archivos, modificar clipboard, lanzar aplicaciones
    (de la allowlist), enviar notificaciones del sistema.

    Ejecución: con sandbox para acciones de desktop, approval en primer uso.
    Audit: log completo con input/output summary.
    """

    HIGH = 3
    """
    Riesgo alto. Acciones con efectos secundarios significativos o potencialmente
    difíciles de revertir. Acceso a recursos fuera de los límites por defecto.

    Ejemplos: leer archivos fuera de dirs permitidos, acciones de configuración
    del sistema, lanzar apps fuera de la allowlist.

    Ejecución: sandbox obligatorio, approval siempre requerido.
    Audit: log detallado con toda la metadata disponible.
    """

    CRITICAL = 4
    """
    Riesgo crítico. Acciones bloqueadas por el PolicyEngine.

    Ejemplos: ejecución de comandos shell arbitrarios, modificación de archivos
    de sistema, acceso a credenciales, instalación de software.

    Ejecución: BLOQUEADA. PolicyEngine retorna DENY sin posibilidad de override.
    Audit: log de intento con nivel CRITICAL.
    """

    def requires_approval(self) -> bool:
        """Indica si este nivel de riesgo requiere aprobación del usuario."""
        return self >= RiskLevel.MEDIUM

    def requires_sandbox(self) -> bool:
        """Indica si este nivel de riesgo requiere ejecución en sandbox."""
        return self >= RiskLevel.MEDIUM

    def is_blocked(self) -> bool:
        """Indica si este nivel de riesgo está bloqueado por el PolicyEngine."""
        return self == RiskLevel.CRITICAL

    def label(self) -> str:
        """Retorna el label legible para el usuario (no técnico)."""
        labels = {
            RiskLevel.NONE: "sin riesgo",
            RiskLevel.LOW: "bajo riesgo",
            RiskLevel.MEDIUM: "riesgo moderado",
            RiskLevel.HIGH: "riesgo alto",
            RiskLevel.CRITICAL: "bloqueado",
        }
        return labels[self]


class MutationType(str, Enum):
    """
    Tipo de mutación que produce la tool sobre el estado del sistema.

    Determina si la ejecución es segura para retry (idempotencia) y qué
    tipo de reversión es posible en caso de error.
    """

    NONE = "none"
    """
    La tool no modifica ningún estado. Operaciones de solo lectura o
    generación de contenido efímero. Siempre segura para retry.
    """

    LOCAL_DATA = "local_data"
    """
    La tool modifica datos locales del usuario (archivos, notas, artifacts).
    El retry puede crear duplicados si no se verifica idempotencia.
    """

    OS_STATE = "os_state"
    """
    La tool modifica estado del sistema operativo (clipboard, apps abiertas,
    notificaciones). Los efectos son visibles para el usuario.
    """

    EXTERNAL = "external"
    """
    La tool interactúa con servicios externos de forma mutante (POST, PUT, DELETE).
    No implementado en V1 — reservado para integraciones futuras.
    """


class Platform(str, Enum):
    """Plataforma de sistema operativo."""

    LINUX = "linux"
    WINDOWS = "windows"
    DARWIN = "darwin"
    ALL = "all"
    """La tool funciona en todas las plataformas (cross-platform puro)."""


class ToolCategory(str, Enum):
    """
    Categoría funcional de la tool.

    Usada para agrupar tools en la UI, filtrar por tipo en el PolicyEngine
    y gestionar permisos a nivel de categoría en el PermissionSet.
    """

    RESEARCH = "research"
    """Búsqueda web, fetch de páginas, búsqueda académica."""

    DOCUMENTS = "documents"
    """Lectura, análisis y extracción de texto de documentos."""

    STUDY = "study"
    """Generación de flashcards, quizzes, mapas conceptuales."""

    DESKTOP = "desktop"
    """Automatización de escritorio: apps, clipboard, screenshots."""

    FILESYSTEM = "filesystem"
    """Lectura y escritura de archivos del sistema local."""

    PRODUCTIVITY = "productivity"
    """Notas, TODOs, organización de información."""

    SYSTEM = "system"
    """Información del sistema, métricas, diagnóstico."""


# =============================================================================
# METADATOS DE CAPABILITIES
# =============================================================================


class RateLimit(BaseModel):
    """
    Límite de tasa para invocaciones de una tool.

    Protege contra loops involuntarios y abuso de APIs externas con
    límites de facturación. El ToolDispatch verifica este límite antes
    de cada invocación.
    """

    model_config = {"frozen": True}

    max_calls: int = Field(
        ge=1,
        description="Número máximo de invocaciones permitidas en la ventana de tiempo.",
    )
    window_seconds: int = Field(
        ge=1,
        description="Duración de la ventana de tiempo en segundos.",
    )
    per_session: bool = Field(
        default=True,
        description=(
            "Si True, el límite aplica por sesión. "
            "Si False, aplica globalmente a todas las sesiones activas."
        ),
    )

    def __str__(self) -> str:
        scope = "por sesión" if self.per_session else "global"
        return f"{self.max_calls} llamadas / {self.window_seconds}s ({scope})"


class ToolCapability(BaseModel):
    """
    Metadatos de seguridad y comportamiento de una tool.

    Este es el objeto que el PolicyEngine consulta para tomar decisiones.
    Toda tool DEBE declarar su ToolCapability honestamente — el sistema
    confía en que este contrato sea verídico.

    La ToolCapability es inmutable — se define en el momento del registro
    de la tool y no cambia durante el ciclo de vida de la aplicación.
    """

    model_config = {"frozen": True}

    # --- Clasificación de riesgo ---
    risk_classification: RiskLevel = Field(
        description=(
            "Nivel de riesgo BASE de la tool en condiciones normales de uso. "
            "El PolicyEngine puede elevar este nivel si el input específico "
            "es más peligroso que el uso típico (ej: file_read en /etc/passwd)."
        ),
    )
    requires_approval_above: RiskLevel = Field(
        default=RiskLevel.HIGH,
        description=(
            "Nivel de riesgo a partir del cual se requiere aprobación explícita "
            "del usuario. Por defecto HIGH — MEDIUM puede ejecutarse con sandbox "
            "sin approval en primer uso si el usuario lo configuró así."
        ),
    )

    # --- Comportamiento de mutación ---
    mutation_type: MutationType = Field(
        description="Tipo de mutación que produce la tool. NONE para tools de solo lectura.",
    )
    idempotent: bool = Field(
        description=(
            "True si la tool produce el mismo resultado al ejecutarse múltiples "
            "veces con los mismos argumentos (seguro para retry). "
            "False si el retry puede crear duplicados o efectos acumulativos."
        ),
    )
    has_rollback: bool = Field(
        default=False,
        description=(
            "True si la tool puede deshacer su efecto via rollback(). "
            "Solo las tools de mutación pueden implementar rollback."
        ),
    )

    # --- Plataformas soportadas ---
    platform_support: list[Platform] = Field(
        description=(
            "Plataformas en las que la tool está disponible. "
            "[Platform.ALL] para tools cross-platform puras."
        ),
    )

    # --- Timeouts ---
    default_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Timeout por defecto para la ejecución de esta tool en segundos.",
    )

    # --- Rate limiting ---
    rate_limit: RateLimit | None = Field(
        default=None,
        description=(
            "Límite de tasa para esta tool. None deshabilita el rate limiting. "
            "Requerido para tools que llaman a APIs externas con billing."
        ),
    )

    # --- Sandbox ---
    requires_sandbox: bool = Field(
        default=False,
        description=(
            "True si la tool SIEMPRE debe ejecutarse en el proceso sandbox, "
            "independientemente del nivel de riesgo del input. "
            "Obligatorio para todas las tools de desktop automation."
        ),
    )

    @model_validator(mode="after")
    def validate_sandbox_consistency(self) -> ToolCapability:
        """
        Valida consistencia entre sandbox y nivel de riesgo.

        Las tools de riesgo CRITICAL nunca deberían llegar al sandbox —
        deben ser bloqueadas por el PolicyEngine antes.
        Las tools MEDIUM con mutation_type=OS_STATE siempre requieren sandbox.
        """
        if (
            self.mutation_type == MutationType.OS_STATE
            and self.risk_classification >= RiskLevel.MEDIUM
            and not self.requires_sandbox
        ):
            raise ValueError(
                "Las tools con mutation_type=OS_STATE y risk_classification>=MEDIUM "
                "deben declarar requires_sandbox=True por política de seguridad."
            )
        return self

    def is_supported_on(self, platform: Platform) -> bool:
        """
        Verifica si la tool está soportada en la plataforma especificada.

        Args:
            platform: Plataforma a verificar.

        Returns:
            True si la tool está disponible en esa plataforma.
        """
        return Platform.ALL in self.platform_support or platform in self.platform_support

    def effective_risk_for_input(
        self,
        input_risk_override: RiskLevel | None = None,
    ) -> RiskLevel:
        """
        Calcula el nivel de riesgo efectivo para una invocación específica.

        El PolicyEngine puede elevar el riesgo base de la tool si el input
        concreto es más peligroso que el uso típico (ej: file_read fuera de
        dirs permitidos). Este método retorna el riesgo efectivo máximo.

        Args:
            input_risk_override: Riesgo adicional detectado por el PolicyEngine
                                 basándose en el input concreto. None si no hay
                                 elevación de riesgo.

        Returns:
            El nivel de riesgo efectivo (max del base y el override).
        """
        if input_risk_override is None:
            return self.risk_classification
        return max(self.risk_classification, input_risk_override)


# =============================================================================
# INPUT Y OUTPUT DE TOOLS
# =============================================================================


class ToolInput(BaseModel):
    """
    Input validado para la invocación de una tool.

    Encapsula los argumentos proporcionados por el LLM para una invocación
    de tool. Los argumentos son validados contra el schema de la tool antes
    de que el ToolInput sea creado — si llegan aquí, están bien formados.

    ToolInput es inmutable. El adapter de la tool recibe una instancia de esta
    clase y no debe mutar su estado.
    """

    model_config = {"frozen": True}

    tool_id: str = Field(
        description="Identificador de la tool a invocar.",
    )
    invocation_id: str = Field(
        description=(
            "ID único de esta invocación, correlacionado con el ToolCall.id del LLM. "
            "Usado para correlacionar en audit log y tracing."
        ),
    )
    arguments: dict[str, Any] = Field(
        description="Argumentos para la tool, ya validados contra su schema.",
    )
    session_id: str | None = Field(
        default=None,
        description="ID de la sesión para contexto y audit.",
    )
    task_id: str | None = Field(
        default=None,
        description="ID de la tarea padre para contexto y audit.",
    )
    timeout_seconds: float | None = Field(
        default=None,
        ge=1.0,
        description=(
            "Override del timeout para esta invocación específica. "
            "None usa el timeout por defecto de la tool."
        ),
    )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtiene un argumento por nombre con valor por defecto.

        Args:
            key:     Nombre del argumento.
            default: Valor por defecto si el argumento no existe.

        Returns:
            El valor del argumento o el default.
        """
        return self.arguments.get(key, default)

    def require(self, key: str) -> Any:
        """
        Obtiene un argumento requerido por nombre.

        Args:
            key: Nombre del argumento requerido.

        Returns:
            El valor del argumento.

        Raises:
            KeyError: Si el argumento no está presente.
        """
        if key not in self.arguments:
            raise KeyError(
                f"Argumento requerido '{key}' no presente en el input de la tool "
                f"'{self.tool_id}'. Argumentos disponibles: {list(self.arguments.keys())}"
            )
        return self.arguments[key]


class ToolOutput(BaseModel):
    """
    Output estructurado de una invocación exitosa de tool.

    Toda tool produce un ToolOutput con su resultado. El resultado puede
    ser texto libre, datos estructurados, o referencias a artifacts.
    El LLM recibirá el contenido serializado como string en el siguiente turno.
    """

    model_config = {"frozen": True}

    tool_id: str = Field(
        description="ID de la tool que produjo este output.",
    )
    invocation_id: str = Field(
        description="ID de la invocación que produjo este output.",
    )
    content: Any = Field(
        description=(
            "Resultado de la tool. Puede ser str, dict, list, o cualquier tipo "
            "serializable a JSON. El ToolDispatch lo serializa antes de devolverlo al LLM."
        ),
    )
    content_type: str = Field(
        default="text/plain",
        description=(
            "MIME type del contenido para que el ToolDispatch sepa cómo serializarlo. "
            "Ejemplos: 'text/plain', 'application/json', 'text/markdown'."
        ),
    )
    artifact_ids: list[str] = Field(
        default_factory=list,
        description=(
            "IDs de artifacts producidos por esta invocación y registrados en el "
            "ArtifactManager. El ArtifactManager los almacena; el output solo "
            "contiene sus referencias."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Metadatos adicionales de la ejecución para debugging y observabilidad. "
            "No se envía al LLM — solo para uso interno del sistema."
        ),
    )
    truncated: bool = Field(
        default=False,
        description=(
            "True si el output fue truncado para caber en el context window del LLM. "
            "Cuando True, el LLM debe saber que la respuesta puede estar incompleta."
        ),
    )

    def serialize_for_llm(self, max_chars: int = 8000) -> str:
        """
        Serializa el output para enviarlo al LLM como resultado de tool.

        Convierte el contenido al formato string que el LLM puede procesar.
        Si el contenido es muy largo, lo trunca respetando el límite indicado.

        Args:
            max_chars: Número máximo de caracteres en el output serializado.
                       Protege el context window del LLM de outputs excesivos.

        Returns:
            String con el contenido serializado, truncado si es necesario.
        """
        import json

        if isinstance(self.content, str):
            raw = self.content
        elif isinstance(self.content, (dict, list)):
            try:
                raw = json.dumps(self.content, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                raw = str(self.content)
        else:
            raw = str(self.content)

        if len(raw) <= max_chars:
            return raw

        truncation_notice = f"\n\n[...output truncado a {max_chars} caracteres...]"
        return raw[:max_chars - len(truncation_notice)] + truncation_notice


# =============================================================================
# REGISTRO DE EJECUCIÓN PARA AUDIT
# =============================================================================


class ToolExecutionRecord(BaseModel):
    """
    Registro completo de una invocación de tool para el audit log.

    Se crea por cada invocación de tool, independientemente del resultado.
    Este objeto es la fuente de verdad para el audit trail del sistema.
    Es inmutable una vez creado — el audit log es append-only.
    """

    model_config = {"frozen": True}

    invocation_id: str = Field(description="ID único de la invocación.")
    tool_id: str = Field(description="ID de la tool invocada.")
    session_id: str | None = Field(description="ID de la sesión.")
    task_id: str | None = Field(description="ID de la tarea padre.")

    # --- Timing ---
    started_at: datetime = Field(description="Momento de inicio de la ejecución (UTC).")
    completed_at: datetime | None = Field(
        default=None,
        description="Momento de finalización. None si aún en ejecución.",
    )

    # --- Resultado ---
    success: bool = Field(description="True si la ejecución fue exitosa.")
    error_code: str | None = Field(
        default=None,
        description="Código de error ForgeError si la ejecución falló.",
    )

    # --- Seguridad ---
    risk_level: RiskLevel = Field(description="Nivel de riesgo efectivo de esta invocación.")
    policy_decision: str = Field(
        description="Decisión del policy engine: 'allow', 'deny', 'require_approval'.",
    )
    approval_id: str | None = Field(
        default=None,
        description="ID de la aprobación del usuario si fue requerida.",
    )
    sandbox_used: bool = Field(
        default=False,
        description="True si la ejecución ocurrió en el proceso sandbox.",
    )

    # --- Resúmenes sanitizados (NUNCA el contenido completo) ---
    input_summary: str | None = Field(
        default=None,
        description=(
            "Resumen breve del input, sanitizado. "
            "NUNCA incluir paths completos de archivos sensibles ni contenidos."
        ),
    )
    output_summary: str | None = Field(
        default=None,
        description="Resumen breve del output, sanitizado.",
    )

    @property
    def duration_ms(self) -> float | None:
        """Duración de la ejecución en milisegundos. None si aún en ejecución."""
        if self.completed_at is None:
            return None
        delta = self.completed_at - self.started_at
        return delta.total_seconds() * 1000

    @classmethod
    def start(
        cls,
        *,
        invocation_id: str,
        tool_id: str,
        session_id: str | None,
        task_id: str | None,
        risk_level: RiskLevel,
        policy_decision: str,
        sandbox_used: bool = False,
        approval_id: str | None = None,
        input_summary: str | None = None,
    ) -> ToolExecutionRecord:
        """
        Factory: crea un registro de inicio de ejecución.

        Se llama al comenzar la ejecución, antes de conocer el resultado.
        El campo completed_at queda None hasta que se llame a complete().

        Args:
            invocation_id:   ID único de la invocación.
            tool_id:         ID de la tool.
            session_id:      ID de la sesión.
            task_id:         ID de la tarea padre.
            risk_level:      Nivel de riesgo efectivo.
            policy_decision: Decisión del policy engine.
            sandbox_used:    Si se usó el sandbox process.
            approval_id:     ID de la aprobación si fue requerida.
            input_summary:   Resumen sanitizado del input.

        Returns:
            ToolExecutionRecord con success=False y completed_at=None.
        """
        return cls(
            invocation_id=invocation_id,
            tool_id=tool_id,
            session_id=session_id,
            task_id=task_id,
            started_at=datetime.now(tz=timezone.utc),
            success=False,  # se actualiza al completar
            risk_level=risk_level,
            policy_decision=policy_decision,
            sandbox_used=sandbox_used,
            approval_id=approval_id,
            input_summary=input_summary,
        )

    def complete(
        self,
        *,
        success: bool,
        output_summary: str | None = None,
        error_code: str | None = None,
    ) -> ToolExecutionRecord:
        """
        Crea una nueva instancia del registro marcándolo como completado.

        Como el modelo es frozen (inmutable), devuelve una nueva instancia
        con completed_at, success y output_summary actualizados.

        Args:
            success:        True si la ejecución fue exitosa.
            output_summary: Resumen sanitizado del output.
            error_code:     Código de error si falló.

        Returns:
            Nueva instancia de ToolExecutionRecord con los campos de completión.
        """
        return self.model_copy(update={
            "completed_at": datetime.now(tz=timezone.utc),
            "success": success,
            "output_summary": output_summary,
            "error_code": error_code,
        })


# =============================================================================
# EVALUACIÓN DE RIESGO
# =============================================================================


class RiskAssessment(BaseModel):
    """
    Evaluación de riesgo de una invocación específica de tool.

    El PolicyEngine produce un RiskAssessment después de evaluar la tool
    y su input concreto. Combina el riesgo base de la tool con el riesgo
    específico del input (paths, argumentos, contexto).

    Un RiskAssessment es siempre específico a un (tool, input) par — no es
    un atributo estático de la tool.
    """

    model_config = {"frozen": True}

    tool_id: str = Field(description="ID de la tool evaluada.")
    base_risk: RiskLevel = Field(
        description="Riesgo base declarado por la tool en su ToolCapability.",
    )
    input_risk: RiskLevel = Field(
        description=(
            "Riesgo adicional detectado por el PolicyEngine basándose en el "
            "input concreto. NONE si el input no eleva el riesgo base."
        ),
    )
    effective_risk: RiskLevel = Field(
        description="Riesgo efectivo = max(base_risk, input_risk).",
    )
    requires_approval: bool = Field(
        description="True si el riesgo efectivo requiere aprobación del usuario.",
    )
    requires_sandbox: bool = Field(
        description="True si la ejecución debe ocurrir en el proceso sandbox.",
    )
    justification: str = Field(
        description=(
            "Explicación legible de la evaluación de riesgo. Se usa para "
            "mostrar al usuario en la UI cuando se requiere aprobación, "
            "y para el audit log."
        ),
    )
    blocking_reason: str | None = Field(
        default=None,
        description=(
            "Si el riesgo es CRITICAL, razón por la que la acción está bloqueada. "
            "None si la acción puede ejecutarse (con o sin approval)."
        ),
    )
    elevated_by: list[str] = Field(
        default_factory=list,
        description=(
            "Lista de razones por las que el riesgo fue elevado sobre el nivel base. "
            "Vacío si effective_risk == base_risk."
        ),
    )

    @model_validator(mode="after")
    def validate_effective_risk(self) -> RiskAssessment:
        """Valida que el effective_risk sea el máximo de base e input risk."""
        expected = max(self.base_risk, self.input_risk)
        if self.effective_risk != expected:
            raise ValueError(
                f"effective_risk ({self.effective_risk}) debe ser max("
                f"base_risk={self.base_risk}, input_risk={self.input_risk}) = {expected}"
            )
        return self

    @property
    def is_blocked(self) -> bool:
        """True si la acción está bloqueada y no puede ejecutarse."""
        return self.effective_risk.is_blocked()

    @property
    def is_safe_to_execute_directly(self) -> bool:
        """
        True si la acción puede ejecutarse sin approval ni sandbox especial.

        Equivalente a: riesgo efectivo < MEDIUM y no requiere sandbox especial.
        """
        return (
            self.effective_risk < RiskLevel.MEDIUM
            and not self.requires_sandbox
            and not self.requires_approval
        )

    @classmethod
    def allow_direct(
        cls,
        tool_id: str,
        base_risk: RiskLevel,
    ) -> RiskAssessment:
        """
        Factory: crea un RiskAssessment para ejecución directa sin restricciones.

        Usado cuando el PolicyEngine determina que la acción es segura
        para ejecutarse sin approval ni sandbox.

        Args:
            tool_id:   ID de la tool.
            base_risk: Riesgo base de la tool.

        Returns:
            RiskAssessment con effective_risk=base_risk, sin approval ni sandbox.
        """
        return cls(
            tool_id=tool_id,
            base_risk=base_risk,
            input_risk=RiskLevel.NONE,
            effective_risk=base_risk,
            requires_approval=False,
            requires_sandbox=False,
            justification=f"Acción de {base_risk.label()} dentro de los límites permitidos.",
        )

    @classmethod
    def require_approval(
        cls,
        tool_id: str,
        base_risk: RiskLevel,
        effective_risk: RiskLevel,
        *,
        justification: str,
        elevated_by: list[str] | None = None,
        input_risk: RiskLevel = RiskLevel.NONE,
    ) -> RiskAssessment:
        """
        Factory: crea un RiskAssessment que requiere aprobación del usuario.

        Args:
            tool_id:       ID de la tool.
            base_risk:     Riesgo base de la tool.
            effective_risk: Riesgo efectivo después de evaluar el input.
            justification: Descripción legible para mostrar al usuario.
            elevated_by:   Razones de elevación de riesgo.
            input_risk:    Riesgo específico detectado en el input.

        Returns:
            RiskAssessment con requires_approval=True.
        """
        return cls(
            tool_id=tool_id,
            base_risk=base_risk,
            input_risk=input_risk,
            effective_risk=effective_risk,
            requires_approval=True,
            requires_sandbox=effective_risk.requires_sandbox(),
            justification=justification,
            elevated_by=elevated_by or [],
        )

    @classmethod
    def block(
        cls,
        tool_id: str,
        base_risk: RiskLevel,
        *,
        reason: str,
    ) -> RiskAssessment:
        """
        Factory: crea un RiskAssessment que bloquea la acción.

        Usado cuando el PolicyEngine determina que la acción es CRITICAL
        y debe denegarse sin posibilidad de override.

        Args:
            tool_id:   ID de la tool.
            base_risk: Riesgo base de la tool.
            reason:    Razón por la que la acción está bloqueada.

        Returns:
            RiskAssessment con effective_risk=CRITICAL y blocking_reason.
        """
        return cls(
            tool_id=tool_id,
            base_risk=base_risk,
            input_risk=RiskLevel.CRITICAL,
            effective_risk=RiskLevel.CRITICAL,
            requires_approval=False,  # no se puede aprobar — está bloqueado
            requires_sandbox=False,   # no se ejecuta — está bloqueado
            justification=f"Acción bloqueada: {reason}",
            blocking_reason=reason,
        )


# =============================================================================
# SCHEMA DE TOOL (para LLM y registro)
# =============================================================================


class ToolSchema(BaseModel):
    """
    Definición completa de una tool que combina la interfaz LLM con la metadata
    de seguridad y comportamiento del sistema Forge.

    ToolSchema es el objeto que se registra en el ToolRegistry y del que el
    ToolDispatch extrae tanto la definición para el LLM (via to_tool_definition())
    como los metadatos para el PolicyEngine (via capability).
    """

    model_config = {"frozen": True}

    # --- Identificación ---
    tool_id: str = Field(
        pattern=r"^[a-zA-Z0-9_]{1,64}$",
        description="Identificador único de la tool en el sistema Forge (snake_case).",
    )
    version: str = Field(
        default="1.0.0",
        pattern=r"^\d+\.\d+\.\d+$",
        description="Versión semántica de la tool.",
    )
    display_name: str = Field(
        description="Nombre legible para el usuario en la UI.",
    )
    category: ToolCategory = Field(
        description="Categoría funcional de la tool.",
    )

    # --- Interfaz LLM ---
    llm_name: str = Field(
        pattern=r"^[a-zA-Z0-9_-]{1,64}$",
        description=(
            "Nombre de la tool tal como el LLM la conoce para function calling. "
            "Puede diferir de tool_id para ser más descriptivo al LLM."
        ),
    )
    llm_description: str = Field(
        min_length=20,
        max_length=1024,
        description=(
            "Descripción para el LLM: qué hace la tool, cuándo usarla, "
            "qué NO hace. Crítico para que el LLM tome buenas decisiones."
        ),
    )
    parameters: ToolParameterSchema = Field(
        description="Schema JSON de los parámetros de entrada para function calling.",
    )

    # --- Metadata de seguridad ---
    capability: ToolCapability = Field(
        description="Metadatos de seguridad y comportamiento de la tool.",
    )

    # --- Configuración adicional ---
    enabled: bool = Field(
        default=True,
        description=(
            "Si False, la tool está registrada pero no disponible para el LLM. "
            "Permite deshabilitar tools temporalmente sin eliminarlas del registry."
        ),
    )

    @field_validator("llm_name")
    @classmethod
    def validate_llm_name_consistency(cls, v: str, info: Any) -> str:
        """
        El llm_name debe ser descriptivo pero no revelar implementación interna.
        Se acepta cualquier nombre válido — esta validación es extensible.
        """
        return v

    def to_tool_definition(self) -> ToolDefinition:
        """
        Convierte el ToolSchema en un ToolDefinition para enviarlo al LLM.

        El ToolDefinition es lo que el LLM recibe en el context para saber
        qué tools puede invocar. Solo incluye la interfaz pública de la tool,
        sin metadatos de seguridad internos.

        Returns:
            ToolDefinition listo para incluir en LLMContext.tools.
        """
        return ToolDefinition(
            name=self.llm_name,
            description=self.llm_description,
            parameters=self.parameters,
            strict=True,
        )

    def is_available_on(self, platform: Platform) -> bool:
        """
        Verifica si la tool está disponible en la plataforma actual y habilitada.

        Args:
            platform: Plataforma a verificar.

        Returns:
            True si la tool está habilitada y soportada en la plataforma.
        """
        return self.enabled and self.capability.is_supported_on(platform)


# =============================================================================
# PUERTO ABSTRACTO DE TOOL
# =============================================================================


class ToolPort(abc.ABC):
    """
    Puerto abstracto que define el contrato que toda tool del sistema debe implementar.

    Cada tool concreta del sistema (WebSearchTool, DocumentReadTool,
    AppLauncherTool, etc.) hereda de ToolPort e implementa sus métodos abstractos.

    El ToolDispatch invoca las tools únicamente a través de este puerto —
    nunca conoce la implementación concreta. Esto permite añadir nuevas tools
    sin tocar el coordinator, planner ni executor.

    Contratos invariantes que toda implementación DEBE respetar:
      1. schema() SIEMPRE retorna el mismo ToolSchema durante la vida de la
         instancia. El schema no cambia en runtime.
      2. validate_input() DEBE ser llamado antes de execute() por el ToolDispatch.
         El execute() puede asumir que el input ya fue validado.
      3. execute() NUNCA lanza excepciones de Python puras — las convierte en
         subclases de ForgeToolError antes de propagarlas al ToolDispatch.
      4. Si la tool implementa rollback(), debe ser idempotente y no lanzar
         excepciones (solo loguear si falla).
      5. El timeout se gestiona externamente por el ToolDispatch — la tool no
         necesita implementar timeouts internos salvo para operaciones de red
         propias.
    """

    @abc.abstractmethod
    def schema(self) -> ToolSchema:
        """
        Retorna el schema completo de la tool.

        El schema es inmutable y se llama una sola vez durante el registro
        en el ToolRegistry. Los campos de ToolSchema son la fuente de verdad
        para la integración con el LLM y el PolicyEngine.

        Returns:
            ToolSchema con toda la metadata de la tool.
        """

    @abc.abstractmethod
    async def execute(self, input: ToolInput) -> ToolOutput:
        """
        Ejecuta la operación de la tool con el input proporcionado.

        El ToolDispatch llama a este método después de:
          1. Validar el input con validate_input().
          2. Verificar permisos con el PolicyEngine.
          3. Obtener approval del usuario si es necesario.
          4. Preparar el sandbox si es necesario.

        La implementación puede asumir que el input es válido (ya fue validado)
        y que tiene autorización para ejecutarse (ya fue aprobada).

        Args:
            input: ToolInput validado con los argumentos del LLM.

        Returns:
            ToolOutput con el resultado de la operación.

        Raises:
            ToolExecutionError:   Si ocurre un error durante la ejecución.
            ToolTimeoutError:     Si la operación excede el timeout.
            ToolPermissionError:  Si el OS rechaza la operación por permisos.
        """

    @abc.abstractmethod
    async def validate_input(self, input: ToolInput) -> list[str]:
        """
        Valida que el input cumpla con el schema y las restricciones de la tool.

        Va más allá de la validación JSON Schema — verifica restricciones
        semánticas específicas de la tool (ej: que una URL sea de un protocolo
        válido, que un path no tenga caracteres inválidos para el OS actual).

        Args:
            input: ToolInput a validar.

        Returns:
            Lista de mensajes de error de validación. Lista vacía si el input
            es válido. El ToolDispatch rechaza la ejecución si la lista no está vacía.
        """

    async def rollback(self, input: ToolInput, output: ToolOutput) -> None:
        """
        Deshace el efecto de una ejecución previa (cuando has_rollback=True).

        Solo las tools que declaran has_rollback=True en su ToolCapability
        necesitan implementar este método. Para las demás, la implementación
        por defecto es un no-op.

        El rollback NUNCA debe lanzar excepciones — si falla, solo loguea el error.
        El sistema no puede garantizar que el rollback sea siempre posible.

        Args:
            input:  El ToolInput original de la ejecución a deshacer.
            output: El ToolOutput producido por la ejecución a deshacer.
        """
        # No-op por defecto. Las tools que soportan rollback lo sobreescriben.

    @property
    def tool_id(self) -> str:
        """Atajo para acceder al tool_id sin llamar a schema()."""
        return self.schema().tool_id

    @property
    def category(self) -> ToolCategory:
        """Atajo para acceder a la categoría sin llamar a schema()."""
        return self.schema().category

    @property
    def risk_level(self) -> RiskLevel:
        """Atajo para acceder al nivel de riesgo sin llamar a schema()."""
        return self.schema().capability.risk_classification