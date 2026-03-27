"""
Tipos de error base del ecosistema Forge Platform.

Este módulo define la jerarquía completa de excepciones que comparten todos los
productos Forge (HiperForge User, HiperForge Developer, futuros productos). Cada
excepción lleva consigo metadatos estructurados suficientes para:

  - Logging estructurado sin pérdida de contexto
  - Clasificación automática por severidad y categoría
  - Recovery hints para que las capas superiores puedan tomar decisiones informadas
  - Serialización a formatos de auditoría y observabilidad
  - Traducción a mensajes UX-friendly sin exponer internals

Principios de diseño:
  1. Fail-loud en desarrollo, fail-safe en producción.
  2. Toda excepción tiene un código único, categoría y severidad.
  3. Los recovery hints son instrucciones para el sistema, no para el usuario.
  4. El contexto adicional se pasa como dict tipado para evitar pérdida de info.
  5. Ninguna excepción de este módulo importa nada de la infraestructura.

Jerarquía:
  ForgeError (base)
  ├── ForgeConfigurationError      — problemas de configuración del sistema
  ├── ForgeLLMError                — errores de integración con providers LLM
  │   ├── LLMConnectionError
  │   ├── LLMRateLimitError
  │   ├── LLMTokenLimitError
  │   ├── LLMContentFilterError
  │   ├── LLMTimeoutError
  │   └── LLMProviderError
  ├── ForgeToolError               — errores en ejecución de tools
  │   ├── ToolInputValidationError
  │   ├── ToolExecutionError
  │   ├── ToolTimeoutError
  │   ├── ToolPermissionError
  │   └── ToolNotAvailableError
  ├── ForgePolicyError             — violaciones del policy engine
  │   ├── PolicyDeniedError
  │   ├── PolicyEvaluationError
  │   └── ApprovalRequiredError
  ├── ForgeStorageError            — errores de persistencia
  │   ├── StorageConnectionError
  │   ├── StorageNotFoundError
  │   ├── StorageCorruptionError
  │   └── StorageMigrationError
  ├── ForgeSecurityError           — violaciones de seguridad
  │   ├── PathTraversalError
  │   ├── SandboxError
  │   └── InputSanitizationError
  └── ForgeObservabilityError      — errores internos de observabilidad
"""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Self


# =============================================================================
# ENUMERACIONES DE CLASIFICACIÓN
# =============================================================================


class ErrorCategory(Enum):
    """
    Categoría funcional del error. Usada para routing en observabilidad
    y para determinar el comportamiento de retry/fallback en la capa de aplicación.
    """

    CONFIGURATION = "configuration"
    """Error en la configuración del sistema. Generalmente no recuperable sin intervención."""

    LLM = "llm"
    """Error en la integración con un proveedor de LLM."""

    TOOL = "tool"
    """Error durante la ejecución de una tool del agente."""

    POLICY = "policy"
    """Violación de una política de seguridad o permisos."""

    STORAGE = "storage"
    """Error en la capa de persistencia local."""

    SECURITY = "security"
    """Violación de restricción de seguridad. Nunca se hace retry automático."""

    OBSERVABILITY = "observability"
    """Error interno del sistema de logging/tracing. No bloquea el flujo principal."""

    DOMAIN = "domain"
    """Violación de una invariante de dominio. Indica un bug en la lógica de negocio."""

    UNKNOWN = "unknown"
    """Categoría de fallback para errores no clasificados."""


class ErrorSeverity(Enum):
    """
    Nivel de severidad del error. Determina el nivel de log y las alertas generadas.

    Mapeo a niveles de log estándar:
      DEBUG    → no aplica (no son errores)
      INFO     → no aplica
      WARNING  → LOW
      ERROR    → MEDIUM, HIGH
      CRITICAL → CRITICAL
    """

    LOW = auto()
    """
    El sistema puede continuar sin intervención. El resultado puede ser degradado.
    Ejemplo: un LLM rate limit con retry disponible.
    """

    MEDIUM = auto()
    """
    La operación actual falló pero el sistema puede continuar con otras operaciones.
    Ejemplo: una tool falla, la tarea se cancela pero la sesión continúa.
    """

    HIGH = auto()
    """
    Un componente crítico está fallando. Requiere atención pero no es catastrófico.
    Ejemplo: la conexión a SQLite falla, las sesiones no se pueden persistir.
    """

    CRITICAL = auto()
    """
    El sistema no puede operar correctamente. Requiere intervención inmediata.
    Ejemplo: corrupción detectada en el storage, violación de sandbox.
    """


class RecoveryHint(Enum):
    """
    Sugerencia de acción de recuperación para la capa que captura el error.
    Estas son instrucciones para el sistema, no mensajes para el usuario.
    """

    RETRY_IMMEDIATELY = auto()
    """Reintentar la operación de inmediato. El error es probablemente transitorio."""

    RETRY_WITH_BACKOFF = auto()
    """Reintentar con espera exponencial. Rate limits, timeouts transitorios."""

    RETRY_WITH_FALLBACK = auto()
    """Reintentar usando un provider/estrategia alternativa."""

    DEGRADE_GRACEFULLY = auto()
    """
    Continuar con funcionalidad reducida. Ejemplo: sin memoria a largo plazo
    si el storage falla, pero la sesión actual continúa.
    """

    ABORT_TASK = auto()
    """Cancelar la tarea actual. El error es irrecuperable para esta operación."""

    ABORT_SESSION = auto()
    """Terminar la sesión. El error compromete el estado de la sesión completa."""

    REQUIRE_USER_ACTION = auto()
    """El usuario debe tomar una acción (configurar API key, otorgar permiso, etc.)."""

    DO_NOT_RETRY = auto()
    """No reintentar bajo ninguna circunstancia. Usado para violaciones de seguridad."""

    LOG_AND_CONTINUE = auto()
    """Registrar el error y continuar. Usado para errores de observabilidad."""


# =============================================================================
# CLASE BASE
# =============================================================================


class ForgeError(Exception):
    """
    Clase base de todos los errores del ecosistema Forge Platform.

    Extiende Exception con metadatos estructurados que permiten al sistema
    tomar decisiones automáticas de recovery y generar observabilidad de alta
    calidad sin necesidad de parsing de mensajes de texto.

    Atributos:
        code:           Código único del error en formato SNAKE_UPPER_CASE.
                        Permite identificar el error sin depender del mensaje.
        message:        Descripción técnica del error para logs y debugging.
        category:       Clasificación funcional del error.
        severity:       Nivel de severidad para alertas y log levels.
        recovery_hint:  Sugerencia de acción de recuperación para el sistema.
        context:        Datos adicionales relevantes para debugging (sanitizados).
        cause:          Excepción original que causó este error (chaining).
        timestamp:      Momento exacto de creación del error (UTC).
        traceback_str:  Stack trace capturado en el momento de creación.
    """

    # Valores por defecto — las subclases los sobreescriben como atributos de clase
    _default_code: str = "FORGE_ERROR"
    _default_category: ErrorCategory = ErrorCategory.UNKNOWN
    _default_severity: ErrorSeverity = ErrorSeverity.MEDIUM
    _default_recovery_hint: RecoveryHint = RecoveryHint.ABORT_TASK

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        category: ErrorCategory | None = None,
        severity: ErrorSeverity | None = None,
        recovery_hint: RecoveryHint | None = None,
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        """
        Inicializa un ForgeError con metadatos completos.

        Args:
            message:       Descripción técnica del error.
            code:          Código único del error. Si no se provee, usa _default_code.
            category:      Categoría funcional. Si no se provee, usa _default_category.
            severity:      Nivel de severidad. Si no se provee, usa _default_severity.
            recovery_hint: Hint de recuperación. Si no se provee, usa _default_recovery_hint.
            context:       Diccionario con datos adicionales para debugging.
                           NUNCA debe contener secrets, passwords, ni API keys.
            cause:         Excepción original para chaining. Se establece también
                           en __cause__ para compatibilidad con el protocolo estándar.
        """
        super().__init__(message)

        self.code: str = code or self._default_code
        self.message: str = message
        self.category: ErrorCategory = category or self._default_category
        self.severity: ErrorSeverity = severity or self._default_severity
        self.recovery_hint: RecoveryHint = recovery_hint or self._default_recovery_hint
        self.context: dict[str, Any] = context or {}
        self.cause: BaseException | None = cause
        self.timestamp: datetime = datetime.now(tz=timezone.utc)
        self.traceback_str: str = traceback.format_exc()

        # Chaining estándar de Python para compatibilidad con raise ... from ...
        if cause is not None:
            self.__cause__ = cause

    def to_dict(self) -> dict[str, Any]:
        """
        Serializa el error a un diccionario apto para logging estructurado y auditoría.

        El formato resultante es compatible con structlog, OpenTelemetry y cualquier
        sistema de observabilidad que consuma dicts. Los secrets nunca deben llegar
        al contexto, pero se aplica una capa adicional de sanitización por defecto.

        Returns:
            Diccionario con todos los metadatos del error, serializable a JSON.
        """
        result: dict[str, Any] = {
            "error_code": self.code,
            "error_message": self.message,
            "error_category": self.category.value,
            "error_severity": self.severity.name,
            "error_recovery_hint": self.recovery_hint.name,
            "error_timestamp": self.timestamp.isoformat(),
            "error_context": self._sanitize_context(self.context),
        }

        if self.cause is not None:
            result["error_cause"] = {
                "type": type(self.cause).__name__,
                "message": str(self.cause),
            }

        return result

    def with_context(self, **kwargs: Any) -> Self:
        """
        Añade contexto adicional al error de forma fluida (builder pattern).

        Útil para agregar información de contexto a medida que el error sube
        por la pila de llamadas sin necesidad de crear nuevas instancias.

        Args:
            **kwargs: Pares clave-valor a agregar al contexto del error.

        Returns:
            La misma instancia del error con el contexto extendido.

        Example:
            raise LLMTimeoutError("Timeout").with_context(
                provider="openai",
                model="gpt-4o",
                timeout_seconds=30,
            )
        """
        self.context.update(kwargs)
        return self

    def is_retryable(self) -> bool:
        """
        Indica si el error puede ser objeto de un reintento automático.

        Returns:
            True si el recovery hint indica que se puede reintentar.
        """
        return self.recovery_hint in {
            RecoveryHint.RETRY_IMMEDIATELY,
            RecoveryHint.RETRY_WITH_BACKOFF,
            RecoveryHint.RETRY_WITH_FALLBACK,
        }

    def is_security_violation(self) -> bool:
        """
        Indica si el error representa una violación de seguridad.

        Los errores de seguridad nunca se reintenta automáticamente y siempre
        se registran con nivel CRITICAL en el audit log.

        Returns:
            True si la categoría es SECURITY o POLICY.
        """
        return self.category in {ErrorCategory.SECURITY, ErrorCategory.POLICY}

    @staticmethod
    def _sanitize_context(context: dict[str, Any]) -> dict[str, Any]:
        """
        Elimina del contexto cualquier clave que pueda contener información sensible.

        Esta es una segunda línea de defensa. La primera es que el código nunca
        debería poner secrets en el contexto. Si una clave sospechosa llega aquí,
        se reemplaza por un marcador '[REDACTED]'.

        Args:
            context: Diccionario de contexto a sanitizar.

        Returns:
            Diccionario sanitizado, seguro para logging.
        """
        sensitive_patterns = {
            "key", "secret", "password", "passwd", "token",
            "credential", "auth", "bearer", "api_key", "apikey",
            "private", "ssn", "credit_card", "card_number",
        }

        sanitized: dict[str, Any] = {}
        for k, v in context.items():
            k_lower = k.lower()
            if any(pattern in k_lower for pattern in sensitive_patterns):
                sanitized[k] = "[REDACTED]"
            elif isinstance(v, dict):
                sanitized[k] = ForgeError._sanitize_context(v)
            else:
                sanitized[k] = v

        return sanitized

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"code={self.code!r}, "
            f"severity={self.severity.name}, "
            f"message={self.message!r}"
            f")"
        )

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


# =============================================================================
# ERRORES DE CONFIGURACIÓN
# =============================================================================


class ForgeConfigurationError(ForgeError):
    """
    Error en la configuración del sistema Forge.

    Se lanza cuando la configuración es inválida, incompleta, o inconsistente.
    Generalmente ocurre al arrancar la aplicación y no es recuperable sin
    intervención del usuario o del administrador del sistema.
    """

    _default_code = "FORGE_CONFIG_ERROR"
    _default_category = ErrorCategory.CONFIGURATION
    _default_severity = ErrorSeverity.HIGH
    _default_recovery_hint = RecoveryHint.REQUIRE_USER_ACTION


class MissingConfigurationError(ForgeConfigurationError):
    """
    Una clave de configuración requerida no está presente en el entorno.

    Ejemplo: la API key de OpenAI no está definida y no hay modelo local configurado.
    """

    _default_code = "CONFIG_MISSING_KEY"

    def __init__(self, key: str, *, hint: str = "") -> None:
        """
        Args:
            key:  Nombre de la clave de configuración faltante.
            hint: Sugerencia de cómo obtener o configurar el valor.
        """
        msg = f"Clave de configuración requerida no encontrada: '{key}'"
        if hint:
            msg = f"{msg}. {hint}"
        super().__init__(msg, context={"missing_key": key, "user_hint": hint})


class InvalidConfigurationError(ForgeConfigurationError):
    """
    Una clave de configuración tiene un valor inválido o fuera de rango.

    Ejemplo: max_tokens configurado como string en lugar de entero.
    """

    _default_code = "CONFIG_INVALID_VALUE"

    def __init__(self, key: str, value: Any, *, reason: str) -> None:
        """
        Args:
            key:    Nombre de la clave de configuración inválida.
            value:  Valor problemático (se sanitizará antes de logging).
            reason: Explicación de por qué el valor es inválido.
        """
        super().__init__(
            f"Valor inválido para configuración '{key}': {reason}",
            context={"config_key": key, "invalid_value": str(value), "reason": reason},
        )


# =============================================================================
# ERRORES DE LLM
# =============================================================================


class ForgeLLMError(ForgeError):
    """
    Error base para todos los problemas de integración con providers LLM.

    Proporciona contexto adicional sobre el provider y modelo involucrado,
    que es crítico para debugging y para el sistema de fallback multi-provider.
    """

    _default_code = "LLM_ERROR"
    _default_category = ErrorCategory.LLM
    _default_severity = ErrorSeverity.MEDIUM
    _default_recovery_hint = RecoveryHint.RETRY_WITH_BACKOFF

    def __init__(
        self,
        message: str,
        *,
        provider: str = "unknown",
        model: str = "unknown",
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        context.update({"llm_provider": provider, "llm_model": model})
        super().__init__(message, context=context, **kwargs)
        self.provider = provider
        self.model = model


class LLMConnectionError(ForgeLLMError):
    """
    No se puede establecer conexión con el endpoint del provider LLM.

    Causas comunes: sin acceso a internet, endpoint caído, firewall corporativo.
    Recovery: retry con backoff, fallback a modelo local si está configurado.
    """

    _default_code = "LLM_CONNECTION_ERROR"
    _default_severity = ErrorSeverity.HIGH
    _default_recovery_hint = RecoveryHint.RETRY_WITH_FALLBACK


class LLMRateLimitError(ForgeLLMError):
    """
    El provider LLM ha rechazado la request por rate limiting.

    El campo retry_after_seconds indica cuándo se puede reintentar.
    Recovery: esperar el tiempo indicado y reintentar con backoff exponencial.
    """

    _default_code = "LLM_RATE_LIMIT"
    _default_severity = ErrorSeverity.LOW
    _default_recovery_hint = RecoveryHint.RETRY_WITH_BACKOFF

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if retry_after_seconds is not None:
            context["retry_after_seconds"] = retry_after_seconds
        super().__init__(message, context=context, **kwargs)
        self.retry_after_seconds = retry_after_seconds


class LLMTokenLimitError(ForgeLLMError):
    """
    El contexto enviado al LLM excede el límite de tokens del modelo.

    Esto indica un bug en ContextBuilder: debería haber truncado el contexto
    antes de enviarlo. Recovery: truncar el contexto y reintentar.
    """

    _default_code = "LLM_TOKEN_LIMIT"
    _default_severity = ErrorSeverity.MEDIUM
    _default_recovery_hint = RecoveryHint.RETRY_IMMEDIATELY

    def __init__(
        self,
        message: str,
        *,
        tokens_sent: int | None = None,
        tokens_limit: int | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if tokens_sent is not None:
            context["tokens_sent"] = tokens_sent
        if tokens_limit is not None:
            context["tokens_limit"] = tokens_limit
        super().__init__(message, context=context, **kwargs)
        self.tokens_sent = tokens_sent
        self.tokens_limit = tokens_limit


class LLMContentFilterError(ForgeLLMError):
    """
    El provider LLM rechazó la request por violación de sus content policies.

    Recovery: no reintentar con el mismo contenido. Notificar al usuario
    que la solicitud no puede procesarse.
    """

    _default_code = "LLM_CONTENT_FILTER"
    _default_severity = ErrorSeverity.MEDIUM
    _default_recovery_hint = RecoveryHint.ABORT_TASK


class LLMTimeoutError(ForgeLLMError):
    """
    El LLM no respondió dentro del tiempo límite configurado.

    Recovery: retry con backoff. Si persiste, considerar degradar a modelo más rápido.
    """

    _default_code = "LLM_TIMEOUT"
    _default_severity = ErrorSeverity.MEDIUM
    _default_recovery_hint = RecoveryHint.RETRY_WITH_BACKOFF

    def __init__(
        self,
        message: str,
        *,
        timeout_seconds: float | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if timeout_seconds is not None:
            context["timeout_seconds"] = timeout_seconds
        super().__init__(message, context=context, **kwargs)
        self.timeout_seconds = timeout_seconds


class LLMProviderError(ForgeLLMError):
    """
    El provider LLM retornó un error interno (HTTP 5xx u equivalente).

    Recovery: retry con backoff exponencial, fallback a otro provider.
    """

    _default_code = "LLM_PROVIDER_ERROR"
    _default_severity = ErrorSeverity.HIGH
    _default_recovery_hint = RecoveryHint.RETRY_WITH_FALLBACK

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if status_code is not None:
            context["http_status_code"] = status_code
        super().__init__(message, context=context, **kwargs)
        self.status_code = status_code


# =============================================================================
# ERRORES DE TOOLS
# =============================================================================


class ForgeToolError(ForgeError):
    """
    Error base para todos los problemas durante la ejecución de tools del agente.

    Incluye el tool_id como metadato crítico para audit logging y debugging.
    """

    _default_code = "TOOL_ERROR"
    _default_category = ErrorCategory.TOOL
    _default_severity = ErrorSeverity.MEDIUM
    _default_recovery_hint = RecoveryHint.ABORT_TASK

    def __init__(
        self,
        message: str,
        *,
        tool_id: str = "unknown",
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        context["tool_id"] = tool_id
        super().__init__(message, context=context, **kwargs)
        self.tool_id = tool_id


class ToolInputValidationError(ForgeToolError):
    """
    El input proporcionado a una tool no cumple con su schema de validación.

    Esto indica un problema en la respuesta del LLM o en el tool dispatch.
    Recovery: no reintentar con el mismo input; reportar al sistema de feedback.
    """

    _default_code = "TOOL_INPUT_INVALID"
    _default_severity = ErrorSeverity.MEDIUM
    _default_recovery_hint = RecoveryHint.ABORT_TASK

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        validation_errors: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if field:
            context["invalid_field"] = field
        if validation_errors:
            context["validation_errors"] = validation_errors
        super().__init__(message, context=context, **kwargs)


class ToolExecutionError(ForgeToolError):
    """
    La tool lanzó un error durante su ejecución.

    La causa original (cause) debe siempre pasarse para mantener el stack trace
    completo en el sistema de observabilidad.
    """

    _default_code = "TOOL_EXECUTION_ERROR"
    _default_severity = ErrorSeverity.MEDIUM
    _default_recovery_hint = RecoveryHint.ABORT_TASK


class ToolTimeoutError(ForgeToolError):
    """
    La tool no completó su ejecución dentro del timeout configurado.

    Recovery: si la tool es idempotente, se puede reintentar. Si no, abortar.
    """

    _default_code = "TOOL_TIMEOUT"
    _default_severity = ErrorSeverity.MEDIUM
    _default_recovery_hint = RecoveryHint.RETRY_WITH_BACKOFF

    def __init__(
        self,
        message: str,
        *,
        timeout_seconds: float | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if timeout_seconds is not None:
            context["timeout_seconds"] = timeout_seconds
        super().__init__(message, context=context, **kwargs)
        self.timeout_seconds = timeout_seconds


class ToolPermissionError(ForgeToolError):
    """
    El sistema operativo rechazó la operación por permisos insuficientes.

    Recovery: notificar al usuario que debe otorgar permisos adicionales.
    """

    _default_code = "TOOL_PERMISSION_DENIED"
    _default_severity = ErrorSeverity.MEDIUM
    _default_recovery_hint = RecoveryHint.REQUIRE_USER_ACTION

    def __init__(
        self,
        message: str,
        *,
        resource: str | None = None,
        required_permission: str | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if resource:
            context["resource"] = resource
        if required_permission:
            context["required_permission"] = required_permission
        super().__init__(message, context=context, **kwargs)


class ToolNotAvailableError(ForgeToolError):
    """
    La tool solicitada no está disponible en la plataforma actual.

    Ejemplo: una tool de automatización de desktop no disponible en modo headless.
    Recovery: degradar a una alternativa o informar al usuario.
    """

    _default_code = "TOOL_NOT_AVAILABLE"
    _default_severity = ErrorSeverity.MEDIUM
    _default_recovery_hint = RecoveryHint.DEGRADE_GRACEFULLY

    def __init__(
        self,
        message: str,
        *,
        platform: str | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if platform:
            context["current_platform"] = platform
        super().__init__(message, context=context, **kwargs)


# =============================================================================
# ERRORES DE POLICY
# =============================================================================


class ForgePolicyError(ForgeError):
    """
    Error base para todos los problemas relacionados con el policy engine.

    Los errores de policy son especialmente críticos porque representan el
    boundary de seguridad del sistema. Siempre se registran en el audit log.
    """

    _default_code = "POLICY_ERROR"
    _default_category = ErrorCategory.POLICY
    _default_severity = ErrorSeverity.HIGH
    _default_recovery_hint = RecoveryHint.DO_NOT_RETRY


class PolicyDeniedError(ForgePolicyError):
    """
    El policy engine denegó explícitamente una acción propuesta.

    Esta es la ruta normal cuando un agente intenta hacer algo no permitido.
    No es un error del sistema — es el sistema funcionando correctamente.
    El mensaje de denegación debe ser informativo pero no revelar detalles
    de implementación de las policies.
    """

    _default_code = "POLICY_DENIED"
    _default_severity = ErrorSeverity.MEDIUM
    _default_recovery_hint = RecoveryHint.ABORT_TASK

    def __init__(
        self,
        message: str,
        *,
        policy_name: str | None = None,
        action_attempted: str | None = None,
        risk_level: str | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if policy_name:
            context["policy_name"] = policy_name
        if action_attempted:
            context["action_attempted"] = action_attempted
        if risk_level:
            context["risk_level"] = risk_level
        super().__init__(message, context=context, **kwargs)
        self.policy_name = policy_name
        self.action_attempted = action_attempted


class PolicyEvaluationError(ForgePolicyError):
    """
    El policy engine encontró un error interno al evaluar una policy.

    COMPORTAMIENTO FAIL-CLOSED: cuando el evaluador falla, el sistema
    deniega la acción automáticamente. Nunca se permite por defecto.
    Este error indica un bug en el sistema de policies.
    """

    _default_code = "POLICY_EVALUATION_ERROR"
    _default_severity = ErrorSeverity.CRITICAL
    _default_recovery_hint = RecoveryHint.ABORT_TASK


class ApprovalRequiredError(ForgePolicyError):
    """
    La acción requiere aprobación explícita del usuario antes de ejecutarse.

    Este no es estrictamente un error — es una señal de control de flujo.
    El executor lo captura y activa el ApprovalWorkflow.
    """

    _default_code = "POLICY_APPROVAL_REQUIRED"
    _default_severity = ErrorSeverity.LOW
    _default_recovery_hint = RecoveryHint.REQUIRE_USER_ACTION

    def __init__(
        self,
        message: str,
        *,
        action_description: str,
        risk_level: str,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        context["action_description"] = action_description
        context["risk_level"] = risk_level
        super().__init__(message, context=context, **kwargs)
        self.action_description = action_description
        self.risk_level = risk_level


# =============================================================================
# ERRORES DE STORAGE
# =============================================================================


class ForgeStorageError(ForgeError):
    """
    Error base para todos los problemas de persistencia local.

    Incluye el store_name para identificar qué almacenamiento falló
    (sessions, artifacts, audit, memory).
    """

    _default_code = "STORAGE_ERROR"
    _default_category = ErrorCategory.STORAGE
    _default_severity = ErrorSeverity.HIGH
    _default_recovery_hint = RecoveryHint.RETRY_WITH_BACKOFF

    def __init__(
        self,
        message: str,
        *,
        store_name: str = "unknown",
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        context["store_name"] = store_name
        super().__init__(message, context=context, **kwargs)
        self.store_name = store_name


class StorageConnectionError(ForgeStorageError):
    """
    No se puede conectar al backend de storage (SQLite, filesystem).

    Recovery: retry con backoff. Si persiste, degradar gracefully
    (sesión continúa sin persistencia — datos se perderán al cerrar).
    """

    _default_code = "STORAGE_CONNECTION_ERROR"
    _default_recovery_hint = RecoveryHint.DEGRADE_GRACEFULLY


class StorageNotFoundError(ForgeStorageError):
    """
    El recurso solicitado no existe en el storage.

    Esto es frecuentemente un estado válido (la sesión no existe todavía),
    no un error crítico. La severidad es LOW.
    """

    _default_code = "STORAGE_NOT_FOUND"
    _default_severity = ErrorSeverity.LOW
    _default_recovery_hint = RecoveryHint.ABORT_TASK

    def __init__(
        self,
        message: str,
        *,
        resource_type: str | None = None,
        resource_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if resource_type:
            context["resource_type"] = resource_type
        if resource_id:
            context["resource_id"] = resource_id
        super().__init__(message, context=context, **kwargs)


class StorageCorruptionError(ForgeStorageError):
    """
    Se detectó corrupción en los datos persistidos.

    Esto es CRÍTICO. El sistema intentará recuperación automática si es posible.
    Si no, se iniciará con storage limpio y se notificará al usuario.
    """

    _default_code = "STORAGE_CORRUPTION"
    _default_severity = ErrorSeverity.CRITICAL
    _default_recovery_hint = RecoveryHint.REQUIRE_USER_ACTION


class StorageMigrationError(ForgeStorageError):
    """
    Una migración de esquema de base de datos falló.

    Ocurre al actualizar la aplicación cuando el esquema cambia.
    Recovery: en la mayoría de casos se puede hacer rollback de la migración.
    """

    _default_code = "STORAGE_MIGRATION_ERROR"
    _default_severity = ErrorSeverity.CRITICAL
    _default_recovery_hint = RecoveryHint.REQUIRE_USER_ACTION

    def __init__(
        self,
        message: str,
        *,
        migration_version: str | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if migration_version:
            context["migration_version"] = migration_version
        super().__init__(message, context=context, **kwargs)


# =============================================================================
# ERRORES DE SEGURIDAD
# =============================================================================


class ForgeSecurityError(ForgeError):
    """
    Error base para todos los problemas de seguridad del sistema.

    Los errores de seguridad tienen prioridad máxima en el audit log
    y NUNCA se reintenta automáticamente. Se registran con nivel CRITICAL.
    """

    _default_code = "SECURITY_ERROR"
    _default_category = ErrorCategory.SECURITY
    _default_severity = ErrorSeverity.CRITICAL
    _default_recovery_hint = RecoveryHint.DO_NOT_RETRY


class PathTraversalError(ForgeSecurityError):
    """
    Se detectó un intento de acceso a una ruta fuera de los directorios permitidos.

    Puede ser un intento malicioso o un bug en la generación del path por el LLM.
    En ambos casos, se bloquea inmediatamente y se registra en el audit log.
    """

    _default_code = "SECURITY_PATH_TRAVERSAL"

    def __init__(
        self,
        message: str,
        *,
        attempted_path: str,
        allowed_base: str | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        context["attempted_path"] = attempted_path
        if allowed_base:
            context["allowed_base"] = allowed_base
        super().__init__(message, context=context, **kwargs)
        self.attempted_path = attempted_path


class SandboxError(ForgeSecurityError):
    """
    Error en el proceso sandbox de automatización de desktop.

    Puede indicar un fallo en el aislamiento (sandbox escape attempt)
    o un error interno del sandbox process. Ambos son CRÍTICOS.
    """

    _default_code = "SECURITY_SANDBOX_ERROR"

    def __init__(
        self,
        message: str,
        *,
        sandbox_pid: int | None = None,
        exit_code: int | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if sandbox_pid is not None:
            context["sandbox_pid"] = sandbox_pid
        if exit_code is not None:
            context["sandbox_exit_code"] = exit_code
        super().__init__(message, context=context, **kwargs)


class InputSanitizationError(ForgeSecurityError):
    """
    El input recibido contiene patrones potencialmente maliciosos.

    El sanitizador bloqueó el input antes de que llegara a la tool.
    Se registra para análisis posterior de patrones de ataque.
    """

    _default_code = "SECURITY_INPUT_SANITIZATION"
    _default_severity = ErrorSeverity.HIGH

    def __init__(
        self,
        message: str,
        *,
        pattern_detected: str | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if pattern_detected:
            context["pattern_detected"] = pattern_detected
        # NUNCA incluir el input original en el contexto — puede contener el payload
        super().__init__(message, context=context, **kwargs)


# =============================================================================
# ERRORES DE OBSERVABILIDAD
# =============================================================================


class ForgeObservabilityError(ForgeError):
    """
    Error en el sistema de observabilidad interno.

    Estos errores son secundarios — nunca bloquean el flujo principal.
    Se registran con LOG_AND_CONTINUE para no afectar la experiencia del usuario.
    """

    _default_code = "OBSERVABILITY_ERROR"
    _default_category = ErrorCategory.OBSERVABILITY
    _default_severity = ErrorSeverity.LOW
    _default_recovery_hint = RecoveryHint.LOG_AND_CONTINUE


# =============================================================================
# ERRORES DE DOMINIO
# =============================================================================


class ForgeDomainError(ForgeError):
    """
    Error base para violaciones de invariantes de dominio.

    Se lanza cuando el código de dominio detecta un estado imposible o
    una transición de estado inválida. Indica un bug en la lógica de negocio,
    no un error de infraestructura.
    """

    _default_code = "DOMAIN_ERROR"
    _default_category = ErrorCategory.DOMAIN
    _default_severity = ErrorSeverity.HIGH
    _default_recovery_hint = RecoveryHint.ABORT_TASK


class InvalidStateTransitionError(ForgeDomainError):
    """
    Se intentó una transición de estado inválida en una entidad de dominio.

    Ejemplo: transicionar una Task de COMPLETED a EXECUTING.
    """

    _default_code = "DOMAIN_INVALID_STATE_TRANSITION"

    def __init__(
        self,
        entity: str,
        from_state: str,
        to_state: str,
        *,
        entity_id: str | None = None,
    ) -> None:
        msg = f"Transición de estado inválida en {entity}: {from_state!r} → {to_state!r}"
        context: dict[str, Any] = {
            "entity_type": entity,
            "from_state": from_state,
            "to_state": to_state,
        }
        if entity_id:
            context["entity_id"] = entity_id
        super().__init__(msg, context=context)
        self.entity = entity
        self.from_state = from_state
        self.to_state = to_state


class DomainInvariantError(ForgeDomainError):
    """
    Una invariante de dominio fue violada.

    Ejemplo: un Plan con más de MAX_STEPS pasos, un TokenBudget negativo.
    """

    _default_code = "DOMAIN_INVARIANT_VIOLATION"

    def __init__(self, invariant: str, *, detail: str = "") -> None:
        msg = f"Invariante de dominio violada: {invariant}"
        if detail:
            msg = f"{msg}. Detalle: {detail}"
        super().__init__(msg, context={"invariant": invariant, "detail": detail})


# =============================================================================
# UTILIDADES
# =============================================================================


def wrap_external_error(
    error: BaseException,
    *,
    forge_error_type: type[ForgeError],
    message: str,
    **kwargs: Any,
) -> ForgeError:
    """
    Envuelve una excepción externa (de librería de terceros) en un ForgeError tipado.

    Esta función es el punto estándar para convertir excepciones de infraestructura
    (httpx.ConnectError, sqlite3.Error, etc.) en errores del dominio Forge.
    Preserva el stack trace original a través del mecanismo de chaining de Python.

    Args:
        error:            La excepción original a envolver.
        forge_error_type: El tipo de ForgeError con el que envolver.
        message:          Mensaje descriptivo del error en contexto Forge.
        **kwargs:         Argumentos adicionales para el constructor del error Forge.

    Returns:
        Una instancia del forge_error_type especificado con la causa original encadenada.

    Example:
        try:
            await client.get(url)
        except httpx.ConnectError as e:
            raise wrap_external_error(
                e,
                forge_error_type=LLMConnectionError,
                message="No se puede conectar a OpenAI API",
                provider="openai",
            ) from e
    """
    return forge_error_type(message, cause=error, **kwargs)