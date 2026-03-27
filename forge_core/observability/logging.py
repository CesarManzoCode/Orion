"""
Sistema de logging estructurado del ecosistema Forge Platform.

Este módulo configura y expone el sistema de logging basado en structlog,
diseñado para producción desde el día uno. Cada log entry es un evento
estructurado con contexto completo, no una cadena de texto libre.

Características principales:
  - Pipeline de processors configurable: JSON para producción, consola
    renderizada para desarrollo.
  - Context binding por módulo y por request/sesión mediante contextvars.
  - Integración automática con la jerarquía de errores ForgeError.
  - Sanitización de campos sensibles en cada log entry.
  - Correlación de eventos via trace_id y session_id en contexto.
  - Rotación automática de archivos de log con límite de tamaño.
  - Compatibilidad total con el módulo estándar logging de Python para
    capturar logs de librerías de terceros (httpx, aiosqlite, etc.).

Arquitectura del pipeline de processors:

  Entrada: logger.info("evento", campo=valor, ...)
      │
      ▼
  [1] add_log_level          — añade el campo 'level'
  [2] add_timestamp           — añade 'timestamp' en ISO 8601 UTC
  [3] add_caller_info         — añade 'module', 'func', 'lineno'
  [4] inject_forge_context    — añade trace_id, session_id desde contextvars
  [5] sanitize_sensitive      — redacta campos con secrets
  [6] format_forge_errors     — serializa ForgeError si está presente
  [7] Renderer:
        JSON mode   → JSONRenderer   (una línea JSON por evento)
        Console mode→ ConsoleRenderer (output coloreado legible)
      │
      ▼
  Salida: archivo de log + stdout (configurable)

Uso:
    from forge_core.observability.logging import configure_logging, get_logger

    # En el bootstrap (una sola vez):
    configure_logging(config.observability)

    # En cualquier módulo:
    logger = get_logger(__name__)
    logger.info("tarea_iniciada", task_id="task_01", tool="web_search")
    logger.error("tool_fallida", error=exc, tool_id="web_search")

    # Binding de contexto para una sesión:
    from forge_core.observability.logging import bind_context, clear_context
    bind_context(session_id="sess_abc", trace_id="trace_xyz")
    logger.info("mensaje")  # automáticamente incluye session_id y trace_id
    clear_context()
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from forge_core.config.schema import LogFormat, LogLevel, ObservabilityConfig
from forge_core.errors.types import ForgeError


# =============================================================================
# CONTEXTO POR COROUTINE / REQUEST
# =============================================================================

# ContextVar que almacena el contexto compartido de logging para la
# coroutine/request actual. Cada asyncio task tiene su propia copia,
# lo que garantiza aislamiento entre requests concurrentes.
_forge_log_context: ContextVar[dict[str, Any]] = ContextVar(
    "forge_log_context",
    default={},
)


def bind_context(**kwargs: Any) -> None:
    """
    Añade campos al contexto de logging de la coroutine actual.

    Los campos añadidos aparecerán automáticamente en todos los logs
    emitidos desde la misma coroutine/task hasta que se llame clear_context()
    o el contexto sea sobreescrito. Esto es seguro en entornos async con
    múltiples sesiones concurrentes porque usa ContextVar con aislamiento
    por asyncio.Task.

    Args:
        **kwargs: Pares clave-valor a añadir al contexto compartido.

    Example:
        bind_context(
            session_id="sess_abc123",
            trace_id="trace_xyz789",
            user_id="usr_01",
        )
        # A partir de aquí, todos los logs de esta coroutine incluyen esos campos.
    """
    current = _forge_log_context.get().copy()
    current.update(kwargs)
    _forge_log_context.set(current)


def unbind_context(*keys: str) -> None:
    """
    Elimina campos específicos del contexto de logging de la coroutine actual.

    Args:
        *keys: Nombres de los campos a eliminar del contexto.
    """
    current = _forge_log_context.get().copy()
    for key in keys:
        current.pop(key, None)
    _forge_log_context.set(current)


def clear_context() -> None:
    """
    Limpia completamente el contexto de logging de la coroutine actual.

    Debe llamarse al finalizar una sesión o request para evitar que
    datos del contexto anterior contaminen el siguiente.
    """
    _forge_log_context.set({})


def get_current_context() -> dict[str, Any]:
    """
    Retorna una copia del contexto de logging actual.

    Returns:
        Diccionario con los campos de contexto actualmente vinculados.
    """
    return _forge_log_context.get().copy()


# =============================================================================
# PROCESSORS DEL PIPELINE
# =============================================================================


def _processor_inject_forge_context(
    logger: Any,
    method: str,
    event_dict: EventDict,
) -> EventDict:
    """
    Processor: inyecta el contexto Forge (session_id, trace_id, etc.) en cada log.

    Este processor lee el contexto del ContextVar y lo fusiona con el event_dict,
    permitiendo que todos los logs de una sesión estén correlacionados sin que
    cada sitio de llamada tenga que pasar los IDs manualmente.

    Los campos del ContextVar no sobreescriben campos explícitos pasados al logger
    — los campos explícitos tienen precedencia.

    Args:
        logger: Logger de structlog (no usado directamente).
        method: Nombre del método de log ('info', 'error', etc.).
        event_dict: Diccionario de evento actual a enriquecer.

    Returns:
        event_dict enriquecido con el contexto Forge.
    """
    context = _forge_log_context.get()
    for key, value in context.items():
        # Los campos explícitos del caller tienen precedencia sobre el contexto
        if key not in event_dict:
            event_dict[key] = value
    return event_dict


def _processor_sanitize_sensitive(
    logger: Any,
    method: str,
    event_dict: EventDict,
) -> EventDict:
    """
    Processor: redacta campos que pueden contener información sensible.

    Aplica la misma lógica de sanitización que ForgeError._sanitize_context()
    para garantizar que ningún secret llegue a los archivos de log,
    independientemente de qué módulo los haya generado.

    Campos redactados: aquellos cuya clave contiene subcadenas como
    'key', 'secret', 'password', 'token', 'credential', 'auth', 'bearer'.

    Args:
        logger: Logger de structlog.
        method: Método de log.
        event_dict: Diccionario de evento a sanitizar.

    Returns:
        event_dict con campos sensibles redactados como '[REDACTED]'.
    """
    _SENSITIVE_SUBSTRINGS = frozenset({
        "key", "secret", "password", "passwd", "token",
        "credential", "auth", "bearer", "api_key", "apikey",
        "private", "ssn", "credit_card",
    })

    sanitized: EventDict = {}
    for k, v in event_dict.items():
        k_lower = k.lower()
        if any(sub in k_lower for sub in _SENSITIVE_SUBSTRINGS):
            sanitized[k] = "[REDACTED]"
        else:
            sanitized[k] = v

    return sanitized


def _processor_format_forge_errors(
    logger: Any,
    method: str,
    event_dict: EventDict,
) -> EventDict:
    """
    Processor: serializa instancias de ForgeError presentes en el event_dict.

    Si el log contiene un campo 'error' (o 'exc') que es una instancia de
    ForgeError, lo reemplaza por su representación estructurada via to_dict().
    Esto garantiza que los errores de dominio aparezcan como objetos JSON
    estructurados en los logs, no como representaciones de excepción crudas.

    Si el error es una excepción genérica (no ForgeError), se extrae el
    tipo y mensaje sin el stack trace para evitar logs excesivamente verbosos
    (el stack trace está en el campo 'exc_info' si se pasó).

    Args:
        logger: Logger de structlog.
        method: Método de log.
        event_dict: Diccionario de evento.

    Returns:
        event_dict con errores formateados estructuradamente.
    """
    for field_name in ("error", "exc", "exception"):
        error_value = event_dict.get(field_name)

        if isinstance(error_value, ForgeError):
            event_dict[field_name] = error_value.to_dict()
            # Añadir campos de primer nivel para facilitar queries en Loki/ES
            event_dict["error_code"] = error_value.code
            event_dict["error_category"] = error_value.category.value
            event_dict["error_severity"] = error_value.severity.name

        elif isinstance(error_value, BaseException):
            event_dict[field_name] = {
                "type": type(error_value).__name__,
                "message": str(error_value),
            }

    return event_dict


def _processor_add_caller_info(
    logger: Any,
    method: str,
    event_dict: EventDict,
) -> EventDict:
    """
    Processor: añade información del sitio de llamada al log.

    Extrae el módulo, función y número de línea del frame que llamó al logger.
    Es más preciso que el callsite_parameters de structlog porque filtra los
    frames internos de structlog y del pipeline de logging.

    Nota de performance: este processor usa sys._getframe() que es O(n) en
    profundidad de stack. En producción con LOG_LEVEL=INFO o superior, el
    impacto es mínimo porque la mayoría de debug logs no se emiten.

    Args:
        logger: Logger de structlog.
        method: Método de log.
        event_dict: Diccionario de evento.

    Returns:
        event_dict con 'caller_module', 'caller_func', 'caller_line'.
    """
    # Frames internos a filtrar para encontrar el caller real
    _INTERNAL_MODULES = frozenset({
        "structlog",
        "logging",
        "forge_core.observability.logging",
    })

    frame = sys._getframe(1)  # noqa: SLF001
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        if not any(module.startswith(internal) for internal in _INTERNAL_MODULES):
            event_dict["caller_module"] = module
            event_dict["caller_func"] = frame.f_code.co_name
            event_dict["caller_line"] = frame.f_lineno
            break
        frame = frame.f_back  # type: ignore[assignment]

    return event_dict


# =============================================================================
# CONFIGURACIÓN DEL SISTEMA DE LOGGING
# =============================================================================


def _build_processors_pipeline(
    log_format: LogFormat,
    *,
    include_caller_info: bool = True,
) -> list[Processor]:
    """
    Construye el pipeline de processors de structlog según el formato configurado.

    El pipeline es la secuencia de transformaciones que cada log event atraviesa
    antes de ser renderizado. El orden importa: los processors posteriores pueden
    depender de campos añadidos por processors anteriores.

    Args:
        log_format: Formato de salida (JSON o CONSOLE).
        include_caller_info: Si True, incluye módulo/función/línea del caller.
                             False mejora el performance en producción de alto volumen.

    Returns:
        Lista ordenada de processors para el pipeline de structlog.
    """
    # Processors comunes a todos los formatos
    shared_processors: list[Processor] = [
        # Añade el nivel de log ('info', 'error', etc.)
        structlog.stdlib.add_log_level,
        # Añade timestamp ISO 8601 en UTC
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        # Inyecta contexto de sesión/trace desde ContextVar
        _processor_inject_forge_context,
        # Sanitiza campos sensibles
        _processor_sanitize_sensitive,
        # Serializa ForgeError a estructura
        _processor_format_forge_errors,
        # Enriquece con stack trace si se pasó exc_info
        structlog.processors.ExceptionRenderer(),
        # Añade stack_info si se solicitó
        structlog.processors.StackInfoRenderer(),
    ]

    if include_caller_info:
        # Insertar después del timestamp pero antes del contexto
        shared_processors.insert(2, _processor_add_caller_info)

    # Renderer específico por formato
    if log_format == LogFormat.JSON:
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        # Console renderer con colores para desarrollo
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
            sort_keys=True,
        )

    return [*shared_processors, renderer]


def configure_logging(
    config: ObservabilityConfig,
    *,
    force_reconfigure: bool = False,
) -> None:
    """
    Configura el sistema de logging completo de Forge Platform.

    Esta función debe llamarse exactamente UNA VEZ durante el bootstrap de la
    aplicación, antes de que cualquier módulo emita logs. Llamamadas subsecuentes
    son no-ops a menos que force_reconfigure=True.

    Configura dos sistemas en paralelo:
      1. structlog: logger principal para código Forge.
      2. logging estándar Python: captura logs de terceros (httpx, aiosqlite,
         openai SDK, etc.) y los redirige por el mismo pipeline structlog.

    También configura handlers de archivo con rotación automática si
    config.log_dir está especificado.

    Args:
        config:            Configuración de observabilidad del sistema.
        force_reconfigure: Si True, reconfigura aunque ya se haya configurado.
                           Útil para testing.

    Raises:
        OSError: Si no se puede crear el directorio de logs o el archivo de log.
    """
    global _logging_configured  # noqa: PLW0603
    if _logging_configured and not force_reconfigure:
        return

    # --- Construir el pipeline de processors ---
    processors = _build_processors_pipeline(
        config.log_format,
        include_caller_info=config.log_level in {LogLevel.DEBUG, LogLevel.INFO},
    )

    # --- Configurar structlog ---
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            _log_level_to_int(config.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # --- Configurar el módulo logging estándar (para terceros) ---
    _configure_stdlib_logging(config)

    _logging_configured = True


# Bandera de inicialización — evita reconfiguración accidental
_logging_configured: bool = False


def _configure_stdlib_logging(config: ObservabilityConfig) -> None:
    """
    Configura el módulo logging estándar de Python para redirigir a structlog.

    Las librerías de terceros (httpx, openai, aiosqlite, etc.) usan el módulo
    estándar logging. Esta función instala un handler que captura esos logs y
    los procesa a través del mismo pipeline structlog, garantizando formato y
    contexto uniformes.

    También configura la rotación de archivos de log si el directorio de log
    está especificado y accesible.

    Args:
        config: Configuración de observabilidad.
    """
    # Nivel de log para stdlib
    stdlib_level = _log_level_to_int(config.log_level)

    # Handler que redirige stdlib → structlog
    structlog_handler = structlog.stdlib.ProcessorFormatter.wrap_for_formatter(
        foreign_pre_chain=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
        ]
    )

    # Formatter que usa el mismo pipeline structlog
    if config.log_format == LogFormat.JSON:
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
        )
    else:
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=True),
        )

    # Handler a stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(stdlib_level)

    handlers: list[logging.Handler] = [stream_handler]

    # Handler a archivo con rotación (si el directorio es accesible)
    log_file_path = config.log_file_path
    try:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_file_path),
            maxBytes=config.log_max_size_mb * 1024 * 1024,
            backupCount=config.log_backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(stdlib_level)
        handlers.append(file_handler)
    except OSError:
        # Si no se puede crear el archivo de log, continuar solo con stdout
        # El sistema no debe fallar solo porque no pueda escribir logs a disco
        pass

    # Configurar el root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(stdlib_level)
    # Limpiar handlers existentes para evitar duplicados en tests
    root_logger.handlers.clear()
    for handler in handlers:
        root_logger.addHandler(handler)

    # Silenciar loggers muy verbosos de terceros
    _configure_third_party_log_levels(stdlib_level)


def _configure_third_party_log_levels(base_level: int) -> None:
    """
    Ajusta los niveles de log de librerías de terceros conocidas.

    Algunas librerías son excesivamente verbosas en DEBUG. Esta función
    establece niveles más altos para ellas para evitar ruido en los logs.

    Args:
        base_level: Nivel base del sistema. Si es DEBUG, algunos terceros
                    se configuran en INFO para reducir el ruido.
    """
    noisy_loggers = {
        "httpx": logging.WARNING,
        "httpcore": logging.WARNING,
        "openai": logging.WARNING,
        "anthropic": logging.WARNING,
        "aiosqlite": logging.WARNING,
        "asyncio": logging.WARNING,
        "urllib3": logging.WARNING,
        "charset_normalizer": logging.WARNING,
    }

    # En DEBUG, permitir INFO para librerías críticas de infraestructura
    if base_level <= logging.DEBUG:
        noisy_loggers["aiosqlite"] = logging.INFO
        noisy_loggers["httpx"] = logging.INFO

    for logger_name, level in noisy_loggers.items():
        logging.getLogger(logger_name).setLevel(level)


def _log_level_to_int(level: LogLevel) -> int:
    """
    Convierte un LogLevel del schema de configuración al entero de stdlib logging.

    Args:
        level: LogLevel del enum de configuración.

    Returns:
        Entero correspondiente en el módulo logging estándar.
    """
    _mapping: dict[LogLevel, int] = {
        LogLevel.DEBUG: logging.DEBUG,
        LogLevel.INFO: logging.INFO,
        LogLevel.WARNING: logging.WARNING,
        LogLevel.ERROR: logging.ERROR,
        LogLevel.CRITICAL: logging.CRITICAL,
    }
    return _mapping[level]


# =============================================================================
# FACTORY DE LOGGERS
# =============================================================================


def get_logger(name: str, **initial_context: Any) -> structlog.BoundLogger:
    """
    Obtiene un logger structlog vinculado al módulo especificado.

    Este es el punto de entrada principal para obtener loggers en cualquier
    módulo del ecosistema Forge. El logger retornado está pre-vinculado con
    el nombre del módulo y cualquier contexto adicional especificado.

    Los loggers son lightweight — crearlos es barato y pueden crearse a nivel
    de módulo (variable global) o dentro de funciones/métodos.

    Args:
        name:            Nombre del módulo (usar __name__ por convención).
        **initial_context: Contexto adicional a vincular permanentemente
                           a este logger (aplica a todos sus logs).

    Returns:
        BoundLogger de structlog listo para usar.

    Example:
        # A nivel de módulo (recomendado para módulos con muchos logs):
        logger = get_logger(__name__)

        # Con contexto adicional permanente:
        logger = get_logger(__name__, component="task_executor", version="1.0")

        # Uso:
        logger.info("tool_invocada", tool_id="web_search", risk="none")
        logger.warning("retry_scheduled", attempt=2, delay_seconds=2.0)
        logger.error("tool_fallida", error=forge_error)
    """
    bound_logger = structlog.get_logger(name)
    if initial_context:
        bound_logger = bound_logger.bind(**initial_context)
    return bound_logger  # type: ignore[return-value]


# =============================================================================
# LOGGER ESPECIALIZADO PARA AUDIT
# =============================================================================


class AuditLogger:
    """
    Logger especializado para el audit trail de acciones del agente.

    Emite logs en un formato estructurado específico para auditoría, separado
    del log general del sistema. Cada entrada de audit incluye campos obligatorios
    que garantizan la trazabilidad completa de cada acción ejecutada.

    A diferencia del logger general, el AuditLogger:
      - Siempre emite en nivel INFO o superior (nunca DEBUG).
      - Incluye campos obligatorios de auditoría en cada entrada.
      - Usa un logger nombrado 'forge.audit' para facilitar su separación
        en sistemas de log aggregation (Loki, Elasticsearch).
      - Nunca descarta entradas por nivel de log — el audit es siempre completo.
    """

    def __init__(self) -> None:
        self._logger = get_logger(
            "forge.audit",
            subsystem="audit",
        )

    def record_tool_execution(
        self,
        *,
        session_id: str,
        task_id: str | None,
        tool_id: str,
        risk_level: str,
        policy_decision: str,
        success: bool,
        duration_ms: float,
        approval_id: str | None = None,
        input_summary: str | None = None,
        output_summary: str | None = None,
        error_code: str | None = None,
    ) -> None:
        """
        Registra la ejecución de una tool en el audit trail.

        Args:
            session_id:      ID de la sesión en la que ocurrió la ejecución.
            task_id:         ID de la tarea padre (None si es ejecución directa).
            tool_id:         Identificador único de la tool ejecutada.
            risk_level:      Nivel de riesgo clasificado ('none', 'low', 'medium', 'high').
            policy_decision: Decisión del policy engine ('allow', 'deny', 'require_approval').
            success:         True si la ejecución completó exitosamente.
            duration_ms:     Duración de la ejecución en milisegundos.
            approval_id:     ID de la aprobación del usuario si se requirió.
            input_summary:   Resumen sanitizado del input (NUNCA el input completo).
            output_summary:  Resumen sanitizado del output.
            error_code:      Código de error ForgeError si la ejecución falló.
        """
        self._logger.info(
            "audit.tool_execution",
            session_id=session_id,
            task_id=task_id,
            tool_id=tool_id,
            risk_level=risk_level,
            policy_decision=policy_decision,
            success=success,
            duration_ms=round(duration_ms, 2),
            approval_id=approval_id,
            input_summary=input_summary,
            output_summary=output_summary,
            error_code=error_code,
            audit_event_type="tool_execution",
        )

    def record_policy_decision(
        self,
        *,
        session_id: str,
        tool_id: str,
        decision: str,
        risk_level: str,
        policy_name: str | None = None,
        reason: str | None = None,
    ) -> None:
        """
        Registra una decisión del policy engine en el audit trail.

        Args:
            session_id:  ID de la sesión.
            tool_id:     Tool para la que se evaluó la política.
            decision:    Decisión tomada ('allow', 'deny', 'require_approval').
            risk_level:  Nivel de riesgo de la acción evaluada.
            policy_name: Nombre de la policy que tomó la decisión.
            reason:      Razón de la decisión (especialmente útil en denials).
        """
        self._logger.info(
            "audit.policy_decision",
            session_id=session_id,
            tool_id=tool_id,
            decision=decision,
            risk_level=risk_level,
            policy_name=policy_name,
            reason=reason,
            audit_event_type="policy_decision",
        )

    def record_approval(
        self,
        *,
        session_id: str,
        approval_id: str,
        action_description: str,
        risk_level: str,
        decision: str,
        decided_by: str,
        duration_seconds: float | None = None,
    ) -> None:
        """
        Registra el resultado de un flujo de aprobación en el audit trail.

        Args:
            session_id:         ID de la sesión.
            approval_id:        ID único de la solicitud de aprobación.
            action_description: Descripción legible de la acción a aprobar.
            risk_level:         Nivel de riesgo de la acción.
            decision:           Decisión del usuario ('granted', 'denied', 'expired').
            decided_by:         Identificador del usuario que decidió.
            duration_seconds:   Tiempo en segundos que tardó el usuario en decidir.
        """
        self._logger.info(
            "audit.approval",
            session_id=session_id,
            approval_id=approval_id,
            action_description=action_description,
            risk_level=risk_level,
            decision=decision,
            decided_by=decided_by,
            duration_seconds=duration_seconds,
            audit_event_type="approval",
        )

    def record_security_event(
        self,
        *,
        session_id: str,
        event_type: str,
        severity: str,
        details: dict[str, Any],
    ) -> None:
        """
        Registra un evento de seguridad en el audit trail con nivel WARNING.

        Los eventos de seguridad (path traversal attempts, sandbox violations,
        input sanitization blocks) siempre se emiten con nivel WARNING o superior
        para que sean capturados incluso con log levels restrictivos.

        Args:
            session_id: ID de la sesión donde ocurrió el evento.
            event_type: Tipo de evento ('path_traversal', 'sandbox_violation', etc.).
            severity:   Severidad del evento ('low', 'medium', 'high', 'critical').
            details:    Detalles adicionales del evento (sanitizados).
        """
        self._logger.warning(
            "audit.security_event",
            session_id=session_id,
            event_type=event_type,
            severity=severity,
            details=details,
            audit_event_type="security_event",
        )


# =============================================================================
# INSTANCIA GLOBAL DEL AUDIT LOGGER
# =============================================================================

# Instancia singleton del AuditLogger. Se inicializa una sola vez y se
# importa directamente en los módulos que necesitan emitir audit entries.
# No requiere configuración adicional — usa el mismo sistema configurado
# por configure_logging().
audit_logger: AuditLogger = AuditLogger()


# =============================================================================
# DECORADOR DE LOGGING PARA FUNCIONES CRÍTICAS
# =============================================================================


def log_execution(
    *,
    level: str = "info",
    include_args: bool = False,
    include_result: bool = False,
) -> Any:
    """
    Decorador que añade logging automático de entrada/salida a funciones críticas.

    Emite un log al inicio y otro al finalizar (con duración) de la función
    decorada. Útil para instrumentar funciones de alto nivel sin añadir
    boilerplate de logging en cada una.

    Args:
        level:          Nivel de log a usar ('debug', 'info', 'warning').
        include_args:   Si True, incluye los argumentos de la llamada en el log.
                        PRECAUCIÓN: puede incluir datos sensibles si no se tiene cuidado.
        include_result: Si True, incluye el resultado en el log de salida.
                        PRECAUCIÓN: el resultado puede ser grande o sensible.

    Returns:
        Decorador que envuelve la función con logging automático.

    Example:
        @log_execution(level="info")
        async def execute_tool(self, tool_id: str, input: ToolInput) -> ToolOutput:
            ...
    """
    import functools
    import time

    def decorator(func: Any) -> Any:
        func_logger = get_logger(func.__module__, function=func.__qualname__)
        log_method = getattr(func_logger, level)

        if _is_coroutine(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                log_kwargs: dict[str, Any] = {}
                if include_args:
                    log_kwargs["args_repr"] = repr(args[1:])  # skip self
                    log_kwargs["kwargs_repr"] = repr(kwargs)

                log_method(f"{func.__name__}.started", **log_kwargs)
                start = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    duration_ms = (time.perf_counter() - start) * 1000
                    result_kwargs: dict[str, Any] = {"duration_ms": round(duration_ms, 2)}
                    if include_result:
                        result_kwargs["result_repr"] = repr(result)
                    log_method(f"{func.__name__}.completed", **result_kwargs)
                    return result
                except Exception as exc:
                    duration_ms = (time.perf_counter() - start) * 1000
                    func_logger.error(
                        f"{func.__name__}.failed",
                        error=exc,
                        duration_ms=round(duration_ms, 2),
                    )
                    raise
            return async_wrapper

        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                log_kwargs = {}
                if include_args:
                    log_kwargs["args_repr"] = repr(args[1:])
                    log_kwargs["kwargs_repr"] = repr(kwargs)

                log_method(f"{func.__name__}.started", **log_kwargs)
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.perf_counter() - start) * 1000
                    result_kwargs = {"duration_ms": round(duration_ms, 2)}
                    if include_result:
                        result_kwargs["result_repr"] = repr(result)
                    log_method(f"{func.__name__}.completed", **result_kwargs)
                    return result
                except Exception as exc:
                    duration_ms = (time.perf_counter() - start) * 1000
                    func_logger.error(
                        f"{func.__name__}.failed",
                        error=exc,
                        duration_ms=round(duration_ms, 2),
                    )
                    raise
            return sync_wrapper

    return decorator


def _is_coroutine(func: Any) -> bool:
    """Determina si una función es una coroutine (async def)."""
    import asyncio
    return asyncio.iscoroutinefunction(func)