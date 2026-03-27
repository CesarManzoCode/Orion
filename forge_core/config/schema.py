"""
Schema de configuración del ecosistema Forge Platform.

Este módulo define la estructura completa de configuración usando Pydantic v2
con Settings management. Es la fuente de verdad única para toda configuración
del sistema — ningún módulo lee variables de entorno directamente.

Principios de diseño:
  1. Defaults sensatos para todo. El usuario solo necesita configurar la API key.
  2. Validación estricta en el arranque. Mejor fallar temprano con un mensaje claro
     que fallar tarde con un error críptico de infraestructura.
  3. Separación por dominio funcional: cada subsistema tiene su propio grupo
     de configuración, facilitando el descubrimiento y la documentación.
  4. Secrets se manejan como SecretStr — nunca se serializan ni se loguean en texto.
  5. El schema es versionado para soportar migraciones automáticas entre versiones.

Resolución de configuración (orden de precedencia, mayor a menor):
  1. Variables de entorno (prefijo FORGE_USER__)
  2. Archivo .env en el directorio de trabajo
  3. Archivo .env en ~/.hiperforge_user/
  4. Valores por defecto definidos en este schema

Variables de entorno clave:
  FORGE_USER__LLM__OPENAI_API_KEY         — API key de OpenAI
  FORGE_USER__LLM__ANTHROPIC_API_KEY      — API key de Anthropic
  FORGE_USER__LLM__DEFAULT_PROVIDER       — Provider LLM por defecto
  FORGE_USER__STORAGE__DATA_DIR           — Directorio de datos
  FORGE_USER__OBSERVABILITY__LOG_LEVEL    — Nivel de logging
  FORGE_USER__SEARCH__TAVILY_API_KEY      — API key de Tavily Search

Uso:
    from forge_core.config.schema import ForgeUserConfig, load_config

    config = load_config()  # carga y valida desde entorno
    api_key = config.llm.openai_api_key.get_secret_value()
"""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# ENUMERACIONES DE CONFIGURACIÓN
# =============================================================================


class LLMProvider(str, Enum):
    """
    Providers de LLM soportados por el ecosistema Forge.
    El valor string coincide con el identificador usado en logs y métricas.
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class LogLevel(str, Enum):
    """Niveles de logging estándar mapeados a los de Python logging."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Formato de salida del sistema de logging estructurado."""

    JSON = "json"
    """
    Salida JSON por línea. Ideal para producción, ingesta por Loki/Elasticsearch.
    """
    CONSOLE = "console"
    """
    Salida legible para humanos con colores. Ideal para desarrollo local.
    """


class TracingExporter(str, Enum):
    """Exporter de trazas OpenTelemetry."""

    NONE = "none"
    """Sin exportación de trazas. Útil para testing."""
    CONSOLE = "console"
    """Exportar trazas a stdout. Útil para desarrollo."""
    OTLP = "otlp"
    """Exportar via OTLP a Jaeger, Grafana Tempo, etc."""


# =============================================================================
# GRUPOS DE CONFIGURACIÓN POR SUBSISTEMA
# =============================================================================


class LLMConfig(BaseModel):
    """
    Configuración del subsistema LLM.

    Gestiona todos los aspectos de la integración con providers de lenguaje:
    credenciales, selección de modelo, timeouts, y políticas de retry.
    """

    model_config = {"frozen": True}

    # --- Selección de provider y modelo ---
    default_provider: LLMProvider = Field(
        default=LLMProvider.ANTHROPIC,
        description="Provider LLM a usar por defecto cuando no se especifica uno.",
    )
    fallback_provider: LLMProvider | None = Field(
        default=None,
        description=(
            "Provider alternativo a usar si el principal falla. "
            "None desactiva el fallback automático."
        ),
    )

    # --- OpenAI ---
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="API key de OpenAI. Requerida si default_provider=openai.",
    )
    openai_model: str = Field(
        default="gpt-4o",
        description="Modelo de OpenAI a usar. Debe soportar function calling.",
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description=(
            "URL base del API de OpenAI. Cambiar solo para proxies o "
            "endpoints compatibles (Azure OpenAI, LiteLLM, etc.)."
        ),
    )

    # --- Anthropic ---
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="API key de Anthropic. Requerida si default_provider=anthropic.",
    )
    anthropic_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Modelo de Anthropic a usar. Debe soportar tool use.",
    )
    anthropic_base_url: str = Field(
        default="https://api.anthropic.com",
        description="URL base del API de Anthropic.",
    )

    # --- Modelo local (Ollama) ---
    local_model_url: str = Field(
        default="http://localhost:11434",
        description="URL del servidor Ollama para inferencia local.",
    )
    local_model_name: str = Field(
        default="llama3.2",
        description="Nombre del modelo cargado en Ollama.",
    )

    # --- Parámetros de generación ---
    max_tokens_response: int = Field(
        default=4096,
        ge=256,
        le=16384,
        description="Máximo de tokens en la respuesta del LLM.",
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description=(
            "Temperatura de muestreo. Valores bajos (0.1-0.4) para tareas "
            "precisas y tool use. Valores altos para creatividad."
        ),
    )
    context_window_tokens: int = Field(
        default=128_000,
        ge=4_096,
        le=2_000_000,
        description="Tamaño máximo del context window del modelo seleccionado.",
    )

    # --- Timeouts y retry ---
    request_timeout_seconds: float = Field(
        default=60.0,
        ge=5.0,
        le=300.0,
        description="Timeout máximo para una request al LLM API.",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Número máximo de reintentos ante errores transitorios.",
    )
    retry_base_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Delay base para backoff exponencial entre reintentos.",
    )
    retry_max_delay_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Delay máximo entre reintentos (cap del backoff exponencial).",
    )

    # --- Streaming ---
    enable_streaming: bool = Field(
        default=True,
        description=(
            "Habilitar streaming de respuestas para mejor UX. "
            "Deshabilitar solo para debugging."
        ),
    )

    @model_validator(mode="after")
    def validate_provider_credentials(self) -> LLMConfig:
        """
        Valida que el provider por defecto tenga las credenciales necesarias.

        El provider local (Ollama) no requiere credenciales. OpenAI y Anthropic
        requieren su respectiva API key. El fallback provider también se valida
        si está configurado.
        """
        def _needs_key(provider: LLMProvider) -> bool:
            return provider in {LLMProvider.OPENAI, LLMProvider.ANTHROPIC}

        def _has_key(provider: LLMProvider) -> bool:
            if provider == LLMProvider.OPENAI:
                return self.openai_api_key is not None
            if provider == LLMProvider.ANTHROPIC:
                return self.anthropic_api_key is not None
            return True  # LOCAL no necesita key

        if _needs_key(self.default_provider) and not _has_key(self.default_provider):
            provider_name = self.default_provider.value
            key_name = f"FORGE_USER__LLM__{provider_name.upper()}_API_KEY"
            raise ValueError(
                f"El provider '{provider_name}' requiere una API key. "
                f"Configura la variable de entorno '{key_name}'."
            )

        if (
            self.fallback_provider is not None
            and _needs_key(self.fallback_provider)
            and not _has_key(self.fallback_provider)
        ):
            provider_name = self.fallback_provider.value
            key_name = f"FORGE_USER__LLM__{provider_name.upper()}_API_KEY"
            raise ValueError(
                f"El fallback provider '{provider_name}' requiere una API key. "
                f"Configura '{key_name}' o desactiva el fallback."
            )

        return self

    @field_validator("retry_max_delay_seconds")
    @classmethod
    def validate_retry_delays(cls, v: float, info: Any) -> float:
        """Asegura que el delay máximo sea mayor que el delay base."""
        base = info.data.get("retry_base_delay_seconds", 1.0)
        if v < base:
            raise ValueError(
                f"retry_max_delay_seconds ({v}) debe ser >= retry_base_delay_seconds ({base})"
            )
        return v


class StorageConfig(BaseModel):
    """
    Configuración del subsistema de persistencia local.

    HiperForge User usa SQLite para todo el storage local. No hay dependencias
    de servicios externos para la persistencia — el sistema funciona completamente
    offline excepto para LLM y búsqueda web.
    """

    model_config = {"frozen": True}

    data_dir: Path = Field(
        default_factory=lambda: Path.home() / ".hiperforge_user" / "data",
        description=(
            "Directorio raíz para todos los datos persistidos. "
            "Se crea automáticamente si no existe."
        ),
    )
    database_filename: str = Field(
        default="forge_user.db",
        description="Nombre del archivo SQLite principal.",
    )
    audit_database_filename: str = Field(
        default="forge_audit.db",
        description=(
            "Nombre del archivo SQLite para el audit log. "
            "Separado del principal para facilitar su exportación y protección."
        ),
    )

    # --- SQLite tuning ---
    connection_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=120.0,
        description="Timeout para adquirir conexión a SQLite.",
    )
    wal_mode: bool = Field(
        default=True,
        description=(
            "Habilitar WAL (Write-Ahead Logging) en SQLite. "
            "Mejora la concurrencia de lecturas/escrituras simultáneas."
        ),
    )
    cache_size_pages: int = Field(
        default=10_000,
        ge=1_000,
        le=100_000,
        description=(
            "Tamaño del cache de páginas SQLite. Cada página es ~4KB. "
            "10000 páginas ≈ 40MB de cache."
        ),
    )
    mmap_size_bytes: int = Field(
        default=256 * 1024 * 1024,  # 256 MB
        ge=0,
        description="Tamaño del memory-mapped I/O para SQLite (0 = deshabilitado).",
    )

    # --- Backup y mantenimiento ---
    auto_vacuum: bool = Field(
        default=True,
        description="Habilitar VACUUM automático para recuperar espacio en SQLite.",
    )
    backup_enabled: bool = Field(
        default=True,
        description="Habilitar backups automáticos de la base de datos.",
    )
    backup_interval_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Intervalo entre backups automáticos, en horas.",
    )
    max_backup_files: int = Field(
        default=7,
        ge=1,
        le=365,
        description="Número máximo de archivos de backup a conservar.",
    )

    # --- Retención de datos ---
    session_retention_days: int = Field(
        default=90,
        ge=1,
        le=3650,
        description="Días de retención para sesiones cerradas antes de archivarlas.",
    )
    audit_retention_days: int = Field(
        default=365,
        ge=30,
        le=3650,
        description="Días de retención para entradas del audit log.",
    )

    @model_validator(mode="after")
    def ensure_data_dir_is_absolute(self) -> StorageConfig:
        """Normaliza el data_dir a una ruta absoluta."""
        object.__setattr__(self, "data_dir", self.data_dir.resolve())
        return self

    @property
    def database_path(self) -> Path:
        """Ruta completa al archivo de base de datos principal."""
        return self.data_dir / self.database_filename

    @property
    def audit_database_path(self) -> Path:
        """Ruta completa al archivo de base de datos de auditoría."""
        return self.data_dir / self.audit_database_filename

    @property
    def backup_dir(self) -> Path:
        """Directorio donde se almacenan los backups automáticos."""
        return self.data_dir / "backups"


class SecurityConfig(BaseModel):
    """
    Configuración del subsistema de seguridad y sandboxing.

    Define los límites de operación del agente: directorios permitidos,
    aplicaciones en la allowlist, timeouts de sandbox, y umbrales de riesgo
    para aprobaciones automáticas.
    """

    model_config = {"frozen": True}

    # --- Filesystem ---
    allowed_directories: list[Path] = Field(
        default_factory=lambda: [
            Path.home(),
            Path.home() / "Documents",
            Path.home() / "Downloads",
            Path.home() / "Desktop",
        ],
        description=(
            "Directorios donde el agente puede leer y escribir archivos. "
            "Rutas fuera de esta lista requieren aprobación explícita."
        ),
    )
    blocked_path_patterns: list[str] = Field(
        default_factory=lambda: [
            "**/.ssh/*",
            "**/.gnupg/*",
            "**/.config/*/credentials*",
            "**/.*token*",
            "**/.*key*",
            "**/.*secret*",
            "/etc/shadow",
            "/etc/passwd",
            "/etc/sudoers",
            "/proc/**",
            "/sys/**",
            "C:\\Windows\\System32\\**",
            "C:\\Windows\\SysWOW64\\**",
            "%APPDATA%\\Microsoft\\Credentials\\**",
        ],
        description=(
            "Patrones glob de rutas nunca accesibles, independientemente de "
            "los directorios permitidos. Bloqueadas a nivel de policy engine."
        ),
    )
    max_file_size_mb: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Tamaño máximo de archivo que el agente puede leer, en MB.",
    )
    max_file_write_size_mb: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Tamaño máximo de archivo que el agente puede escribir, en MB.",
    )

    # --- Desktop automation ---
    app_allowlist: list[str] = Field(
        default_factory=lambda: [
            # Navegadores
            "firefox", "google-chrome", "chromium", "chromium-browser",
            "brave-browser", "microsoft-edge",
            # Productividad
            "libreoffice", "libreoffice-writer", "libreoffice-calc",
            "libreoffice-impress", "code", "codium", "gedit", "kate",
            "notepad", "notepad++", "wordpad",
            # Utilidades
            "calculator", "gnome-calculator", "kcalc",
            "nautilus", "dolphin", "thunar", "files",
            "explorer.exe",
            # Comunicación
            "thunderbird", "evolution",
        ],
        description=(
            "Lista de aplicaciones que el agente puede lanzar sin aprobación "
            "después de la primera vez. Identificadores de ejecutable (sin extensión)."
        ),
    )
    clipboard_access_enabled: bool = Field(
        default=False,
        description=(
            "Habilitar acceso al portapapeles. False por defecto para privacidad. "
            "Se activa con aprobación explícita del usuario (first-time approval)."
        ),
    )
    screenshot_enabled: bool = Field(
        default=True,
        description="Permitir tomar screenshots del escritorio.",
    )

    # --- Aprobaciones ---
    approval_timeout_seconds: float = Field(
        default=300.0,
        ge=10.0,
        le=3600.0,
        description=(
            "Tiempo máximo de espera para una aprobación del usuario. "
            "Expirado este tiempo, la aprobación se deniega automáticamente."
        ),
    )
    remember_approvals: bool = Field(
        default=True,
        description=(
            "Recordar las decisiones de aprobación del usuario para no preguntarle "
            "de nuevo (ej: una vez aprobado 'firefox', no se vuelve a preguntar)."
        ),
    )

    # --- Sandbox ---
    sandbox_timeout_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=120.0,
        description="Timeout máximo para operaciones ejecutadas en el proceso sandbox.",
    )
    sandbox_enabled: bool = Field(
        default=True,
        description=(
            "Ejecutar acciones de desktop en proceso sandbox separado. "
            "Deshabilitar solo para debugging — NUNCA en producción."
        ),
    )

    # --- Rate limiting ---
    max_tool_calls_per_minute: int = Field(
        default=60,
        ge=1,
        le=300,
        description="Límite de invocaciones de tools por minuto en toda la sesión.",
    )
    max_web_requests_per_minute: int = Field(
        default=30,
        ge=1,
        le=120,
        description="Límite de requests web (search + fetch) por minuto.",
    )

    # --- Red ---
    allowed_domains: list[str] = Field(
        default_factory=lambda: ["*"],
        description=(
            "Dominios web permitidos para búsqueda y fetch. "
            "'*' permite todos. Útil en entornos corporativos para restringir."
        ),
    )
    blocked_domains: list[str] = Field(
        default_factory=list,
        description="Dominios web explícitamente bloqueados, independientemente de allowed_domains.",
    )

    @model_validator(mode="after")
    def resolve_allowed_directories(self) -> SecurityConfig:
        """Normaliza todos los directorios permitidos a rutas absolutas."""
        resolved = [d.resolve() for d in self.allowed_directories]
        object.__setattr__(self, "allowed_directories", resolved)
        return self


class ObservabilityConfig(BaseModel):
    """
    Configuración del subsistema de observabilidad.

    Gestiona logging estructurado, tracing distribuido y métricas.
    En V1, todo se almacena localmente. Los puertos están diseñados para
    conectar con sistemas externos (Loki, Jaeger, Prometheus) en V2/V3.
    """

    model_config = {"frozen": True}

    # --- Logging ---
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Nivel mínimo de log a emitir.",
    )
    log_format: LogFormat = Field(
        default=LogFormat.CONSOLE,
        description=(
            "Formato de los logs. JSON para producción/ingesta, "
            "CONSOLE para desarrollo local."
        ),
    )
    log_dir: Path = Field(
        default_factory=lambda: Path.home() / ".hiperforge_user" / "logs",
        description="Directorio donde se almacenan los archivos de log.",
    )
    log_filename: str = Field(
        default="forge_user.log",
        description="Nombre del archivo de log principal.",
    )
    log_max_size_mb: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Tamaño máximo del archivo de log antes de rotar, en MB.",
    )
    log_backup_count: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Número de archivos de log rotados a conservar.",
    )
    log_include_stacktrace: bool = Field(
        default=True,
        description=(
            "Incluir stack traces completos en los logs de error. "
            "True recomendado salvo en entornos con PII estricto."
        ),
    )

    # --- Tracing OpenTelemetry ---
    tracing_enabled: bool = Field(
        default=False,
        description=(
            "Habilitar tracing distribuido via OpenTelemetry. "
            "False por defecto para no añadir overhead en V1."
        ),
    )
    tracing_exporter: TracingExporter = Field(
        default=TracingExporter.NONE,
        description="Exporter de trazas a usar cuando tracing_enabled=True.",
    )
    tracing_otlp_endpoint: str = Field(
        default="http://localhost:4317",
        description="Endpoint OTLP (gRPC) cuando tracing_exporter=otlp.",
    )
    tracing_sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Fracción de requests a trazar (0.0 = ninguna, 1.0 = todas). "
            "Reducir en producción si el volumen es alto."
        ),
    )
    service_name: str = Field(
        default="hiperforge-user",
        description="Nombre del servicio para identificación en sistemas de tracing.",
    )
    service_version: str = Field(
        default="0.1.0",
        description="Versión del servicio para correlación de traces con deploys.",
    )

    # --- Métricas ---
    metrics_enabled: bool = Field(
        default=True,
        description="Habilitar recolección de métricas básicas.",
    )
    metrics_dir: Path = Field(
        default_factory=lambda: Path.home() / ".hiperforge_user" / "metrics",
        description="Directorio para almacenar snapshots de métricas locales.",
    )

    @model_validator(mode="after")
    def resolve_paths(self) -> ObservabilityConfig:
        """Normaliza los directorios de log y métricas a rutas absolutas."""
        object.__setattr__(self, "log_dir", self.log_dir.resolve())
        object.__setattr__(self, "metrics_dir", self.metrics_dir.resolve())
        return self

    @property
    def log_file_path(self) -> Path:
        """Ruta completa al archivo de log principal."""
        return self.log_dir / self.log_filename


class MemoryConfig(BaseModel):
    """
    Configuración del subsistema de memoria y contexto conversacional.

    Controla cómo el agente gestiona el historial de conversación, cuándo
    compactar el contexto, y qué información persiste entre sesiones.
    """

    model_config = {"frozen": True}

    # --- Context window ---
    system_prompt_reserved_tokens: int = Field(
        default=2_000,
        ge=500,
        le=8_000,
        description="Tokens reservados para el system prompt en el context window.",
    )
    tools_schema_reserved_tokens: int = Field(
        default=3_000,
        ge=500,
        le=10_000,
        description=(
            "Tokens reservados para los schemas de tools en el context window. "
            "Varía según cuántas tools estén activas."
        ),
    )
    response_reserved_tokens: int = Field(
        default=4_096,
        ge=256,
        le=16_384,
        description="Tokens reservados para la respuesta del LLM en el context window.",
    )
    user_profile_reserved_tokens: int = Field(
        default=300,
        ge=100,
        le=1_000,
        description="Tokens reservados para el resumen del perfil de usuario.",
    )

    # --- Compactación de historia ---
    compaction_threshold_ratio: float = Field(
        default=0.85,
        ge=0.5,
        le=0.95,
        description=(
            "Fracción del budget de historia que, al superarse, activa la "
            "compactación automática. 0.85 = compactar cuando se usa el 85%."
        ),
    )
    compaction_summary_tokens: int = Field(
        default=500,
        ge=100,
        le=2_000,
        description=(
            "Tokens máximos del resumen generado durante compactación. "
            "El resumen reemplaza los turns compactados en el context."
        ),
    )
    min_turns_before_compaction: int = Field(
        default=10,
        ge=3,
        le=50,
        description=(
            "Número mínimo de turns que deben existir antes de considerar "
            "compactación. Evita compactar conversaciones muy cortas."
        ),
    )

    # --- Sesiones ---
    session_auto_pause_minutes: int = Field(
        default=240,
        ge=10,
        le=1440,
        description=(
            "Minutos de inactividad tras los cuales la sesión se pausa "
            "automáticamente para liberar recursos."
        ),
    )
    restore_last_session: bool = Field(
        default=True,
        description=(
            "Al arrancar la app, restaurar automáticamente la última sesión "
            "activa. False para empezar siempre con sesión nueva."
        ),
    )
    max_active_sessions: int = Field(
        default=1,
        ge=1,
        le=10,
        description=(
            "Número máximo de sesiones activas simultáneamente. "
            "En V1 es siempre 1 (single-user). Preparado para V3 multi-user."
        ),
    )

    # --- Memoria a largo plazo ---
    long_term_memory_enabled: bool = Field(
        default=True,
        description=(
            "Habilitar persistencia de hechos y preferencias entre sesiones. "
            "Desactivar para máxima privacidad (amnesia entre sesiones)."
        ),
    )
    max_long_term_facts: int = Field(
        default=500,
        ge=10,
        le=10_000,
        description=(
            "Número máximo de hechos a largo plazo almacenados por usuario. "
            "Al superarse, se descartan los más antiguos (LRU)."
        ),
    )


class SearchConfig(BaseModel):
    """
    Configuración del subsistema de búsqueda web.

    Gestiona las credenciales y comportamiento del provider de búsqueda.
    En V1, se soporta Tavily (optimizado para LLM). El puerto está diseñado
    para añadir providers adicionales (Brave, SerpAPI, etc.) en V2.
    """

    model_config = {"frozen": True}

    tavily_api_key: SecretStr | None = Field(
        default=None,
        description=(
            "API key de Tavily Search. Requerida para herramientas de búsqueda web. "
            "Obtener en: https://tavily.com"
        ),
    )
    max_results_per_search: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Número máximo de resultados por búsqueda web.",
    )
    search_timeout_seconds: float = Field(
        default=15.0,
        ge=3.0,
        le=60.0,
        description="Timeout para requests de búsqueda web.",
    )
    fetch_timeout_seconds: float = Field(
        default=20.0,
        ge=3.0,
        le=120.0,
        description="Timeout para fetch de contenido de páginas web.",
    )
    max_page_content_tokens: int = Field(
        default=8_000,
        ge=500,
        le=50_000,
        description=(
            "Máximo de tokens del contenido de una página web a incluir "
            "en el context del LLM tras el fetch."
        ),
    )
    search_enabled: bool = Field(
        default=True,
        description=(
            "Habilitar la capacidad de búsqueda web. "
            "False deshabilita las tools de búsqueda aunque haya API key configurada."
        ),
    )

    @model_validator(mode="after")
    def warn_if_search_enabled_without_key(self) -> SearchConfig:
        """
        Emite una advertencia si la búsqueda está habilitada pero no hay API key.

        No es un error fatal porque el sistema funciona sin búsqueda web —
        simplemente las tools de búsqueda no estarán disponibles.
        """
        if self.search_enabled and self.tavily_api_key is None:
            import warnings
            warnings.warn(
                "FORGE_USER__SEARCH__TAVILY_API_KEY no está configurada. "
                "Las herramientas de búsqueda web no estarán disponibles. "
                "Configura la variable de entorno o deshabilita search_enabled=false.",
                stacklevel=2,
            )
        return self


class AgentConfig(BaseModel):
    """
    Configuración del comportamiento del agente conversacional.

    Controla la personalidad, los límites de planificación, los timeouts de
    ejecución y otras características del agente que impactan directamente
    en la experiencia del usuario.
    """

    model_config = {"frozen": True}

    # --- Identidad ---
    agent_name: str = Field(
        default="Forge",
        min_length=1,
        max_length=50,
        description="Nombre con el que el agente se identifica en la UI.",
    )
    agent_language: str = Field(
        default="es",
        pattern=r"^[a-z]{2}(-[A-Z]{2})?$",
        description=(
            "Idioma principal del agente (código BCP 47). "
            "Afecta el idioma de los system prompts y respuestas por defecto."
        ),
    )

    # --- Planificación ---
    max_plan_steps: int = Field(
        default=10,
        ge=2,
        le=25,
        description=(
            "Número máximo de pasos en un plan multi-step. "
            "El documento de arquitectura establece 10 como límite para evitar over-planning."
        ),
    )
    planning_complexity_threshold: int = Field(
        default=2,
        ge=1,
        le=5,
        description=(
            "Umbral de complejidad (número de tool calls estimados) a partir del "
            "cual se activa la planificación. Por debajo, se ejecuta directamente."
        ),
    )

    # --- Ejecución ---
    max_tool_calls_per_task: int = Field(
        default=20,
        ge=1,
        le=50,
        description=(
            "Número máximo de invocaciones de tools por tarea. "
            "Previene loops infinitos y sobre-ejecución."
        ),
    )
    task_timeout_seconds: float = Field(
        default=300.0,
        ge=10.0,
        le=1800.0,
        description="Timeout máximo para completar una tarea completa (incluyendo todas sus tools).",
    )
    single_tool_timeout_seconds: float = Field(
        default=30.0,
        ge=3.0,
        le=300.0,
        description="Timeout default para una invocación individual de tool.",
    )

    # --- Comportamiento conversacional ---
    show_thinking_indicators: bool = Field(
        default=True,
        description=(
            "Mostrar indicadores de progreso en la UI mientras el agente trabaja "
            "('Buscando...', 'Analizando...', etc.)."
        ),
    )
    include_source_citations: bool = Field(
        default=True,
        description=(
            "Incluir referencias a las fuentes usadas en las respuestas "
            "que involucran búsqueda web o documentos."
        ),
    )
    max_response_length_tokens: int = Field(
        default=2_000,
        ge=100,
        le=8_000,
        description=(
            "Longitud máxima sugerida de las respuestas del agente en tokens. "
            "Es una guía para el LLM, no un límite técnico absoluto."
        ),
    )


# =============================================================================
# CONFIGURACIÓN RAÍZ
# =============================================================================


class ForgeUserConfig(BaseSettings):
    """
    Configuración raíz de HiperForge User.

    Agrega todos los grupos de configuración en un único objeto tipado.
    Se carga automáticamente desde variables de entorno y archivos .env,
    con prefijo 'FORGE_USER__' y separador '__' para grupos anidados.

    Ejemplo de variables de entorno:
        FORGE_USER__LLM__OPENAI_API_KEY=sk-...
        FORGE_USER__LLM__DEFAULT_PROVIDER=openai
        FORGE_USER__STORAGE__DATA_DIR=/custom/path
        FORGE_USER__OBSERVABILITY__LOG_LEVEL=DEBUG
        FORGE_USER__SECURITY__SANDBOX_ENABLED=true

    La configuración es inmutable (frozen=True) después de la carga inicial.
    Cualquier cambio requiere recargar la configuración completa.
    """

    model_config = SettingsConfigDict(
        # Prefijo de variables de entorno
        env_prefix="FORGE_USER__",
        # Separador para grupos anidados (LLM__OPENAI_API_KEY)
        env_nested_delimiter="__",
        # Archivos .env a leer (en orden de precedencia)
        env_file=[
            Path.home() / ".hiperforge_user" / ".env",
            Path(".env"),
        ],
        env_file_encoding="utf-8",
        # Ignorar variables de entorno no reconocidas
        extra="ignore",
        # Inmutable después de inicialización
        frozen=True,
        # Validar los valores por defecto
        validate_default=True,
        # Case-insensitive para variables de entorno
        case_sensitive=False,
    )

    # --- Versión del schema para migraciones futuras ---
    config_schema_version: Annotated[
        str,
        Field(default="1.0.0", description="Versión del schema de configuración."),
    ] = "1.0.0"

    # --- Grupos de configuración ---
    llm: LLMConfig = Field(default_factory=LLMConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)

    # --- Entorno de ejecución ---
    environment: str = Field(
        default="production",
        pattern=r"^(development|staging|production|testing)$",
        description=(
            "Entorno de ejecución. Afecta defaults de logging, "
            "sandboxing y verificaciones de seguridad."
        ),
    )
    debug_mode: bool = Field(
        default=False,
        description=(
            "Modo de depuración. Habilita logging verbose, deshabilita "
            "algunas restricciones de seguridad (NUNCA en producción)."
        ),
    )

    @model_validator(mode="after")
    def validate_debug_security(self) -> ForgeUserConfig:
        """
        Emite una advertencia crítica si debug_mode está activo en producción.

        El modo debug puede desactivar restricciones de seguridad (como el sandbox)
        y aumentar el nivel de logging, potencialmente exponiendo información sensible.
        """
        if self.debug_mode and self.environment == "production":
            import warnings
            warnings.warn(
                "⚠️  DEBUG MODE ACTIVO EN ENTORNO DE PRODUCCIÓN. "
                "Esto puede comprometer la seguridad del sistema. "
                "Configura FORGE_USER__DEBUG_MODE=false.",
                stacklevel=2,
                category=RuntimeWarning,
            )
        return self

    @model_validator(mode="after")
    def apply_debug_overrides(self) -> ForgeUserConfig:
        """
        En modo debug, ajusta configuración para facilitar el desarrollo.

        Cambios aplicados en debug_mode=True:
          - Log level → DEBUG
          - Log format → CONSOLE
          - Tracing → habilitado con exporter CONSOLE
        """
        if self.debug_mode:
            # Pydantic v2 frozen=True requiere object.__setattr__ para overrides
            debug_obs = self.observability.model_copy(update={
                "log_level": LogLevel.DEBUG,
                "log_format": LogFormat.CONSOLE,
                "tracing_enabled": True,
                "tracing_exporter": TracingExporter.CONSOLE,
            })
            object.__setattr__(self, "observability", debug_obs)
        return self

    def is_development(self) -> bool:
        """Indica si el sistema está corriendo en entorno de desarrollo."""
        return self.environment == "development" or self.debug_mode

    def is_production(self) -> bool:
        """Indica si el sistema está corriendo en entorno de producción."""
        return self.environment == "production" and not self.debug_mode

    def get_effective_llm_provider(self) -> LLMProvider:
        """
        Retorna el provider LLM efectivo considerando disponibilidad de credenciales.

        En la mayoría de casos retorna llm.default_provider. Existe como método
        centralizado para que los adapters no necesiten implementar esta lógica.
        """
        return self.llm.default_provider

    def get_platform(self) -> str:
        """
        Retorna el identificador de plataforma normalizado.

        Returns:
            'linux' | 'windows' | 'darwin' | 'unknown'
        """
        platform_map = {
            "linux": "linux",
            "win32": "windows",
            "darwin": "darwin",
        }
        return platform_map.get(sys.platform, "unknown")


# =============================================================================
# FUNCIÓN DE CARGA
# =============================================================================


def load_config(
    *,
    env_file: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> ForgeUserConfig:
    """
    Carga y valida la configuración completa de HiperForge User.

    Esta es la función principal de entrada para obtener la configuración.
    Debe llamarse una vez en el bootstrap del sistema y el resultado debe
    inyectarse como dependencia en todos los componentes que lo necesiten.

    Args:
        env_file:  Ruta explícita a un archivo .env a cargar. Si se provee,
                   tiene mayor precedencia que los archivos .env por defecto.
                   Útil para testing con configuraciones específicas.
        overrides: Diccionario de overrides para testing o fixtures.
                   Las claves usan notación anidada con '__' como separador.
                   Ejemplo: {"llm__default_provider": "local"}

    Returns:
        Instancia validada e inmutable de ForgeUserConfig.

    Raises:
        ForgeConfigurationError: Si la configuración tiene errores de validación.
            El mensaje incluye todos los errores encontrados para facilitar la corrección.

    Example:
        # Carga estándar desde entorno
        config = load_config()

        # Para testing con archivo específico
        config = load_config(env_file=Path("tests/fixtures/.env.test"))

        # Para testing con overrides en memoria
        config = load_config(overrides={"llm__default_provider": "local"})
    """
    from forge_core.errors.types import ForgeConfigurationError

    try:
        init_kwargs: dict[str, Any] = {}

        if overrides:
            init_kwargs.update(overrides)

        if env_file is not None:
            # Pydantic Settings acepta _env_file como kwarg de inicialización
            init_kwargs["_env_file"] = str(env_file)

        return ForgeUserConfig(**init_kwargs)

    except Exception as exc:
        # Pydantic ValidationError tiene un formato rico; lo preservamos
        error_detail = str(exc)
        raise ForgeConfigurationError(
            f"Error de configuración al inicializar HiperForge User: {error_detail}",
            context={"validation_detail": error_detail},
        ) from exc