"""
Contenedor de inyección de dependencias de HiperForge User.

El container es el único lugar en todo el sistema donde se instancian
las implementaciones concretas y se conectan a sus puertos abstractos.
Este es el «composition root» de la arquitectura hexagonal.

Principio:
  Toda la lógica de negocio depende de abstracciones (puertos).
  El container conecta las abstracciones con las implementaciones concretas.
  Solo el container conoce ambos lados — nadie más.

Orden de construcción (respeta las dependencias):
  1. Configuración (desde env vars + archivo TOML)
  2. Storage (SQLite stores)
  3. LLM adapter (Anthropic)
  4. Tool registry (vacío, herramientas se registran después)
  5. Policy engine + Approval workflow
  6. Servicios de aplicación (SessionManager, MemoryManager, etc.)
  7. ConversationCoordinator (orquestador principal)
  8. ConversationAdminPort (para el debug CLI)

El container es un dataclass inmutable después de construirse.
La construcción es async porque el storage necesita initialize().
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge_core.config.schema import ForgeUserConfig, load_config
from forge_core.observability.logging import get_logger

from src.application.approval.approval_workflow import DefaultApprovalWorkflow
from src.application.artifacts.artifact_manager import ArtifactManager
from src.application.context.context_builder import ContextBuilder
from src.application.coordinator.conversation_coordinator import (
    ConversationAdminPort,
    ConversationCoordinator,
)
from src.application.memory.memory_manager import MemoryManager
from src.application.policy_engine.policy_engine import DefaultPolicyEngine
from src.application.session.session_manager import SessionManager
from src.infrastructure.llm.anthropic_adapter import AnthropicAdapter
from src.infrastructure.storage.sqlite_storage import (
    SQLiteArtifactStore,
    SQLiteAuditLog,
    SQLiteSessionStore,
    SQLiteUserProfileStore,
    create_sqlite_stores,
)
from src.ports.inbound.conversation_port import AdminPort, ConversationPort


logger = get_logger(__name__, component="bootstrap")


@dataclass(frozen=True)
class AppContainer:
    """
    Contenedor de todas las dependencias del sistema.

    Es el registro central de servicios. Todos los servicios son
    singletons — una instancia por proceso. El container se crea
    una vez al arrancar y se pasa a los entry points (Tauri, CLI).

    Inmutable después de construirse: los servicios no se reemplazan
    en runtime. Para cambiar configuración, se reinicia el proceso.
    """

    # --- Configuración ---
    config: ForgeUserConfig

    # --- Infraestructura de storage ---
    session_store: SQLiteSessionStore
    artifact_store: SQLiteArtifactStore
    audit_log: SQLiteAuditLog
    profile_store: SQLiteUserProfileStore

    # --- Infraestructura LLM ---
    llm_adapter: AnthropicAdapter

    # --- Servicios de aplicación ---
    policy_engine: DefaultPolicyEngine
    approval_workflow: DefaultApprovalWorkflow
    session_manager: SessionManager
    artifact_manager: ArtifactManager
    context_builder: ContextBuilder
    memory_manager: MemoryManager

    # --- Puertos de entrada (implementados) ---
    conversation_port: ConversationPort
    admin_port: AdminPort

    @property
    def db_path(self) -> str:
        """Ruta al archivo de base de datos SQLite."""
        return self.session_store._db_path


async def build_container(
    *,
    config_overrides: dict[str, Any] | None = None,
    db_path_override: Path | None = None,
) -> AppContainer:
    """
    Construye el contenedor de dependencias del sistema.

    Inicializa todos los servicios en el orden correcto respetando
    el grafo de dependencias. Es async porque el storage necesita
    crear las tablas SQLite.

    Args:
        config_overrides:  Overrides de configuración (útil para tests).
        db_path_override:  Path alternativo para la BD (útil para tests).

    Returns:
        AppContainer completamente inicializado y listo para usar.

    Raises:
        ValueError:        Si faltan configuraciones requeridas (API key, etc.).
        ForgeStorageError: Si no se puede inicializar el storage.
    """
    logger.info("bootstrap_iniciando")

    # =========================================================================
    # 1. CONFIGURACIÓN
    # =========================================================================

    config = load_config(overrides=config_overrides or {})

    logger.info(
        "config_cargada",
        llm_model=config.llm.model,
        storage_path=config.storage.db_path,
        log_level=config.observability.log_level,
    )

    # =========================================================================
    # 2. STORAGE — SQLite
    # =========================================================================

    db_path = db_path_override or Path(config.storage.db_path).expanduser()

    logger.info("storage_inicializando", db_path=str(db_path))

    session_store, artifact_store, audit_log, profile_store = (
        await create_sqlite_stores(db_path)
    )

    logger.info("storage_listo")

    # =========================================================================
    # 3. LLM ADAPTER — Anthropic
    # =========================================================================

    api_key = config.llm.api_key.get_secret_value() if config.llm.api_key else None
    if not api_key:
        raise ValueError(
            "No se encontró la API key de Anthropic. "
            "Configura FORGE_USER__LLM__API_KEY en las variables de entorno."
        )

    llm_adapter = AnthropicAdapter(
        api_key=api_key,
        model=config.llm.model or "claude-sonnet-4-5",
        max_tokens=config.llm.max_tokens or 8_192,
        timeout_seconds=config.llm.timeout_seconds or 120.0,
        max_retries=config.llm.max_retries or 3,
    )

    logger.info("llm_adapter_listo", model=llm_adapter.model_name)

    # =========================================================================
    # 4. TOOL REGISTRY — construido vacío, las tools se registran en startup.py
    # =========================================================================

    from src.infrastructure.tools.tool_registry import InMemoryToolRegistry
    tool_registry = InMemoryToolRegistry()

    from src.infrastructure.tools.tool_dispatch import DefaultToolDispatch
    # El dispatch se crea sin policy engine aún (se inyecta después)
    # Para romper la dependencia circular, usamos un placeholder que se configura en startup
    tool_dispatch_placeholder: Any = None

    # =========================================================================
    # 5. POLICY ENGINE + APPROVAL WORKFLOW
    # =========================================================================

    policy_engine = DefaultPolicyEngine()
    approval_workflow = DefaultApprovalWorkflow(
        default_timeout_seconds=config.agent.approval_timeout_seconds or 300.0,
    )

    logger.info(
        "policy_engine_listo",
        policies=len(policy_engine.get_active_policies()),
    )

    # =========================================================================
    # 6. TOOL DISPATCH (ahora con policy engine listo)
    # =========================================================================

    tool_dispatch = DefaultToolDispatch(
        tool_registry=tool_registry,
        policy_engine=policy_engine,
        approval_port=approval_workflow,
        audit_log=audit_log,
    )

    # =========================================================================
    # 7. SERVICIOS DE APLICACIÓN
    # =========================================================================

    session_manager = SessionManager(
        session_store=session_store,
        profile_store=profile_store,
        audit_log=audit_log,
        default_token_budget=_select_token_budget(llm_adapter.max_context_tokens),
        inactivity_timeout_minutes=config.agent.session_inactivity_timeout_minutes or 240,
        auto_pause_enabled=True,
    )

    artifact_manager = ArtifactManager(
        artifact_store=artifact_store,
    )

    context_builder = ContextBuilder(
        tool_registry=tool_registry,
        artifact_store=artifact_store,
        max_artifacts_in_context=config.memory.max_artifacts_in_context or 3,
    )

    memory_manager = MemoryManager(
        llm_port=llm_adapter,
        session_manager=session_manager,
        session_store=session_store,
        long_term_memory_enabled=config.memory.long_term_memory_enabled,
        max_long_term_facts=config.memory.max_long_term_facts or 500,
    )

    logger.info("servicios_aplicacion_listos")

    # =========================================================================
    # 8. CONVERSATION COORDINATOR (orquestador principal)
    # =========================================================================

    coordinator = ConversationCoordinator(
        llm_port=llm_adapter,
        session_store=session_store,
        artifact_store=artifact_store,
        audit_log=audit_log,
        profile_store=profile_store,
        tool_registry=tool_registry,
        tool_dispatch=tool_dispatch,
        policy_engine=policy_engine,
        approval_port=approval_workflow,
        planning_threshold=config.agent.planning_threshold or 2,
        max_tool_calls_per_turn=config.agent.max_tool_calls_per_turn or 20,
        thinking_indicators_enabled=True,
    )

    # =========================================================================
    # 9. ADMIN PORT (para el debug CLI)
    # =========================================================================

    admin_port = ConversationAdminPort(
        session_store=session_store,
        artifact_store=artifact_store,
        audit_log=audit_log,
        tool_registry=tool_registry,
        policy_engine=policy_engine,
    )

    logger.info("bootstrap_completado")

    return AppContainer(
        config=config,
        session_store=session_store,
        artifact_store=artifact_store,
        audit_log=audit_log,
        profile_store=profile_store,
        llm_adapter=llm_adapter,
        policy_engine=policy_engine,
        approval_workflow=approval_workflow,
        session_manager=session_manager,
        artifact_manager=artifact_manager,
        context_builder=context_builder,
        memory_manager=memory_manager,
        conversation_port=coordinator,
        admin_port=admin_port,
    )


def _select_token_budget(context_window: int):
    """
    Selecciona el ContextTokenBudget apropiado según el context window del modelo.

    Args:
        context_window: Context window del modelo activo en tokens.

    Returns:
        ContextTokenBudget configurado para ese context window.
    """
    from src.domain.value_objects.token_budget import (
        BUDGET_8K,
        BUDGET_32K,
        BUDGET_128K,
        BUDGET_200K,
    )
    if context_window >= 200_000:
        return BUDGET_200K
    if context_window >= 128_000:
        return BUDGET_128K
    if context_window >= 32_000:
        return BUDGET_32K
    return BUDGET_8K