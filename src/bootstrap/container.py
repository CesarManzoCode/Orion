"""
Contenedor de inyección de dependencias de HiperForge User.

El container es el único lugar en todo el sistema donde se instancian
las implementaciones concretas y se conectan a sus puertos abstractos.
Este es el «composition root» de la arquitectura hexagonal.

Orden de construcción (respeta las dependencias):
  1. Configuración (desde env vars + archivo TOML)
  2. Storage (SQLite stores)
  3. LLM adapter (Anthropic)
  4. Tool registry (vacío — las tools se registran en startup.py después)
  5. Policy engine + Approval workflow
  6. Tool dispatch (requiere policy engine ya construido)
  7. Servicios de aplicación
  8. ConversationCoordinator
  9. AdminPort

El container es un dataclass inmutable después de construirse.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge_core.config.schema import ForgeUserConfig, LLMProvider, load_config
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
from src.infrastructure.llm.openai_adapter import OpenAIAdapter
from src.infrastructure.storage.sqlite_storage import (
    SQLiteArtifactStore,
    SQLiteAuditLog,
    SQLiteMemoryStore,
    SQLiteSessionStore,
    SQLiteUserProfileStore,
    create_sqlite_stores,
)
from src.infrastructure.tools.tool_dispatch import DefaultToolDispatch
from src.infrastructure.tools.tool_registry import InMemoryToolRegistry
from src.ports.inbound.conversation_port import AdminPort, ConversationPort
from src.ports.outbound.llm_port import UserLLMPort


logger = get_logger(__name__, component="bootstrap")


@dataclass(frozen=True)
class AppContainer:
    """
    Contenedor de todas las dependencias del sistema.

    Todos los servicios son singletons — una instancia por proceso.
    Inmutable después de construirse.
    """

    # --- Configuración ---
    config: ForgeUserConfig

    # --- Infraestructura de storage ---
    session_store: SQLiteSessionStore
    artifact_store: SQLiteArtifactStore
    audit_log: SQLiteAuditLog
    profile_store: SQLiteUserProfileStore
    memory_store: SQLiteMemoryStore

    # --- Infraestructura LLM ---
    llm_adapter: UserLLMPort

    # --- Infraestructura de tools ---
    tool_registry: InMemoryToolRegistry
    tool_dispatch: DefaultToolDispatch

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

    # Determinar configuración LLM según provider.
    if config.llm.default_provider == LLMProvider.ANTHROPIC:
        model_name = config.llm.anthropic_model
        api_key = (
            config.llm.anthropic_api_key.get_secret_value()
            if config.llm.anthropic_api_key
            else None
        )
    elif config.llm.default_provider == LLMProvider.OPENAI:
        model_name = config.llm.openai_model
        api_key = (
            config.llm.openai_api_key.get_secret_value()
            if config.llm.openai_api_key
            else None
        )
    else:  # LOCAL
        raise ValueError(
            "El provider LOCAL aún no está implementado en build_container. "
            "Usa FORGE_USER__LLM__DEFAULT_PROVIDER=anthropic por ahora."
        )

    logger.info(
        "config_cargada",
        llm_provider=config.llm.default_provider.value,
        llm_model=model_name,
        storage_path=config.storage.database_path,
        log_level=config.observability.log_level,
    )

    # =========================================================================
    # 2. STORAGE — SQLite
    # =========================================================================

    db_path = db_path_override or Path(config.storage.database_path).expanduser()

    logger.info("storage_inicializando", db_path=str(db_path))

    session_store, artifact_store, audit_log, profile_store, memory_store = (
        await create_sqlite_stores(db_path)
    )

    logger.info("storage_listo")

    # =========================================================================
    # 3. LLM ADAPTER — Anthropic (u otro provider)
    # =========================================================================

    if not api_key:
        raise ValueError(
            f"No se encontró la API key para provider '{config.llm.default_provider.value}'. "
            f"Configura FORGE_USER__LLM__{config.llm.default_provider.value.upper()}_API_KEY."
        )

    if config.llm.default_provider == LLMProvider.ANTHROPIC:
        llm_adapter = AnthropicAdapter(
            api_key=api_key,
            model=model_name,
            max_tokens=config.llm.max_tokens_response,
            timeout_seconds=config.llm.request_timeout_seconds,
            max_retries=config.llm.max_retries,
            base_url=config.llm.anthropic_base_url,
        )
    else:  # OPENAI
        llm_adapter = OpenAIAdapter(
            api_key=api_key,
            model=model_name,
            max_tokens=config.llm.max_tokens_response,
            timeout_seconds=config.llm.request_timeout_seconds,
            max_retries=config.llm.max_retries,
            base_url=config.llm.openai_base_url,
        )

    logger.info("llm_adapter_listo", model=llm_adapter.model_name)

    # =========================================================================
    # 4. TOOL REGISTRY — vacío; startup.py llama register_builtin_tools()
    # =========================================================================

    tool_registry = InMemoryToolRegistry()

    # =========================================================================
    # 5. POLICY ENGINE + APPROVAL WORKFLOW
    # =========================================================================

    policy_engine = DefaultPolicyEngine()
    approval_workflow = DefaultApprovalWorkflow(
        default_timeout_seconds=300.0,
    )

    logger.info(
        "policy_engine_listo",
        policies=len(policy_engine.get_active_policies()),
    )

    # =========================================================================
    # 6. TOOL DISPATCH
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
        inactivity_timeout_minutes=240,
        auto_pause_enabled=True,
    )

    artifact_manager = ArtifactManager(
        artifact_store=artifact_store,
    )

    context_builder = ContextBuilder(
        tool_registry=tool_registry,
        artifact_store=artifact_store,
        max_artifacts_in_context=3,
    )

    memory_manager = MemoryManager(
        llm_port=llm_adapter,
        session_manager=session_manager,
        session_store=session_store,
        memory_store=memory_store,
        profile_store=profile_store,
        long_term_memory_enabled=False,
        max_long_term_facts=500,
    )

    logger.info("servicios_aplicacion_listos")

    # =========================================================================
    # 8. CONVERSATION COORDINATOR
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
        planning_threshold=2,
        max_tool_calls_per_turn=20,
        thinking_indicators_enabled=True,
    )

    # =========================================================================
    # 9. ADMIN PORT
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
        memory_store=memory_store,
        tool_registry=tool_registry,
        tool_dispatch=tool_dispatch,
        policy_engine=policy_engine,
        approval_workflow=approval_workflow,
        session_manager=session_manager,
        artifact_manager=artifact_manager,
        context_builder=context_builder,
        memory_manager=memory_manager,
        conversation_port=coordinator,
        admin_port=admin_port,
    )


def _select_token_budget(context_window: int) -> Any:
    """Selecciona el ContextTokenBudget según el context window del modelo."""
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