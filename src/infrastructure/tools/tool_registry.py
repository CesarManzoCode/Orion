"""
InMemoryToolRegistry — implementación en memoria del catálogo de tools.

El ToolRegistry es el catálogo central de todas las tools disponibles
en el sistema. En V1 se mantiene completamente en memoria porque:
  - Las tools se registran una vez en el bootstrap y no cambian
  - El acceso es frecuente (cada turno del LLM necesita los schemas)
  - El volumen es pequeño (< 50 tools en V1)

El registry es thread-safe para asyncio — todas las operaciones son
síncronas y atómicas desde la perspectiva del event loop.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from forge_core.llm.protocol import ToolDefinition
from forge_core.observability.logging import get_logger
from forge_core.tools.protocol import (
    MutationType,
    Platform,
    RiskLevel,
    ToolCategory,
    ToolSchema,
)

from src.domain.value_objects.permission import PermissionSet
from src.domain.value_objects.token_budget import TokenEstimator
from src.ports.outbound.tool_port import (
    ToolFilter,
    ToolRegistration,
    ToolRegistryPort,
)


logger = get_logger(__name__, component="tool_registry")


class InMemoryToolRegistry(ToolRegistryPort):
    """
    Implementación en memoria del catálogo de tools.

    Mantiene un dict {tool_id → ToolRegistration} y un índice secundario
    {llm_name → tool_id} para resolución rápida del nombre que usa el LLM.

    Las tools se registran durante el bootstrap con register() y permanecen
    estáticas durante toda la vida del proceso (V1).
    """

    def __init__(self) -> None:
        # Catálogo principal: tool_id → ToolRegistration
        self._catalog: dict[str, ToolRegistration] = {}

        # Índice por llm_name: llm_name → tool_id
        # El LLM puede usar nombres distintos al tool_id interno
        self._by_llm_name: dict[str, str] = {}

        # Caché de schemas para el LLM (se invalida al registrar tools)
        self._llm_definitions_cache: list[ToolDefinition] | None = None

        # Caché de token count estimado para los schemas
        self._token_count_cache: int | None = None

    # =========================================================================
    # ToolRegistryPort — implementación
    # =========================================================================

    def register(self, schema: ToolSchema) -> None:
        """
        Registra una tool en el catálogo.

        Si ya existe una tool con el mismo tool_id, la reemplaza.
        Invalida el caché de schemas del LLM.

        Args:
            schema: Schema completo de la tool a registrar.
        """
        registration = ToolRegistration(
            schema=schema,
            registered_at=datetime.now(tz=timezone.utc),
        )

        self._catalog[schema.tool_id] = registration

        # Indexar por llm_name (puede diferir del tool_id)
        llm_name = getattr(schema, "llm_name", schema.tool_id)
        self._by_llm_name[llm_name] = schema.tool_id

        # Invalida caché
        self._llm_definitions_cache = None
        self._token_count_cache = None

        logger.debug(
            "tool_registrada",
            tool_id=schema.tool_id,
            category=schema.category.value,
            risk_level=schema.capability.risk_classification.name,
        )

    def unregister(self, tool_id: str) -> bool:
        """Elimina una tool del catálogo por su ID."""
        registration = self._catalog.pop(tool_id, None)
        if registration is None:
            return False

        # Limpiar índice por llm_name
        llm_name = getattr(registration.schema, "llm_name", tool_id)
        self._by_llm_name.pop(llm_name, None)

        # Invalida caché
        self._llm_definitions_cache = None
        self._token_count_cache = None

        return True

    def get(self, tool_id: str) -> ToolRegistration | None:
        """Obtiene el registro de una tool por su ID."""
        return self._catalog.get(tool_id)

    def get_by_llm_name(self, llm_name: str) -> ToolRegistration | None:
        """
        Obtiene el registro de una tool por el nombre que el LLM usa.

        Primero busca en el índice llm_name → tool_id.
        Si no encuentra, intenta con tool_id directo (fallback).

        Args:
            llm_name: Nombre de la tool tal como el LLM la invocó.

        Returns:
            ToolRegistration si existe, None si no.
        """
        # Busca en el índice de llm_name
        tool_id = self._by_llm_name.get(llm_name)
        if tool_id:
            return self._catalog.get(tool_id)

        # Fallback: buscar directamente por tool_id
        return self._catalog.get(llm_name)

    def list(self, filter: ToolFilter | None = None) -> list[ToolRegistration]:
        """
        Lista las tools del catálogo aplicando el filtro.

        Args:
            filter: Criterios de filtrado. None retorna todas.

        Returns:
            Lista de ToolRegistration que cumplen los criterios,
            ordenada por categoría y luego por tool_id.
        """
        registrations = list(self._catalog.values())

        if filter is None:
            return sorted(registrations, key=lambda r: (r.category.value, r.tool_id))

        result = []
        for reg in registrations:
            if not self._matches_filter(reg, filter):
                continue
            result.append(reg)

        return sorted(result, key=lambda r: (r.category.value, r.tool_id))

    def get_schemas_for_llm(
        self,
        platform: Platform,
        permissions: PermissionSet,
    ) -> list[ToolDefinition]:
        """
        Retorna los schemas de tools disponibles para enviar al LLM.

        Solo incluye tools:
          - Habilitadas
          - Compatibles con la plataforma actual
          - Permitidas por el PermissionSet del usuario

        Usa caché porque este método se llama en cada turno del LLM.

        Args:
            platform:    Plataforma actual del sistema.
            permissions: PermissionSet del usuario activo.

        Returns:
            Lista de ToolDefinition ordenada consistentemente.
        """
        # Construir la lista filtrada (sin caché porque depende de permissions)
        definitions = []
        for reg in self._catalog.values():
            if not reg.schema.enabled:
                continue
            if not self._compatible_with_platform(reg.schema, platform):
                continue
            if not permissions.allows(
                tool_id=reg.tool_id,
                category=reg.schema.category,
            ):
                continue
            if not reg.is_healthy:
                logger.warning(
                    "tool_excluida_unhealthy",
                    tool_id=reg.tool_id,
                    failure_count=reg.failure_count,
                )
                continue

            definitions.append(reg.to_llm_definition())

        # Orden consistente para hashability del contexto
        return sorted(definitions, key=lambda d: d.name)

    def estimate_schemas_token_count(
        self,
        platform: Platform,
        permissions: PermissionSet,
    ) -> int:
        """
        Estima los tokens de los schemas de tools disponibles.

        Recalcula cuando el catálogo cambia. En V1 el catálogo es
        estático después del bootstrap, por lo que esto es muy eficiente.

        Args:
            platform:    Plataforma actual.
            permissions: PermissionSet del usuario.

        Returns:
            Estimación de tokens de todos los schemas disponibles.
        """
        schemas = self.get_schemas_for_llm(platform, permissions)
        if not schemas:
            return 0

        # Estimar tokens serializando los schemas como JSON
        import json
        schemas_json = json.dumps(
            [{"name": s.name, "description": s.description, "parameters": s.parameters}
             for s in schemas],
            ensure_ascii=False,
        )
        return TokenEstimator.estimate(schemas_json)

    def get_mutation_tool_ids(self) -> frozenset[str]:
        """
        Retorna los tool_ids de tools que producen mutación del sistema.

        Usado por el LLMResponseAnalyzer para clasificar el routing
        de respuestas del LLM.

        Returns:
            frozenset de tool_ids con MutationType != NONE.
        """
        return frozenset(
            reg.tool_id
            for reg in self._catalog.values()
            if reg.schema.capability.mutation_type != MutationType.NONE
        )

    def record_invocation(
        self,
        tool_id: str,
        *,
        success: bool,
    ) -> None:
        """
        Registra una invocación de tool para estadísticas y circuit breaking.

        Actualiza invocation_count, last_invoked_at, y failure_count.
        Si falla más de N veces, is_healthy retorna False (circuit breaker).

        Args:
            tool_id: ID de la tool invocada.
            success: True si la invocación fue exitosa.
        """
        reg = self._catalog.get(tool_id)
        if reg is None:
            return

        now = datetime.now(tz=timezone.utc)

        # ToolRegistration es frozen — crear nueva instancia con valores actualizados
        new_failure_count = reg.failure_count if success else reg.failure_count + 1
        # Reset failure_count si fue exitosa
        if success and reg.failure_count > 0:
            new_failure_count = 0

        updated = ToolRegistration(
            schema=reg.schema,
            registered_at=reg.registered_at,
            invocation_count=reg.invocation_count + 1,
            last_invoked_at=now,
            failure_count=new_failure_count,
        )
        self._catalog[tool_id] = updated

    def is_registered(self, tool_id: str) -> bool:
        """Verifica si una tool está registrada en el catálogo."""
        return tool_id in self._catalog

    @property
    def registered_count(self) -> int:
        """Número total de tools registradas."""
        return len(self._catalog)

    @property
    def enabled_count(self) -> int:
        """Número de tools habilitadas."""
        return sum(1 for r in self._catalog.values() if r.schema.enabled)

    # =========================================================================
    # HELPERS INTERNOS
    # =========================================================================

    def _matches_filter(
        self,
        registration: ToolRegistration,
        filter: ToolFilter,
    ) -> bool:
        """
        Verifica si una ToolRegistration cumple los criterios del filtro.

        Args:
            registration: ToolRegistration a verificar.
            filter:        Criterios de filtrado.

        Returns:
            True si la tool cumple todos los criterios.
        """
        schema = registration.schema

        # Filtro enabled_only
        if filter.enabled_only and not schema.enabled:
            return False

        # Filtro por categorías
        if filter.categories is not None:
            if schema.category not in filter.categories:
                return False

        # Filtro por nivel de riesgo máximo
        if filter.max_risk_level is not None:
            if registration.risk_level > filter.max_risk_level:
                return False

        # Filtro por plataforma
        if filter.platform is not None:
            if not self._compatible_with_platform(schema, filter.platform):
                return False

        # Filtro por permisos del usuario
        if filter.allowed_by_permissions is not None:
            if not filter.allowed_by_permissions.allows(
                tool_id=registration.tool_id,
                category=schema.category,
            ):
                return False

        return True

    @staticmethod
    def _compatible_with_platform(schema: ToolSchema, platform: Platform) -> bool:
        """
        Verifica si una tool es compatible con la plataforma actual.

        Una tool es compatible si:
          - Su plataforma es Platform.ALL (disponible en todas)
          - Su plataforma coincide exactamente con la actual

        Args:
            schema:   Schema de la tool.
            platform: Plataforma actual del sistema.

        Returns:
            True si la tool es compatible.
        """
        tool_platform = getattr(schema.capability, "platform", Platform.ALL)
        if tool_platform == Platform.ALL:
            return True
        return tool_platform == platform

    def get_stats(self) -> dict[str, Any]:
        """
        Retorna estadísticas del registry para el admin CLI y diagnóstico.

        Returns:
            Dict con estadísticas del catálogo.
        """
        by_category: dict[str, int] = {}
        by_risk: dict[str, int] = {}
        unhealthy: list[str] = []

        for reg in self._catalog.values():
            cat = reg.category.value
            by_category[cat] = by_category.get(cat, 0) + 1
            risk = reg.risk_level.name
            by_risk[risk] = by_risk.get(risk, 0) + 1
            if not reg.is_healthy:
                unhealthy.append(reg.tool_id)

        return {
            "total": self.registered_count,
            "enabled": self.enabled_count,
            "disabled": self.registered_count - self.enabled_count,
            "by_category": by_category,
            "by_risk_level": by_risk,
            "unhealthy": unhealthy,
            "mutation_tools": len(self.get_mutation_tool_ids()),
        }