"""
Modelo de permisos del dominio HiperForge User.

Este módulo define el sistema de control de acceso basado en capacidades
(capability-based access control) del agente. Es la capa de permisos que
el PolicyEngine consulta para verificar si el usuario autorizó el acceso
a un recurso o categoría de tools.

Principios de diseño:
  1. Default-deny: si no hay un permiso explícito que autorice una acción,
     está denegada. Nunca se infiere autorización desde la ausencia de denegación.
  2. Granularidad por scope: los permisos pueden ser por categoría (RESEARCH,
     FILESYSTEM), por tool específica (file_write), o por recurso concreto
     (un path específico).
  3. Permisos explícitos tienen precedencia sobre permisos de categoría.
     Un permiso DENY específico anula un ALLOW de categoría.
  4. El PermissionSet es inmutable. Cambiar permisos requiere crear uno nuevo.
     Esto garantiza que las evaluaciones de policy son consistentes durante
     el ciclo de vida de una sesión.
  5. Los permisos de primera vez (first_use_approval) se almacenan en el
     PermissionSet para que ApprovalWorkflow no vuelva a preguntar.

Modelo de permisos en capas (de menor a mayor especificidad):

  Capa 1: Permisos de categoría
    RESEARCH → FULL, FILESYSTEM → READ, DESKTOP → LIMITED

  Capa 2: Permisos de tool específica
    file_write → EXECUTE overrides FILESYSTEM → READ

  Capa 3: Permisos de recurso concreto
    path:/home/user/docs → READ overrides file_read general

  Capa 4: Allowlists y blocklists
    app_allowlist: [firefox, libreoffice]
    blocked_paths: [/etc/shadow, ~/.ssh/*]

El PolicyEngine evalúa las capas en orden de especificidad — la más
específica gana. Esto es el principio del mínimo privilegio aplicado:
las reglas más estrechas siempre tienen precedencia.

Jerarquía de tipos:

  AccessLevel          — nivel de acceso (DENY, READ, EXECUTE, FULL)
  PermissionScope      — scope de un permiso (category, tool, resource)
  Permission           — un permiso individual: scope + target + level
  FilesystemPermissions — permisos granulares de filesystem
  DesktopPermissions   — permisos de automatización de escritorio
  NetworkPermissions   — permisos de acceso a red
  PermissionSet        — conjunto completo de permisos del usuario
"""

from __future__ import annotations

import fnmatch
import sys
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any

from forge_core.tools.protocol import ToolCategory


# =============================================================================
# NIVEL DE ACCESO
# =============================================================================


class AccessLevel(IntEnum):
    """
    Nivel de acceso a un recurso o capacidad.

    Hereda de IntEnum para comparaciones ordinales directas:
        AccessLevel.READ < AccessLevel.EXECUTE → True
        AccessLevel.FULL >= AccessLevel.EXECUTE → True

    Los niveles son acumulativos hacia arriba: EXECUTE implica READ,
    FULL implica EXECUTE y READ.
    """

    DENY = 0
    """
    Acceso explícitamente denegado. Tiene la máxima precedencia cuando
    es un permiso específico — anula permisos más generales de mayor nivel.
    """

    READ = 1
    """
    Acceso de solo lectura. Permite consultar, listar y leer recursos
    sin modificarlos. Seguro para retry y sin efectos secundarios visibles.
    """

    EXECUTE = 2
    """
    Permite ejecutar acciones y mutations en el scope del permiso.
    Implica READ. Requiere aprobación del usuario en primer uso.
    """

    FULL = 3
    """
    Acceso completo sin restricciones dentro del scope.
    Implica EXECUTE y READ. Reservado para categorías de total confianza.
    """

    def can_read(self) -> bool:
        """True si el nivel de acceso permite lectura."""
        return self >= AccessLevel.READ

    def can_execute(self) -> bool:
        """True si el nivel de acceso permite ejecución/mutación."""
        return self >= AccessLevel.EXECUTE

    def is_denied(self) -> bool:
        """True si el acceso está explícitamente denegado."""
        return self == AccessLevel.DENY

    def label(self) -> str:
        """Retorna el label legible del nivel de acceso."""
        labels = {
            AccessLevel.DENY: "denegado",
            AccessLevel.READ: "solo lectura",
            AccessLevel.EXECUTE: "lectura y ejecución",
            AccessLevel.FULL: "acceso completo",
        }
        return labels[self]


# =============================================================================
# SCOPE DE PERMISO
# =============================================================================


class PermissionScope(Enum):
    """
    Alcance de un permiso individual.

    Determina a qué nivel de granularidad aplica el permiso y cómo
    el PolicyEngine lo compara con la acción solicitada.
    """

    CATEGORY = "category"
    """
    El permiso aplica a toda una categoría de tools (ToolCategory).
    Es el nivel más amplio — afecta a todas las tools de esa categoría.
    Ejemplo: FILESYSTEM → READ permite todas las tools de lectura de filesystem.
    """

    TOOL = "tool"
    """
    El permiso aplica a una tool específica por su tool_id.
    Más específico que CATEGORY — puede ampliar o restringir el permiso de categoría.
    Ejemplo: file_write → EXECUTE overrides FILESYSTEM → READ.
    """

    RESOURCE = "resource"
    """
    El permiso aplica a un recurso concreto (path, URL, app identifier).
    El nivel más específico — overrides TOOL y CATEGORY.
    Ejemplo: path:/home/user/projects → FULL para ese directorio específico.
    """


# =============================================================================
# PERMISO INDIVIDUAL
# =============================================================================


@dataclass(frozen=True)
class Permission:
    """
    Un permiso individual: scope + target + access level.

    Los permisos son los bloques de construcción del PermissionSet.
    Cada permiso describe una autorización específica del usuario para
    acceder a una categoría, tool o recurso.

    Los permisos son inmutables. El PermissionSet los almacena como
    frozenset para garantizar la consistencia durante una sesión.
    """

    scope: PermissionScope
    """Alcance del permiso (categoría, tool o recurso específico)."""

    target: str
    """
    Identificador del target del permiso. Su semántica depende del scope:
    - CATEGORY: nombre del ToolCategory enum (ej: 'research', 'filesystem')
    - TOOL: tool_id (ej: 'file_write', 'web_search')
    - RESOURCE: identificador del recurso (ej: 'path:/home/user/docs',
                'app:firefox', 'domain:tavily.com')
    """

    access: AccessLevel
    """Nivel de acceso otorgado o denegado."""

    granted_at: str | None = field(
        default=None,
        compare=False,
        hash=False,
    )
    """
    Timestamp ISO 8601 de cuándo se otorgó el permiso.
    No participa en igualdad ni hash — es solo metadata de auditoría.
    """

    granted_by: str = field(
        default="user",
        compare=False,
        hash=False,
    )
    """
    Quién otorgó el permiso: 'user' (aprobación explícita), 'default' (default config),
    'system' (política del sistema). No participa en igualdad ni hash.
    """

    @classmethod
    def allow_category(
        cls,
        category: ToolCategory,
        access: AccessLevel = AccessLevel.FULL,
        *,
        granted_by: str = "default",
    ) -> Permission:
        """
        Factory: crea un permiso de acceso para toda una categoría.

        Args:
            category:   Categoría de tools a permitir.
            access:     Nivel de acceso otorgado (por defecto FULL).
            granted_by: Origen del permiso.

        Returns:
            Permission de scope CATEGORY.
        """
        return cls(
            scope=PermissionScope.CATEGORY,
            target=category.value,
            access=access,
            granted_by=granted_by,
        )

    @classmethod
    def deny_category(
        cls,
        category: ToolCategory,
        *,
        granted_by: str = "default",
    ) -> Permission:
        """
        Factory: crea un permiso de denegación para toda una categoría.

        Args:
            category:   Categoría de tools a denegar.
            granted_by: Origen de la denegación.

        Returns:
            Permission de scope CATEGORY con AccessLevel.DENY.
        """
        return cls(
            scope=PermissionScope.CATEGORY,
            target=category.value,
            access=AccessLevel.DENY,
            granted_by=granted_by,
        )

    @classmethod
    def allow_tool(
        cls,
        tool_id: str,
        access: AccessLevel = AccessLevel.EXECUTE,
        *,
        granted_by: str = "user",
    ) -> Permission:
        """
        Factory: crea un permiso de acceso para una tool específica.

        Args:
            tool_id:    ID de la tool a permitir.
            access:     Nivel de acceso (por defecto EXECUTE para tools).
            granted_by: Origen del permiso.

        Returns:
            Permission de scope TOOL.
        """
        return cls(
            scope=PermissionScope.TOOL,
            target=tool_id,
            access=access,
            granted_by=granted_by,
        )

    @classmethod
    def allow_resource(
        cls,
        resource_type: str,
        resource_id: str,
        access: AccessLevel = AccessLevel.READ,
        *,
        granted_by: str = "user",
    ) -> Permission:
        """
        Factory: crea un permiso de acceso para un recurso concreto.

        El target se construye como '{resource_type}:{resource_id}',
        por ejemplo: 'path:/home/user/docs', 'app:firefox', 'domain:tavily.com'.

        Args:
            resource_type: Tipo de recurso ('path', 'app', 'domain').
            resource_id:   Identificador del recurso.
            access:        Nivel de acceso.
            granted_by:    Origen del permiso.

        Returns:
            Permission de scope RESOURCE.
        """
        return cls(
            scope=PermissionScope.RESOURCE,
            target=f"{resource_type}:{resource_id}",
            access=access,
            granted_by=granted_by,
        )

    def __str__(self) -> str:
        return f"Permission({self.scope.value}:{self.target} → {self.access.label()})"


# =============================================================================
# PERMISOS GRANULARES POR SUBSISTEMA
# =============================================================================


@dataclass(frozen=True)
class FilesystemPermissions:
    """
    Permisos granulares de acceso al sistema de archivos local.

    Complementa el sistema de permisos general con reglas específicas
    de filesystem: directorios permitidos, patrones bloqueados, y
    límites de tamaño de archivo.
    """

    allowed_directories: tuple[Path, ...]
    """
    Directorios donde el agente puede leer y (si tiene EXECUTE) escribir.
    El agente no puede acceder a paths fuera de estos directorios sin
    aprobación explícita del usuario.
    """

    blocked_path_patterns: tuple[str, ...]
    """
    Patrones glob de paths que NUNCA son accesibles, independientemente
    de los directorios permitidos. Se evalúan después del DomainRiskClassifier.
    """

    max_read_size_mb: int = 50
    """Tamaño máximo de archivo que el agente puede leer, en MB."""

    max_write_size_mb: int = 10
    """Tamaño máximo de archivo que el agente puede escribir, en MB."""

    allow_hidden_files: bool = False
    """
    Si True, el agente puede acceder a archivos ocultos (prefijo '.')
    dentro de los directorios permitidos (excepto los bloqueados).
    False por defecto — los archivos ocultos frecuentemente contienen config sensible.
    """

    def is_path_allowed(self, path: Path) -> bool:
        """
        Verifica si un path está dentro de los directorios permitidos.

        La verificación es por ancestro: /home/user/docs/report.pdf está
        permitido si /home/user/docs o /home/user está en allowed_directories.

        Args:
            path: Path a verificar (debe ser absoluto).

        Returns:
            True si el path está dentro de algún directorio permitido.
        """
        resolved = path.resolve()
        return any(
            _is_subpath(resolved, allowed_dir)
            for allowed_dir in self.allowed_directories
        )

    def is_path_blocked(self, path: Path) -> bool:
        """
        Verifica si un path coincide con algún patrón bloqueado.

        Args:
            path: Path a verificar.

        Returns:
            True si el path debe ser bloqueado por política de filesystem.
        """
        path_str = str(path).lower()
        for pattern in self.blocked_path_patterns:
            if fnmatch.fnmatch(path_str, pattern.lower()):
                return True
            # También verificar contra cada parte del path
            for part in path.parts:
                if fnmatch.fnmatch(part.lower(), pattern.lower().split("/")[-1]):
                    return True
        return False

    def get_access_for_path(self, path: Path) -> AccessLevel:
        """
        Retorna el nivel de acceso efectivo para un path específico.

        Lógica:
          1. Si el path coincide con un patrón bloqueado → DENY
          2. Si el path está en allowed_directories → READ (mínimo)
          3. En cualquier otro caso → DENY

        Args:
            path: Path a verificar.

        Returns:
            AccessLevel efectivo para ese path.
        """
        if self.is_path_blocked(path):
            return AccessLevel.DENY
        if self.is_path_allowed(path):
            return AccessLevel.READ
        return AccessLevel.DENY


@dataclass(frozen=True)
class DesktopPermissions:
    """
    Permisos de automatización de escritorio.

    Controla qué aplicaciones puede lanzar el agente, si puede acceder
    al portapapeles, y si puede tomar screenshots.
    """

    app_allowlist: frozenset[str]
    """
    Conjunto de identificadores de aplicaciones que el agente puede lanzar.
    Los identificadores son nombres de ejecutable en minúsculas, sin extensión.
    Ejemplo: {'firefox', 'libreoffice-writer', 'code', 'calculator'}.
    """

    clipboard_read_allowed: bool = False
    """
    True si el agente puede leer el contenido del portapapeles.
    False por defecto — requiere aprobación explícita del usuario.
    """

    clipboard_write_allowed: bool = False
    """
    True si el agente puede escribir en el portapapeles.
    False por defecto — requiere aprobación explícita.
    """

    screenshot_allowed: bool = True
    """True si el agente puede tomar screenshots del escritorio."""

    notifications_allowed: bool = True
    """True si el agente puede enviar notificaciones del sistema."""

    def can_launch_app(self, app_identifier: str) -> bool:
        """
        Verifica si el agente puede lanzar una aplicación específica.

        La verificación es case-insensitive y también verifica prefijos
        comunes (ej: 'libreoffice' cubre 'libreoffice-writer').

        Args:
            app_identifier: Identificador de la aplicación a verificar.

        Returns:
            True si la aplicación está en la allowlist.
        """
        app_lower = app_identifier.lower().strip()

        # Verificación exacta
        if app_lower in self.app_allowlist:
            return True

        # Verificación por prefijo (ej: 'libreoffice' → 'libreoffice-writer')
        return any(
            app_lower.startswith(allowed) or allowed.startswith(app_lower)
            for allowed in self.app_allowlist
        )


@dataclass(frozen=True)
class NetworkPermissions:
    """
    Permisos de acceso a red para tools de búsqueda y fetch.

    Permite restringir qué dominios puede contactar el agente,
    útil en entornos corporativos o de alta privacidad.
    """

    allowed_domains: tuple[str, ...]
    """
    Dominios permitidos para búsqueda web y fetch. El patrón '*' permite
    todos los dominios. Soporta wildcards: '*.openai.com', '*.anthropic.com'.
    """

    blocked_domains: tuple[str, ...]
    """
    Dominios explícitamente bloqueados, independientemente de allowed_domains.
    Los dominios bloqueados tienen precedencia sobre los permitidos.
    """

    def is_domain_allowed(self, domain: str) -> bool:
        """
        Verifica si un dominio está permitido según las reglas de red.

        Lógica:
          1. Si coincide con blocked_domains → False (bloqueado tiene precedencia)
          2. Si allowed_domains es ['*'] → True (permitir todo)
          3. Si coincide con allowed_domains → True
          4. En cualquier otro caso → False

        Args:
            domain: Nombre de dominio a verificar (ej: 'api.openai.com').

        Returns:
            True si el dominio está permitido.
        """
        domain_lower = domain.lower()

        # Los bloqueados tienen siempre precedencia
        for blocked in self.blocked_domains:
            if fnmatch.fnmatch(domain_lower, blocked.lower()):
                return False

        # Si la allowlist tiene '*', permitir todo lo no bloqueado
        if "*" in self.allowed_domains:
            return True

        # Verificar contra la allowlist
        return any(
            fnmatch.fnmatch(domain_lower, allowed.lower())
            for allowed in self.allowed_domains
        )


# =============================================================================
# CONJUNTO DE PERMISOS
# =============================================================================


@dataclass(frozen=True)
class PermissionSet:
    """
    Conjunto completo de permisos del usuario para una sesión.

    El PermissionSet es el objeto central del control de acceso. El PolicyEngine
    lo consulta para cada ProposedAction para determinar si el usuario tiene
    los permisos necesarios.

    Es inmutable por diseño. Los cambios de permisos (ej: usuario aprueba
    una tool en primer uso) crean un nuevo PermissionSet mediante with_permission().
    Esto garantiza que las evaluaciones de policy durante una sesión son
    consistentes — no hay cambios de permisos a mitad de la evaluación.

    Arquitectura de evaluación:
      1. Verificar permisos de recurso específico (máxima especificidad)
      2. Verificar permisos de tool específica
      3. Verificar permisos de categoría
      4. Verificar subsistemas específicos (filesystem, desktop, network)
      5. Si ninguno aplica → DENY (default-deny)
    """

    # --- Permisos generales (capas 1-3) ---
    permissions: frozenset[Permission]
    """
    Conjunto de permisos individuales del usuario.
    Immutable — se usa frozenset para garantizar consistencia de hash.
    """

    # --- Subsistemas específicos ---
    filesystem: FilesystemPermissions
    """Permisos granulares de filesystem."""

    desktop: DesktopPermissions
    """Permisos de automatización de escritorio."""

    network: NetworkPermissions
    """Permisos de acceso a red."""

    # --- Metadata ---
    profile_id: str = "default"
    """Identificador del perfil de usuario que define estos permisos."""

    version: int = 1
    """
    Versión del PermissionSet. Se incrementa cada vez que se crea un nuevo
    PermissionSet por cambio de permisos (ej: first_use approval).
    Usado para invalidar caché del PolicyEngine si aplica.
    """

    def get_category_access(self, category: ToolCategory) -> AccessLevel:
        """
        Retorna el nivel de acceso para una categoría de tools.

        Busca el permiso de scope CATEGORY que corresponde a esta categoría.
        Si no hay permiso explícito, retorna DENY (default-deny).

        Args:
            category: Categoría a verificar.

        Returns:
            AccessLevel efectivo para la categoría.
        """
        for perm in self.permissions:
            if (
                perm.scope == PermissionScope.CATEGORY
                and perm.target == category.value
            ):
                return perm.access
        return AccessLevel.DENY

    def get_tool_access(self, tool_id: str) -> AccessLevel | None:
        """
        Retorna el nivel de acceso específico para una tool.

        Si hay un permiso de scope TOOL para este tool_id, lo retorna.
        Si no hay permiso específico, retorna None (indicando que se debe
        usar el permiso de categoría).

        Args:
            tool_id: ID de la tool a verificar.

        Returns:
            AccessLevel si hay permiso específico, None si heredar de categoría.
        """
        for perm in self.permissions:
            if perm.scope == PermissionScope.TOOL and perm.target == tool_id:
                return perm.access
        return None

    def get_resource_access(
        self,
        resource_type: str,
        resource_id: str,
    ) -> AccessLevel | None:
        """
        Retorna el nivel de acceso para un recurso concreto.

        Busca un permiso de scope RESOURCE con target '{resource_type}:{resource_id}'.
        Si no existe, retorna None (indicando que se debe usar el permiso de tool
        o categoría).

        Args:
            resource_type: Tipo de recurso ('path', 'app', 'domain').
            resource_id:   Identificador del recurso.

        Returns:
            AccessLevel si hay permiso específico, None si heredar del nivel superior.
        """
        target = f"{resource_type}:{resource_id}"
        for perm in self.permissions:
            if perm.scope == PermissionScope.RESOURCE and perm.target == target:
                return perm.access
        return None

    def allows(
        self,
        tool_id: str,
        category: ToolCategory,
        *,
        required_level: AccessLevel = AccessLevel.EXECUTE,
    ) -> bool:
        """
        Determina si el usuario tiene acceso suficiente para ejecutar una tool.

        Evalúa en orden de especificidad (mayor especificidad primero):
          1. Permiso de tool específica (scope=TOOL)
          2. Permiso de categoría (scope=CATEGORY)
          3. Default: DENY

        Args:
            tool_id:        ID de la tool a verificar.
            category:       Categoría de la tool.
            required_level: Nivel de acceso mínimo requerido.

        Returns:
            True si el usuario tiene acceso >= required_level para la tool.
        """
        # Nivel 1: permiso específico de tool (mayor especificidad)
        tool_access = self.get_tool_access(tool_id)
        if tool_access is not None:
            return tool_access >= required_level

        # Nivel 2: permiso de categoría
        category_access = self.get_category_access(category)
        return category_access >= required_level

    def allows_path(
        self,
        path: Path,
        *,
        for_writing: bool = False,
    ) -> bool:
        """
        Verifica si el usuario tiene acceso a un path del filesystem.

        Consulta FilesystemPermissions para la verificación de acceso.
        Si for_writing=True, verifica que hay permiso EXECUTE además de
        que el path esté en los directorios permitidos.

        Args:
            path:        Path a verificar.
            for_writing: True si se necesita acceso de escritura.

        Returns:
            True si el acceso al path está autorizado.
        """
        path_access = self.filesystem.get_access_for_path(path)

        if for_writing:
            # Escritura requiere EXECUTE en filesystem + el path debe estar permitido
            fs_category = self.get_category_access(ToolCategory.FILESYSTEM)
            tool_access = self.get_tool_access("file_write")
            effective = tool_access if tool_access is not None else fs_category
            return path_access >= AccessLevel.READ and effective >= AccessLevel.EXECUTE

        return path_access >= AccessLevel.READ

    def allows_app_launch(self, app_identifier: str) -> bool:
        """
        Verifica si el usuario ha autorizado el lanzamiento de una aplicación.

        Args:
            app_identifier: Identificador de la aplicación.

        Returns:
            True si la aplicación está en la allowlist de desktop.
        """
        return self.desktop.can_launch_app(app_identifier)

    def allows_domain(self, domain: str) -> bool:
        """
        Verifica si el acceso a un dominio web está autorizado.

        Args:
            domain: Dominio a verificar.

        Returns:
            True si el dominio está permitido por las reglas de red.
        """
        return self.network.is_domain_allowed(domain)

    def has_first_use_approval(self, tool_id: str) -> bool:
        """
        Verifica si el usuario ya aprobó el uso de una tool en primer uso.

        Cuando una tool requiere first_use_approval y el usuario la aprueba,
        se añade un permiso TOOL → EXECUTE con granted_by='user' al PermissionSet.
        Este método verifica la presencia de ese permiso.

        Args:
            tool_id: ID de la tool a verificar.

        Returns:
            True si ya existe un permiso de usuario para esa tool.
        """
        for perm in self.permissions:
            if (
                perm.scope == PermissionScope.TOOL
                and perm.target == tool_id
                and perm.granted_by == "user"
                and perm.access >= AccessLevel.EXECUTE
            ):
                return True
        return False

    def with_permission(self, new_permission: Permission) -> PermissionSet:
        """
        Crea un nuevo PermissionSet añadiendo o actualizando un permiso.

        Si ya existe un permiso con el mismo scope y target, lo reemplaza
        con el nuevo. Si no existe, lo añade.

        Este es el mecanismo para registrar aprobaciones de first_use:
            updated_permissions = permission_set.with_permission(
                Permission.allow_tool("clipboard_read", granted_by="user")
            )

        Args:
            new_permission: Permiso a añadir o actualizar.

        Returns:
            Nuevo PermissionSet con el permiso añadido/actualizado.
        """
        # Remover permiso existente con mismo scope+target si hay
        filtered = frozenset(
            p for p in self.permissions
            if not (p.scope == new_permission.scope and p.target == new_permission.target)
        )
        updated = filtered | {new_permission}

        return PermissionSet(
            permissions=updated,
            filesystem=self.filesystem,
            desktop=self.desktop,
            network=self.network,
            profile_id=self.profile_id,
            version=self.version + 1,
        )

    def with_app_approval(self, app_identifier: str) -> PermissionSet:
        """
        Crea un nuevo PermissionSet añadiendo una aplicación a la allowlist.

        Se usa cuando el usuario aprueba el lanzamiento de una app que no
        estaba en la allowlist original.

        Args:
            app_identifier: Identificador de la app aprobada.

        Returns:
            Nuevo PermissionSet con la app añadida a la allowlist.
        """
        new_allowlist = self.desktop.app_allowlist | {app_identifier.lower()}
        new_desktop = DesktopPermissions(
            app_allowlist=new_allowlist,
            clipboard_read_allowed=self.desktop.clipboard_read_allowed,
            clipboard_write_allowed=self.desktop.clipboard_write_allowed,
            screenshot_allowed=self.desktop.screenshot_allowed,
            notifications_allowed=self.desktop.notifications_allowed,
        )
        return PermissionSet(
            permissions=self.permissions,
            filesystem=self.filesystem,
            desktop=new_desktop,
            network=self.network,
            profile_id=self.profile_id,
            version=self.version + 1,
        )

    def with_clipboard_approval(
        self,
        *,
        read: bool = False,
        write: bool = False,
    ) -> PermissionSet:
        """
        Crea un nuevo PermissionSet con acceso al portapapeles habilitado.

        Args:
            read:  True para habilitar lectura del portapapeles.
            write: True para habilitar escritura en el portapapeles.

        Returns:
            Nuevo PermissionSet con los permisos de portapapeles actualizados.
        """
        new_desktop = DesktopPermissions(
            app_allowlist=self.desktop.app_allowlist,
            clipboard_read_allowed=self.desktop.clipboard_read_allowed or read,
            clipboard_write_allowed=self.desktop.clipboard_write_allowed or write,
            screenshot_allowed=self.desktop.screenshot_allowed,
            notifications_allowed=self.desktop.notifications_allowed,
        )
        return PermissionSet(
            permissions=self.permissions,
            filesystem=self.filesystem,
            desktop=new_desktop,
            network=self.network,
            profile_id=self.profile_id,
            version=self.version + 1,
        )

    def to_summary_dict(self) -> dict[str, Any]:
        """
        Serializa el PermissionSet a un diccionario para logging y debugging.

        No incluye información sensible — solo los niveles de acceso
        y la lista de tools/categorías permitidas.

        Returns:
            Diccionario con resumen de permisos para audit.
        """
        category_perms = {
            p.target: p.access.label()
            for p in self.permissions
            if p.scope == PermissionScope.CATEGORY
        }
        tool_perms = {
            p.target: p.access.label()
            for p in self.permissions
            if p.scope == PermissionScope.TOOL
        }
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "category_permissions": category_perms,
            "tool_permissions": tool_perms,
            "allowed_directories": [str(d) for d in self.filesystem.allowed_directories],
            "app_allowlist": sorted(self.desktop.app_allowlist),
            "clipboard_read": self.desktop.clipboard_read_allowed,
            "clipboard_write": self.desktop.clipboard_write_allowed,
            "screenshot": self.desktop.screenshot_allowed,
            "allowed_domains": list(self.network.allowed_domains),
        }


# =============================================================================
# UTILIDADES INTERNAS
# =============================================================================


def _is_subpath(child: Path, parent: Path) -> bool:
    """
    Verifica si child está dentro de parent (inclusive).

    Args:
        child:  Path a verificar (debe ser absoluto y resuelto).
        parent: Path del directorio padre (debe ser absoluto y resuelto).

    Returns:
        True si child es parent o un subdirectorio/archivo dentro de parent.
    """
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _detect_platform() -> str:
    """Detecta la plataforma actual del sistema operativo."""
    platform_map = {"linux": "linux", "win32": "windows", "darwin": "darwin"}
    return platform_map.get(sys.platform, "unknown")


# =============================================================================
# PERMISOS POR DEFECTO
# =============================================================================


def build_default_permissions() -> PermissionSet:
    """
    Construye el PermissionSet por defecto para un usuario nuevo.

    Estos son los permisos con los que arranca HiperForge User sin ninguna
    configuración del usuario. Siguen el principio de mínimo privilegio:
    solo lo necesario para las funciones básicas del producto, con aprobación
    requerida para acciones más sensibles.

    Filosofía de los defaults:
    - RESEARCH, DOCUMENTS, STUDY, PRODUCTIVITY: FULL (sin riesgo)
    - FILESYSTEM: READ (el usuario puede leer, no escribir por defecto)
    - DESKTOP: LIMITED (apps de allowlist + notificaciones, sin clipboard)
    - SYSTEM: READ (info del sistema, solo lectura)

    Returns:
        PermissionSet con los permisos por defecto del producto.
    """
    platform = _detect_platform()

    # --- Permisos de categoría ---
    category_permissions = [
        # Sin riesgo — acceso completo
        Permission.allow_category(ToolCategory.RESEARCH, AccessLevel.FULL),
        Permission.allow_category(ToolCategory.DOCUMENTS, AccessLevel.FULL),
        Permission.allow_category(ToolCategory.STUDY, AccessLevel.FULL),
        Permission.allow_category(ToolCategory.PRODUCTIVITY, AccessLevel.FULL),
        Permission.allow_category(ToolCategory.SYSTEM, AccessLevel.READ),
        # Requiere aprobación para mutación
        Permission.allow_category(ToolCategory.FILESYSTEM, AccessLevel.READ),
        Permission.allow_category(ToolCategory.DESKTOP, AccessLevel.READ),
    ]

    # --- Permisos de filesystem ---
    home = Path.home()
    allowed_dirs = (
        home,
        home / "Documents",
        home / "Downloads",
        home / "Desktop",
        home / "Pictures",
        home / "Videos",
        home / "Music",
    )

    # Patrones bloqueados específicos de la plataforma
    if platform == "windows":
        blocked_patterns = (
            "*\\Windows\\System32\\*",
            "*\\Windows\\SysWOW64\\*",
            "*\\AppData\\Roaming\\Microsoft\\Credentials\\*",
            "*\\AppData\\Local\\Microsoft\\Credentials\\*",
            "*\\.ssh\\*",
            "*\\.aws\\credentials",
            "**\\*.pem",
            "**\\*.key",
            "**\\.env",
        )
    else:
        blocked_patterns = (
            "*/.ssh/*",
            "*/.gnupg/*",
            "/etc/shadow",
            "/etc/passwd",
            "/etc/sudoers",
            "/proc/*",
            "/sys/*",
            "/boot/*",
            "*/.aws/credentials",
            "**/*.pem",
            "**/*.key",
            "**/.env",
            "**/*token*",
            "**/*secret*",
        )

    fs_permissions = FilesystemPermissions(
        allowed_directories=tuple(d.resolve() for d in allowed_dirs if d.exists() or d == home),
        blocked_path_patterns=blocked_patterns,
        max_read_size_mb=50,
        max_write_size_mb=10,
        allow_hidden_files=False,
    )

    # --- Permisos de desktop ---
    default_app_allowlist = frozenset({
        # Navegadores
        "firefox", "google-chrome", "chromium", "chromium-browser",
        "brave-browser", "microsoft-edge", "edge",
        # Suite ofimática
        "libreoffice", "libreoffice-writer", "libreoffice-calc",
        "libreoffice-impress", "libreoffice-draw",
        # Editores
        "code", "codium", "vscodium", "gedit", "kate", "mousepad",
        "notepad", "notepad++", "wordpad",
        # Calculadoras
        "calculator", "gnome-calculator", "kcalc", "galculator",
        # Gestores de archivos
        "nautilus", "dolphin", "thunar", "nemo", "files", "explorer",
        # Comunicación
        "thunderbird", "evolution",
        # PDF
        "evince", "okular", "zathura", "acrobat",
        # Imagen
        "eog", "gwenview", "shotwell", "gimp",
    })

    desktop_permissions = DesktopPermissions(
        app_allowlist=default_app_allowlist,
        clipboard_read_allowed=False,   # requiere first_use approval
        clipboard_write_allowed=False,  # requiere first_use approval
        screenshot_allowed=True,
        notifications_allowed=True,
    )

    # --- Permisos de red ---
    network_permissions = NetworkPermissions(
        allowed_domains=("*",),   # permitir todos por defecto
        blocked_domains=(
            # Dominios de minería de datos y trackers
            "doubleclick.net",
            "google-analytics.com",
            "facebook.com/tr",
            # Dominios de phishing conocidos (ejemplos ilustrativos)
            "*.xyz",  # TLD frecuentemente abusado
        ),
    )

    return PermissionSet(
        permissions=frozenset(category_permissions),
        filesystem=fs_permissions,
        desktop=desktop_permissions,
        network=network_permissions,
        profile_id="default",
        version=1,
    )


# Singleton de permisos por defecto — creado una sola vez al importar el módulo.
# El bootstrap lo usa para inicializar el perfil de usuario nuevo.
# No es mutable — con_permission() crea nuevas instancias.
DEFAULT_PERMISSIONS: PermissionSet = build_default_permissions()