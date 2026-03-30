"""
Entidad UserProfile del dominio HiperForge User.

El UserProfile encapsula toda la información persistente del usuario:
sus preferencias de interacción con el agente, sus permisos de acceso,
y los hechos recordados entre sesiones. Es la entidad que el IdentityPort
gestiona y que el ConversationCoordinator adjunta a cada sesión activa.

En V1, HiperForge User es single-user local. El UserProfile existe como
entidad del dominio (en lugar de valores hardcoded) porque:
  1. El PermissionSet evoluciona — los first_use approvals del usuario
     se persisten en el perfil para no preguntar de nuevo.
  2. Las preferencias se aprenden — el agente puede inferir preferencias
     del usuario a partir de las interacciones.
  3. V3 será multi-user — el puerto IdentityPort ya está diseñado para ello
     y el UserProfile como entidad permite la transición sin romper el dominio.

Ciclo de vida del UserProfile:
  - Se crea una vez al instalar la aplicación.
  - Se actualiza cada vez que el usuario cambia preferencias explícitamente.
  - Se actualiza cada vez que el usuario aprueba una tool por primera vez.
  - Se actualiza cuando el agente infiere una nueva preferencia del usuario.
  - Persiste entre sesiones — no se recrea en cada sesión.

Principios de diseño:
  1. El UserProfile es mutable — evoluciona con el usuario.
  2. Las preferencias son el único estado que el usuario controla explícitamente.
  3. Los permisos son gestionados por el PermissionSet (value object inmutable).
     El UserProfile es el contenedor que los persiste entre sesiones.
  4. El ProfileSummary es una vista comprimida del perfil para el LLM:
     incluye preferencias relevantes sin exponer IDs internos ni detalles técnicos.

Jerarquía de tipos:

  ResponseStyle          — estilo de respuesta del agente
  DetailLevel            — nivel de detalle de las respuestas
  UserPreferences        — todas las preferencias configurables del usuario
  UserProfile            — entidad principal del perfil de usuario
  ProfileSummary         — vista comprimida para el context del LLM
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from forge_core.errors.types import ForgeDomainError

from src.domain.value_objects.identifiers import UserId
from src.domain.value_objects.permission import (
    DEFAULT_PERMISSIONS,
    Permission,
    PermissionSet,
)


# =============================================================================
# ENUMERACIONES DE PREFERENCIAS
# =============================================================================


class ResponseStyle(Enum):
    """
    Estilo de comunicación preferido del agente al responder.

    Afecta cómo el LLM formula sus respuestas — se incluye en el system prompt
    como instrucción de personalidad del agente.
    """

    CONCISE = "concise"
    """
    Respuestas directas y al punto. Sin preámbulos ni explicaciones de más.
    Ideal para usuarios que saben lo que quieren y prefieren eficiencia.
    Longitud típica: 1-3 párrafos.
    """

    BALANCED = "balanced"
    """
    Equilibrio entre completitud y brevedad.
    La opción por defecto — apropiada para la mayoría de usuarios y contextos.
    Longitud típica: 2-5 párrafos según la complejidad.
    """

    DETAILED = "detailed"
    """
    Respuestas completas y exhaustivas. Incluye contexto, ejemplos y matices.
    Ideal para tareas de aprendizaje o cuando el usuario necesita entender en profundidad.
    Longitud típica: múltiples párrafos o secciones.
    """

    EDUCATIONAL = "educational"
    """
    Estilo pedagógico — explica el por qué, usa analogías y ejemplos.
    Ideal para sesiones de estudio y aprendizaje de nuevos conceptos.
    Similar a DETAILED pero con foco en comprensión y retención.
    """


class DetailLevel(Enum):
    """
    Nivel de detalle técnico en las respuestas del agente.

    Complementa ResponseStyle para calibrar la profundidad técnica.
    """

    SIMPLE = "simple"
    """
    Lenguaje cotidiano, sin jerga técnica.
    Ideal para usuarios no técnicos — es el default de HiperForge User.
    """

    MODERATE = "moderate"
    """
    Algo de terminología técnica cuando es necesario, siempre con explicación.
    Equilibrio entre accesibilidad y precisión.
    """

    TECHNICAL = "technical"
    """
    Terminología técnica sin simplificación. Útil para usuarios con formación
    específica en el tema que se discute.
    """


class AgentLanguage(Enum):
    """
    Idioma en el que el agente se comunica con el usuario.

    Afecta el system prompt y los mensajes de UI generados por el agente.
    """

    SPANISH = "es"
    """Español (por defecto — HiperForge User está orientado a hispanohablantes)."""

    ENGLISH = "en"
    """Inglés."""

    PORTUGUESE = "pt"
    """Portugués."""

    FRENCH = "fr"
    """Francés."""

    GERMAN = "de"
    """Alemán."""

    ITALIAN = "it"
    """Italiano."""


# =============================================================================
# PREFERENCIAS DEL USUARIO
# =============================================================================


@dataclass
class UserPreferences:
    """
    Conjunto de preferencias configurables del usuario.

    Las preferencias controlan cómo el agente se comunica y se comporta.
    Son mutables — el usuario puede cambiarlas en cualquier momento desde
    la configuración de la app, o el agente puede actualizarlas cuando el
    usuario las menciona explícitamente en la conversación.

    Se persisten en el UserProfile entre sesiones. Se incluyen en el
    ProfileSummary que el ContextBuilder añade al contexto del LLM.
    """

    # --- Comunicación ---
    response_style: ResponseStyle = ResponseStyle.BALANCED
    """Estilo de respuesta preferido."""

    detail_level: DetailLevel = DetailLevel.SIMPLE
    """Nivel de detalle técnico de las respuestas."""

    language: AgentLanguage = AgentLanguage.SPANISH
    """Idioma de comunicación del agente."""

    display_name: str = "Usuario"
    """
    Nombre con el que el agente se dirige al usuario.
    Ejemplo: 'César', 'Dr. Martínez', 'Profe'.
    """

    # --- Comportamiento del agente ---
    show_thinking_indicators: bool = True
    """
    Si True, muestra indicadores de progreso en la UI ('Buscando...', etc.).
    Desactivar para experiencia más limpia en uso experto.
    """

    include_source_citations: bool = True
    """
    Si True, el agente cita las fuentes usadas (URLs, nombres de archivo).
    Relevante para búsquedas web y análisis de documentos.
    """

    proactive_suggestions: bool = True
    """
    Si True, el agente sugiere acciones relacionadas después de completar una tarea.
    Ejemplo: 'También puedo generar flashcards de este resumen'.
    """

    remember_preferences: bool = True
    """
    Si True, el agente puede actualizar las preferencias en base a las
    interacciones del usuario (aprendizaje implícito).
    False = solo se actualizan manualmente desde settings.
    """

    # --- Estudio (para funciones de aprendizaje) ---
    study_session_duration_minutes: int = 25
    """
    Duración sugerida de una sesión de estudio en minutos.
    Basado en la técnica Pomodoro por defecto (25 min).
    """

    flashcard_cards_per_session: int = 20
    """Número de flashcards a revisar en una sesión de repaso."""

    preferred_explanation_style: str = "analogies"
    """
    Estilo de explicación preferido para conceptos nuevos.
    Opciones: 'analogies', 'examples', 'formal', 'visual'.
    """

    # --- Privacy y control ---
    save_artifacts: bool = True
    """
    Si True, los artifacts (resúmenes, flashcards, etc.) se persisten
    entre sesiones. False = se descartan al cerrar la sesión.
    """

    long_term_memory_enabled: bool = True
    """
    Si True, el agente recuerda hechos entre sesiones (nombre del profesor,
    temas de estudio, etc.). False = amnesia entre sesiones (máxima privacidad).
    """

    def to_system_prompt_fragment(self) -> str:
        """
        Genera el fragmento del system prompt que describe las preferencias del usuario.

        Este fragmento se incluye en el LLMContext para que el LLM adapte
        su estilo de respuesta a las preferencias específicas del usuario.

        Returns:
            String con las instrucciones de comportamiento para el LLM.
        """
        fragments: list[str] = []

        # Dirección al usuario
        if self.display_name and self.display_name != "Usuario":
            fragments.append(
                f"El usuario se llama {self.display_name}. "
                f"Dirígete a él por su nombre cuando sea natural hacerlo."
            )

        # Estilo de respuesta
        style_instructions = {
            ResponseStyle.CONCISE: (
                "Sé conciso y directo. Evita preámbulos y explicaciones de más. "
                "Ve al grano."
            ),
            ResponseStyle.BALANCED: (
                "Equilibra completitud y brevedad. Responde lo que se pregunta "
                "sin extenderte innecesariamente."
            ),
            ResponseStyle.DETAILED: (
                "Sé exhaustivo y completo. Incluye contexto, matices y ejemplos "
                "cuando sean relevantes."
            ),
            ResponseStyle.EDUCATIONAL: (
                "Adopta un estilo pedagógico. Explica el por qué, usa analogías "
                "y ejemplos concretos. Facilita la comprensión y retención."
            ),
        }
        fragments.append(style_instructions[self.response_style])

        # Nivel de detalle
        detail_instructions = {
            DetailLevel.SIMPLE: (
                "Usa lenguaje cotidiano y accesible. Evita jerga técnica. "
                "Si necesitas un término técnico, explícalo."
            ),
            DetailLevel.MODERATE: (
                "Puedes usar terminología técnica moderada cuando aporte precisión, "
                "pero acompáñala de una breve explicación."
            ),
            DetailLevel.TECHNICAL: (
                "Puedes usar terminología técnica sin simplificación. "
                "El usuario tiene conocimiento del área."
            ),
        }
        fragments.append(detail_instructions[self.detail_level])

        # Citas y fuentes
        if self.include_source_citations:
            fragments.append(
                "Cuando uses información de fuentes específicas (URLs, documentos), "
                "menciona la fuente de forma natural en tu respuesta."
            )

        # Sugerencias proactivas
        if self.proactive_suggestions:
            fragments.append(
                "Después de completar una tarea, puedes sugerir acciones relacionadas "
                "que podrían ser útiles para el usuario."
            )

        return "\n".join(fragments)

    def to_dict(self) -> dict[str, Any]:
        """Serializa las preferencias para persistencia."""
        return {
            "response_style": self.response_style.value,
            "detail_level": self.detail_level.value,
            "language": self.language.value,
            "display_name": self.display_name,
            "show_thinking_indicators": self.show_thinking_indicators,
            "include_source_citations": self.include_source_citations,
            "proactive_suggestions": self.proactive_suggestions,
            "remember_preferences": self.remember_preferences,
            "study_session_duration_minutes": self.study_session_duration_minutes,
            "flashcard_cards_per_session": self.flashcard_cards_per_session,
            "preferred_explanation_style": self.preferred_explanation_style,
            "save_artifacts": self.save_artifacts,
            "long_term_memory_enabled": self.long_term_memory_enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserPreferences:
        """
        Reconstruye las preferencias desde un diccionario persistido.

        Es tolerante a campos faltantes — usa defaults para cualquier campo
        no presente. Esto permite añadir nuevas preferencias en versiones
        futuras sin romper perfiles existentes.

        Args:
            data: Dict serializado por to_dict().

        Returns:
            UserPreferences reconstruidas.
        """
        prefs = cls()

        if "response_style" in data:
            try:
                prefs.response_style = ResponseStyle(data["response_style"])
            except ValueError:
                pass  # valor desconocido → mantener default

        if "detail_level" in data:
            try:
                prefs.detail_level = DetailLevel(data["detail_level"])
            except ValueError:
                pass

        if "language" in data:
            try:
                prefs.language = AgentLanguage(data["language"])
            except ValueError:
                pass

        if "display_name" in data:
            prefs.display_name = str(data["display_name"])

        # Booleanos — con get() para tolerancia a campos faltantes
        for bool_field in [
            "show_thinking_indicators", "include_source_citations",
            "proactive_suggestions", "remember_preferences",
            "save_artifacts", "long_term_memory_enabled",
        ]:
            if bool_field in data:
                setattr(prefs, bool_field, bool(data[bool_field]))

        # Enteros
        for int_field in [
            "study_session_duration_minutes", "flashcard_cards_per_session",
        ]:
            if int_field in data:
                try:
                    setattr(prefs, int_field, int(data[int_field]))
                except (ValueError, TypeError):
                    pass

        if "preferred_explanation_style" in data:
            prefs.preferred_explanation_style = str(data["preferred_explanation_style"])

        return prefs

    @classmethod
    def defaults(cls) -> UserPreferences:
        """
        Crea las preferencias por defecto para un usuario nuevo.

        Returns:
            UserPreferences con todos los valores por defecto.
        """
        return cls()


# =============================================================================
# ENTIDAD USER PROFILE
# =============================================================================


class UserProfile:
    """
    Perfil del usuario — entidad mutable que persiste entre sesiones.

    Contiene las preferencias del usuario, sus permisos (que evolucionan
    con las aprobaciones first_use), y los hechos recordados a largo plazo.

    Es la entidad central de identidad del sistema. El SessionManager
    la adjunta a cada sesión activa, y el ConversationCoordinator la
    consulta para construir el contexto del LLM.

    Ciclo de vida:
      - UserProfile.create_default() → perfil nuevo con defaults
      - Se actualiza cuando el usuario cambia preferencias en la app
      - Se actualiza cuando el agente aprende preferencias nuevas
      - Se actualiza cuando el usuario aprueba una tool (first_use)
      - Se persiste automáticamente después de cada cambio
    """

    def __init__(
        self,
        user_id: UserId,
        preferences: UserPreferences,
        permissions: PermissionSet,
        *,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> None:
        """
        Inicializa un UserProfile.

        Args:
            user_id:     ID único del usuario.
            preferences: Preferencias del usuario.
            permissions: PermissionSet del usuario.
            created_at:  Timestamp de creación. None = ahora en UTC.
            updated_at:  Timestamp de última actualización. None = ahora.
        """
        self._user_id = user_id
        self._preferences = preferences
        self._permissions = permissions
        now = datetime.now(tz=timezone.utc)
        self._created_at = created_at or now
        self._updated_at = updated_at or now
        self._session_count: int = 0
        self._total_turns: int = 0
        self._is_dirty: bool = False

    # =========================================================================
    # FACTORIES
    # =========================================================================

    @classmethod
    def create_default(cls) -> UserProfile:
        """
        Factory: crea un UserProfile nuevo con configuración por defecto.

        Se llama la primera vez que arranca la aplicación, cuando no existe
        ningún perfil persistido.

        Returns:
            UserProfile nuevo con preferencias y permisos por defecto.
        """
        return cls(
            user_id=UserId.generate(),
            preferences=UserPreferences.defaults(),
            permissions=DEFAULT_PERMISSIONS,
        )

    @classmethod
    def restore(
        cls,
        user_id: UserId,
        preferences_dict: dict[str, Any],
        permissions: PermissionSet,
        *,
        created_at: datetime,
        updated_at: datetime,
        session_count: int = 0,
        total_turns: int = 0,
    ) -> UserProfile:
        """
        Factory: restaura un UserProfile desde el storage.

        Args:
            user_id:          ID del usuario.
            preferences_dict: Preferencias serializadas.
            permissions:      PermissionSet reconstruido.
            created_at:       Timestamp de creación original.
            updated_at:       Timestamp de última actualización.
            session_count:    Número de sesiones históricas.
            total_turns:      Total de turns en toda la historia.

        Returns:
            UserProfile restaurado con el estado exacto del storage.
        """
        profile = cls(
            user_id=user_id,
            preferences=UserPreferences.from_dict(preferences_dict),
            permissions=permissions,
            created_at=created_at,
            updated_at=updated_at,
        )
        profile._session_count = session_count
        profile._total_turns = total_turns
        return profile

    # =========================================================================
    # PROPIEDADES
    # =========================================================================

    @property
    def user_id(self) -> UserId:
        """ID único del usuario."""
        return self._user_id

    @property
    def preferences(self) -> UserPreferences:
        """Preferencias actuales del usuario."""
        return self._preferences

    @property
    def permissions(self) -> PermissionSet:
        """PermissionSet actual del usuario."""
        return self._permissions

    @property
    def created_at(self) -> datetime:
        """Timestamp de creación del perfil (UTC)."""
        return self._created_at

    @property
    def updated_at(self) -> datetime:
        """Timestamp de última actualización (UTC)."""
        return self._updated_at

    @property
    def session_count(self) -> int:
        """Número total de sesiones históricas del usuario."""
        return self._session_count

    @property
    def total_turns(self) -> int:
        """Total de turns en toda la historia del usuario."""
        return self._total_turns

    @property
    def is_dirty(self) -> bool:
        """
        True si el perfil tiene cambios pendientes de persistir.

        El SessionManager consulta este flag para saber si debe
        guardar el perfil al finalizar una sesión.
        """
        return self._is_dirty

    @property
    def display_name(self) -> str:
        """Nombre de display del usuario (atajo a preferences.display_name)."""
        return self._preferences.display_name

    # =========================================================================
    # COMANDOS DE PREFERENCIAS
    # =========================================================================

    def update_preferences(self, **kwargs: Any) -> None:
        """
        Actualiza preferencias individuales del usuario.

        Acepta cualquier combinación de campos de UserPreferences como
        keyword arguments. Los campos no especificados no se modifican.

        Args:
            **kwargs: Campos de UserPreferences a actualizar.

        Raises:
            ForgeDomainError: Si algún campo no existe en UserPreferences.

        Example:
            profile.update_preferences(
                response_style=ResponseStyle.CONCISE,
                display_name='César',
            )
        """
        valid_fields = {
            "response_style", "detail_level", "language", "display_name",
            "show_thinking_indicators", "include_source_citations",
            "proactive_suggestions", "remember_preferences",
            "study_session_duration_minutes", "flashcard_cards_per_session",
            "preferred_explanation_style", "save_artifacts",
            "long_term_memory_enabled",
        }

        invalid_fields = set(kwargs.keys()) - valid_fields
        if invalid_fields:
            raise ForgeDomainError(
                f"Campos de preferencias desconocidos: {sorted(invalid_fields)}. "
                f"Campos válidos: {sorted(valid_fields)}.",
                context={"invalid_fields": sorted(invalid_fields)},
            )

        for field_name, value in kwargs.items():
            setattr(self._preferences, field_name, value)

        self._touch()

    def set_display_name(self, name: str) -> None:
        """
        Actualiza el nombre de display del usuario.

        Args:
            name: Nuevo nombre de display.

        Raises:
            ForgeDomainError: Si el nombre está vacío o es demasiado largo.
        """
        name = name.strip()
        if not name:
            raise ForgeDomainError(
                "El nombre de display no puede estar vacío.",
                context={"user_id": self._user_id.to_str()},
            )
        if len(name) > 100:
            raise ForgeDomainError(
                f"El nombre de display no puede superar los 100 caracteres "
                f"(longitud actual: {len(name)}).",
                context={"user_id": self._user_id.to_str(), "name_length": len(name)},
            )
        self._preferences.display_name = name
        self._touch()

    # =========================================================================
    # COMANDOS DE PERMISOS
    # =========================================================================

    def grant_tool_permission(
        self,
        tool_id: str,
        *,
        remember: bool = True,
    ) -> None:
        """
        Registra la aprobación first_use del usuario para una tool.

        Se llama desde el ApprovalWorkflow cuando el usuario aprueba una
        tool por primera vez y elige recordar la decisión.

        Args:
            tool_id: ID de la tool aprobada.
            remember: True para persistir el permiso (recordar decisión).
                      False para autorizar solo esta vez sin persistir.
        """
        if not remember:
            return

        new_permission = Permission.allow_tool(
            tool_id=tool_id,
            granted_by="user",
        )
        self._permissions = self._permissions.with_permission(new_permission)
        self._touch()

    def grant_app_permission(self, app_identifier: str) -> None:
        """
        Añade una aplicación a la allowlist del usuario.

        Se llama cuando el usuario aprueba el lanzamiento de una app
        que no estaba en la allowlist por defecto.

        Args:
            app_identifier: Identificador de la app aprobada.
        """
        self._permissions = self._permissions.with_app_approval(app_identifier)
        self._touch()

    def grant_clipboard_access(
        self,
        *,
        read: bool = False,
        write: bool = False,
    ) -> None:
        """
        Habilita acceso al portapapeles en el perfil del usuario.

        Args:
            read:  True para habilitar lectura del portapapeles.
            write: True para habilitar escritura en el portapapeles.
        """
        self._permissions = self._permissions.with_clipboard_approval(
            read=read,
            write=write,
        )
        self._touch()

    def revoke_tool_permission(self, tool_id: str) -> None:
        """
        Revoca el permiso de una tool específica.

        Crea un permiso DENY explícito para la tool, que tiene precedencia
        sobre permisos de categoría.

        Args:
            tool_id: ID de la tool cuyo permiso se revoca.
        """
        from src.domain.value_objects.permission import AccessLevel
        deny_permission = Permission(
            scope=__import__(
                'src.domain.value_objects.permission',
                fromlist=['PermissionScope'],
            ).PermissionScope.TOOL,
            target=tool_id,
            access=AccessLevel.DENY,
            granted_by="user",
        )
        self._permissions = self._permissions.with_permission(deny_permission)
        self._touch()

    # =========================================================================
    # ESTADÍSTICAS Y MÉTRICAS
    # =========================================================================

    def record_session_started(self) -> None:
        """
        Registra el inicio de una nueva sesión.

        El SessionManager llama a este método al crear una nueva sesión.
        """
        self._session_count += 1
        self._touch()

    def record_turns_completed(self, turn_count: int) -> None:
        """
        Registra el número de turns completados en una sesión.

        El SessionManager llama a este método al cerrar una sesión.

        Args:
            turn_count: Número de turns de la sesión que termina.
        """
        if turn_count < 0:
            raise ForgeDomainError(
                f"El número de turns no puede ser negativo: {turn_count}.",
                context={"user_id": self._user_id.to_str()},
            )
        self._total_turns += turn_count
        self._touch()

    def mark_clean(self) -> None:
        """
        Marca el perfil como limpio (sin cambios pendientes).

        El SessionManager llama a este método después de persistir el perfil.
        """
        self._is_dirty = False

    # =========================================================================
    # VISTAS
    # =========================================================================

    def to_profile_summary(self) -> ProfileSummary:
        """
        Crea el ProfileSummary para incluir en el contexto del LLM.

        El ContextBuilder usa este summary para construir la parte del
        contexto que describe al usuario al LLM.

        Returns:
            ProfileSummary con la información relevante del perfil.
        """
        return ProfileSummary(
            display_name=self._preferences.display_name,
            language=self._preferences.language.value,
            response_style=self._preferences.response_style.value,
            detail_level=self._preferences.detail_level.value,
            include_citations=self._preferences.include_source_citations,
            proactive_suggestions=self._preferences.proactive_suggestions,
            preferred_explanation_style=self._preferences.preferred_explanation_style,
            session_count=self._session_count,
            is_new_user=self._session_count <= 1,
            system_prompt_fragment=self._preferences.to_system_prompt_fragment(),
        )

    def to_storage_dict(self) -> dict[str, Any]:
        """
        Serializa el UserProfile para persistencia en SQLite.

        Returns:
            Dict con todos los campos del perfil serializados.
        """
        return {
            "user_id": self._user_id.to_str(),
            "preferences": self._preferences.to_dict(),
            "created_at": self._created_at.isoformat(),
            "updated_at": self._updated_at.isoformat(),
            "session_count": self._session_count,
            "total_turns": self._total_turns,
        }

    # =========================================================================
    # HELPERS INTERNOS
    # =========================================================================

    def _touch(self) -> None:
        """Actualiza el timestamp de última modificación y marca como dirty."""
        self._updated_at = datetime.now(tz=timezone.utc)
        self._is_dirty = True

    def __repr__(self) -> str:
        return (
            f"UserProfile("
            f"id={self._user_id.to_str()!r}, "
            f"name={self._preferences.display_name!r}, "
            f"sessions={self._session_count}"
            f")"
        )


# =============================================================================
# PROFILE SUMMARY (para contexto del LLM)
# =============================================================================


@dataclass(frozen=True)
class ProfileSummary:
    """
    Vista comprimida del perfil de usuario para el contexto del LLM.

    Es la representación del perfil que el ContextBuilder incluye en el
    LLMContext. Contiene solo la información que el LLM necesita para
    personalizar sus respuestas — sin IDs internos, sin permisos técnicos,
    sin metadata de storage.

    Se diseñó para ser pequeña (~300 tokens) — ver MemoryConfig.user_profile_reserve.
    """

    display_name: str
    """Nombre del usuario para dirigirse a él."""

    language: str
    """Idioma de comunicación (código BCP 47)."""

    response_style: str
    """Estilo de respuesta preferido."""

    detail_level: str
    """Nivel de detalle técnico."""

    include_citations: bool
    """Si el usuario quiere que se citen las fuentes."""

    proactive_suggestions: bool
    """Si el agente debe hacer sugerencias proactivas."""

    preferred_explanation_style: str
    """Estilo preferido de explicación para conceptos nuevos."""

    session_count: int
    """Número total de sesiones del usuario (para adaptar onboarding)."""

    is_new_user: bool
    """True si el usuario tiene pocas sesiones (≤ 1)."""

    system_prompt_fragment: str
    """
    Fragmento del system prompt generado desde las preferencias.
    Listo para incluir directamente en el LLMContext.
    """

    def to_context_string(self) -> str:
        """
        Genera el string de contexto del perfil para el system prompt del LLM.

        Este string se incluye en el LLMContext como parte del contexto
        de personalización del agente. Es conciso y accionable para el LLM.

        Returns:
            String con el contexto del perfil, listo para el system prompt.
        """
        parts: list[str] = ["## Perfil del Usuario\n"]

        if self.display_name and self.display_name != "Usuario":
            parts.append(f"El usuario se llama **{self.display_name}**.")

        if self.is_new_user:
            parts.append(
                "Es un usuario nuevo — adapta tu nivel de familiaridad con la herramienta."
            )

        parts.append("\n### Instrucciones de estilo:")
        parts.append(self.system_prompt_fragment)

        return "\n".join(parts)

    def estimated_tokens(self) -> int:
        """Estimación de tokens de este summary (heurística 4 chars/token)."""
        total_chars = len(self.to_context_string())
        return max(1, total_chars // 4)

    def to_dict(self) -> dict[str, Any]:
        """Serializa el summary para logging y debugging."""
        return {
            "display_name": self.display_name,
            "language": self.language,
            "response_style": self.response_style,
            "detail_level": self.detail_level,
            "session_count": self.session_count,
            "is_new_user": self.is_new_user,
            "estimated_tokens": self.estimated_tokens(),
        }