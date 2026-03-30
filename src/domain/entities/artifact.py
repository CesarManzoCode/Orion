"""
Entidad Artifact del dominio HiperForge User.

Los artifacts son los productos del trabajo del agente: resúmenes de documentos,
reportes de investigación, resultados de búsqueda, flashcards generadas,
planes de estudio, tablas comparativas, y cualquier otro contenido estructurado
que el agente produce durante una sesión.

A diferencia de las entidades Session y Task (que son mutables y evolucionan),
los Artifacts son inmutables una vez creados. Representan un resultado concreto
y finito del trabajo del agente en un momento específico.

Características del diseño:
  1. Inmutabilidad completa — los artifacts no se modifican después de crearse.
     Si se necesita una versión actualizada, se crea un nuevo artifact.
  2. Contenido tipado — cada ArtifactType tiene sus propias expectativas de
     contenido y metadata. El sistema valida la coherencia tipo-contenido.
  3. Referenciabilidad — los artifacts tienen IDs estables que la Session,
     las Tasks, y el ContextBuilder pueden referenciar sin cargar el contenido.
  4. Tamaño controlado — el contenido tiene límites de tamaño para proteger
     el context window del LLM y el storage local.
  5. Exportabilidad — los artifacts pueden exportarse a formatos externos
     (Markdown, JSON, texto plano) para uso del usuario.

Jerarquía de tipos:

  ArtifactType          — tipo semántico del artifact
  ArtifactContentType   — tipo técnico del contenido almacenado
  ArtifactContent       — contenedor tipado del contenido
  ArtifactMetadata      — metadata de búsqueda y visualización
  Artifact              — entidad inmutable principal
  ArtifactSummary       — vista ligera para listados en UI
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.domain.value_objects.identifiers import (
    ArtifactId,
    SessionId,
    TaskId,
    TurnId,
)


# =============================================================================
# ENUMERACIONES
# =============================================================================


class ArtifactType(Enum):
    """
    Tipo semántico del artifact — qué representa para el usuario.

    El tipo determina cómo se muestra en la UI, qué acciones están disponibles
    (exportar como Markdown, generar quiz desde flashcards, etc.), y cómo el
    ContextBuilder lo incluye en el contexto del LLM.
    """

    SUMMARY = "summary"
    """
    Resumen de un documento o fuente de información.
    Producido por: document_summarize, document_analyze.
    UI: vista de texto con opción de exportar a Markdown.
    Acciones disponibles: exportar, generar flashcards desde resumen.
    """

    SEARCH_RESULTS = "search_results"
    """
    Resultados de búsqueda web con sus snippets y URLs.
    Producido por: web_search, academic_search.
    UI: lista de resultados con links clickeables.
    Acciones disponibles: exportar lista, hacer fetch de resultados específicos.
    """

    ANALYSIS = "analysis"
    """
    Análisis profundo de un documento, tema, o conjunto de datos.
    Producido por: document_analyze, compare_documents.
    UI: informe estructurado con secciones.
    """

    COMPARISON_TABLE = "comparison_table"
    """
    Tabla comparativa de opciones, productos, o conceptos.
    Producido por: compare_options, research_comparison.
    UI: tabla renderizada con columnas y filas.
    Acciones disponibles: exportar como CSV o Markdown.
    """

    FLASHCARDS = "flashcards"
    """
    Conjunto de tarjetas pregunta-respuesta para estudio.
    Producido por: flashcard_generate.
    UI: deck de flashcards interactivo.
    Acciones disponibles: iniciar sesión de repaso, exportar a Anki.
    """

    QUIZ = "quiz"
    """
    Quiz de preguntas con respuestas correctas e incorrectas.
    Producido por: quiz_generate.
    UI: interfaz de quiz interactivo.
    Acciones disponibles: iniciar quiz, ver respuestas.
    """

    STUDY_PLAN = "study_plan"
    """
    Plan de estudio estructurado por temas y sesiones.
    Producido por: study_plan_generate.
    UI: lista de temas con estimaciones de tiempo.
    """

    CONCEPT_MAP = "concept_map"
    """
    Mapa conceptual de relaciones entre conceptos.
    Producido por: concept_map.
    UI: diagrama visual de nodos y relaciones.
    Exporta: SVG, JSON de grafo.
    """

    GENERATED_FILE = "generated_file"
    """
    Archivo generado y guardado en el filesystem local.
    Producido por: file_write, document_create.
    UI: link al archivo con opción de abrir.
    Metadata: path del archivo, tamaño, tipo MIME.
    """

    NOTE = "note"
    """
    Nota o apunte creado durante la sesión.
    Producido por: notes_create.
    UI: texto simple con opción de editar (en V2).
    """

    ACTION_LOG = "action_log"
    """
    Registro de acciones ejecutadas (para audit trail visible al usuario).
    Producido por: el sistema de logging del TaskExecutor.
    UI: lista cronológica de acciones con iconos de estado.
    """

    EXTRACTED_TEXT = "extracted_text"
    """
    Texto extraído de un documento (PDF, DOCX, etc.) para referencia.
    Producido por: document_read.
    UI: texto con paginación.
    """

    WEB_PAGE = "web_page"
    """
    Contenido de una página web obtenida por fetch.
    Producido por: web_fetch.
    UI: texto del contenido con URL de origen.
    """


class ArtifactContentType(Enum):
    """
    Tipo técnico del contenido almacenado en el artifact.

    Determina cómo se serializa y deserializa el contenido para storage
    y para inclusión en el contexto del LLM.
    """

    PLAIN_TEXT = "plain_text"
    """Texto sin formato. Serialización directa a string."""

    MARKDOWN = "markdown"
    """Texto Markdown. Renderizado en la UI, enviado como texto al LLM."""

    JSON = "json"
    """Datos estructurados JSON. Serializado a string para el LLM."""

    STRUCTURED_LIST = "structured_list"
    """Lista de items estructurados (flashcards, quiz questions, search results)."""

    FILE_REFERENCE = "file_reference"
    """
    Referencia a un archivo en el filesystem.
    El contenido es un path, no el contenido del archivo.
    """


# =============================================================================
# CONTENIDO DEL ARTIFACT
# =============================================================================


@dataclass(frozen=True)
class ArtifactContent:
    """
    Contenedor tipado del contenido de un artifact.

    Encapsula el contenido real junto con su tipo técnico para garantizar
    que la serialización y el truncado para el LLM sean correctos.

    El contenido tiene un límite de tamaño para proteger el context window
    y el storage local. Los artifacts muy grandes se truncan con un indicador.
    """

    content_type: ArtifactContentType
    """Tipo técnico del contenido."""

    raw_content: str
    """
    Contenido serializado como string.
    Para JSON y STRUCTURED_LIST, es el JSON serializado.
    Para FILE_REFERENCE, es el path absoluto del archivo.
    Máximo: MAX_CONTENT_CHARS caracteres.
    """

    is_truncated: bool = False
    """
    True si el contenido fue truncado para respetar el límite de tamaño.
    El usuario ve una nota de truncación en la UI.
    """

    original_size_chars: int = 0
    """
    Tamaño original del contenido antes de truncar (si aplica).
    0 si no fue necesario truncar.
    """

    # Límite de tamaño del contenido almacenado
    MAX_CONTENT_CHARS: int = field(default=500_000, init=False, repr=False)
    """Máximo de caracteres del contenido almacenado (~125K tokens)."""

    LLM_TRUNCATION_CHARS: int = field(default=32_000, init=False, repr=False)
    """Máximo de caracteres al incluir en el contexto del LLM (~8K tokens)."""

    def for_llm_context(self, max_chars: int | None = None) -> str:
        """
        Retorna el contenido preparado para incluir en el contexto del LLM.

        Trunca el contenido si supera el límite del LLM, añadiendo un indicador
        de truncación para que el LLM sepa que el contenido es parcial.

        Args:
            max_chars: Límite de caracteres. None usa LLM_TRUNCATION_CHARS.

        Returns:
            String del contenido listo para el LLM, truncado si necesario.
        """
        limit = max_chars or self.LLM_TRUNCATION_CHARS

        if len(self.raw_content) <= limit:
            return self.raw_content

        truncation_notice = (
            f"\n\n[...contenido truncado a {limit} caracteres de "
            f"{len(self.raw_content)} totales...]"
        )
        return self.raw_content[:limit - len(truncation_notice)] + truncation_notice

    def size_chars(self) -> int:
        """Retorna el tamaño del contenido en caracteres."""
        return len(self.raw_content)

    def estimated_tokens(self) -> int:
        """Estimación de tokens del contenido (heurística 4 chars/token)."""
        return max(1, len(self.raw_content) // 4)

    @classmethod
    def from_text(cls, text: str) -> ArtifactContent:
        """
        Factory: crea contenido de texto plano.

        Args:
            text: Texto sin formato.

        Returns:
            ArtifactContent de tipo PLAIN_TEXT.
        """
        return cls(
            content_type=ArtifactContentType.PLAIN_TEXT,
            raw_content=text,
        )

    @classmethod
    def from_markdown(cls, markdown: str) -> ArtifactContent:
        """
        Factory: crea contenido Markdown.

        Args:
            markdown: Texto en formato Markdown.

        Returns:
            ArtifactContent de tipo MARKDOWN.
        """
        return cls(
            content_type=ArtifactContentType.MARKDOWN,
            raw_content=markdown,
        )

    @classmethod
    def from_json(cls, data: Any) -> ArtifactContent:
        """
        Factory: crea contenido JSON serializado.

        Args:
            data: Datos a serializar como JSON.

        Returns:
            ArtifactContent de tipo JSON.
        """
        import json
        try:
            serialized = json.dumps(data, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            serialized = str(data)
        return cls(
            content_type=ArtifactContentType.JSON,
            raw_content=serialized,
        )

    @classmethod
    def from_structured_list(cls, items: list[dict[str, Any]]) -> ArtifactContent:
        """
        Factory: crea contenido de lista estructurada (flashcards, quiz, etc.).

        Args:
            items: Lista de items como dicts.

        Returns:
            ArtifactContent de tipo STRUCTURED_LIST.
        """
        import json
        serialized = json.dumps(items, ensure_ascii=False, indent=2)
        return cls(
            content_type=ArtifactContentType.STRUCTURED_LIST,
            raw_content=serialized,
        )

    @classmethod
    def from_file_path(cls, path: str) -> ArtifactContent:
        """
        Factory: crea una referencia a un archivo del filesystem.

        Args:
            path: Path absoluto del archivo generado.

        Returns:
            ArtifactContent de tipo FILE_REFERENCE.
        """
        return cls(
            content_type=ArtifactContentType.FILE_REFERENCE,
            raw_content=path,
        )


# =============================================================================
# METADATA DEL ARTIFACT
# =============================================================================


@dataclass(frozen=True)
class ArtifactMetadata:
    """
    Metadata del artifact para búsqueda, visualización y exportación.

    La metadata se indexa en SQLite para permitir búsquedas rápidas
    por tipo, fuente, tags, y sesión sin cargar el contenido completo.
    """

    source_url: str | None = None
    """URL de origen si el artifact deriva de contenido web."""

    source_filename: str | None = None
    """Nombre del archivo origen si deriva de un documento local."""

    source_tool_id: str | None = None
    """ID de la tool que produjo este artifact."""

    tags: tuple[str, ...] = field(default_factory=tuple)
    """
    Tags semánticos para búsqueda y clasificación.
    Ejemplo: ('machine_learning', 'paper', 'summary').
    """

    language: str = "es"
    """Idioma del contenido (código BCP 47). Por defecto español."""

    page_count: int | None = None
    """Número de páginas si deriva de un documento paginado."""

    item_count: int | None = None
    """
    Número de items si el artifact es una colección.
    Ej: número de flashcards, preguntas de quiz, resultados de búsqueda.
    """

    file_size_bytes: int | None = None
    """Tamaño del archivo referenciado, si aplica."""

    file_mime_type: str | None = None
    """MIME type del archivo referenciado, si aplica."""

    extra: dict[str, Any] = field(default_factory=dict)
    """
    Metadata adicional específica por tipo de artifact.
    No se indexa — se usa para visualización y exportación.
    """

    def to_index_dict(self) -> dict[str, Any]:
        """
        Serializa los campos indexables para almacenamiento en SQLite.

        Returns:
            Dict con solo los campos que se indexan en la base de datos.
        """
        return {
            "source_url": self.source_url,
            "source_filename": self.source_filename,
            "source_tool_id": self.source_tool_id,
            "tags": list(self.tags),
            "language": self.language,
            "item_count": self.item_count,
        }


# =============================================================================
# ARTIFACT — ENTIDAD PRINCIPAL
# =============================================================================


@dataclass(frozen=True)
class Artifact:
    """
    Artifact del dominio — producto inmutable del trabajo del agente.

    Un Artifact es un resultado concreto y referenciable producido durante
    la ejecución de una task. Es inmutable porque representa un hecho histórico:
    el agente procesó esto y produjo esto otro, en este momento.

    Si el usuario pide una versión actualizada, se crea un nuevo Artifact
    con una referencia al anterior (parent_artifact_id).

    Los Artifacts son el mecanismo de memoria a medio plazo del agente:
    la Session los referencia, el ContextBuilder puede incluirlos en el
    contexto del LLM, y el usuario puede acceder a ellos desde la UI.
    """

    artifact_id: ArtifactId
    """ID único del artifact."""

    artifact_type: ArtifactType
    """Tipo semántico del artifact."""

    display_name: str
    """
    Nombre legible para mostrar en la UI.
    Ejemplo: "Resumen: Biología Celular Cap. 3", "Flashcards: Fotosíntesis".
    """

    content: ArtifactContent
    """Contenido del artifact."""

    metadata: ArtifactMetadata
    """Metadata de búsqueda y visualización."""

    session_id: SessionId
    """ID de la sesión en la que fue producido."""

    source_task_id: TaskId
    """ID de la task que produjo este artifact."""

    source_turn_id: TurnId
    """ID del turn en el que fue solicitado."""

    created_at: datetime
    """Timestamp de creación (UTC)."""

    parent_artifact_id: ArtifactId | None = None
    """
    ID del artifact del que este deriva (si es una versión actualizada).
    None si es un artifact original.
    """

    def for_llm_context(
        self,
        *,
        max_chars: int | None = None,
        include_metadata: bool = True,
    ) -> str:
        """
        Prepara el artifact para incluir en el contexto del LLM.

        Genera una representación de texto que el LLM puede procesar,
        incluyendo el tipo, nombre, y contenido (posiblemente truncado).

        Args:
            max_chars:        Límite de caracteres del contenido.
            include_metadata: Si True, incluye metadata relevante (fuente, etc.).

        Returns:
            String con la representación del artifact para el LLM.
        """
        parts: list[str] = [
            f"[{self.artifact_type.value.upper()}: {self.display_name}]",
        ]

        if include_metadata:
            if self.metadata.source_filename:
                parts.append(f"Fuente: {self.metadata.source_filename}")
            elif self.metadata.source_url:
                parts.append(f"Fuente: {self.metadata.source_url}")
            if self.metadata.item_count is not None:
                parts.append(f"Items: {self.metadata.item_count}")

        parts.append("")  # línea en blanco
        parts.append(self.content.for_llm_context(max_chars=max_chars))

        return "\n".join(parts)

    def to_summary(self) -> ArtifactSummary:
        """
        Crea un ArtifactSummary ligero para listados en UI.

        Returns:
            ArtifactSummary con metadata pero sin el contenido completo.
        """
        return ArtifactSummary(
            artifact_id=self.artifact_id,
            artifact_type=self.artifact_type,
            display_name=self.display_name,
            session_id=self.session_id,
            source_task_id=self.source_task_id,
            created_at=self.created_at,
            content_size_chars=self.content.size_chars(),
            estimated_tokens=self.content.estimated_tokens(),
            is_truncated=self.content.is_truncated,
            item_count=self.metadata.item_count,
            source_filename=self.metadata.source_filename,
            tags=self.metadata.tags,
        )

    def to_storage_dict(self) -> dict[str, Any]:
        """
        Serializa el artifact para persistencia en SQLite.

        El ArtifactStorePort usa este dict para construir las queries de INSERT.

        Returns:
            Dict con todos los campos del artifact serializados.
        """
        return {
            "artifact_id": self.artifact_id.to_str(),
            "artifact_type": self.artifact_type.value,
            "display_name": self.display_name,
            "content_type": self.content.content_type.value,
            "raw_content": self.content.raw_content,
            "is_truncated": self.content.is_truncated,
            "original_size_chars": self.content.original_size_chars,
            "session_id": self.session_id.to_str(),
            "source_task_id": self.source_task_id.to_str(),
            "source_turn_id": self.source_turn_id.to_str(),
            "created_at": self.created_at.isoformat(),
            "parent_artifact_id": (
                self.parent_artifact_id.to_str()
                if self.parent_artifact_id else None
            ),
            "metadata": self.metadata.to_index_dict(),
        }

    # =========================================================================
    # FACTORIES
    # =========================================================================

    @classmethod
    def create_summary(
        cls,
        *,
        display_name: str,
        content: str,
        session_id: SessionId,
        source_task_id: TaskId,
        source_turn_id: TurnId,
        source_filename: str | None = None,
        source_url: str | None = None,
        source_tool_id: str | None = None,
        tags: tuple[str, ...] = (),
        language: str = "es",
    ) -> Artifact:
        """
        Factory: crea un artifact de tipo SUMMARY.

        Args:
            display_name:    Nombre legible del resumen.
            content:         Texto del resumen (Markdown).
            session_id:      ID de la sesión.
            source_task_id:  ID de la task que lo produjo.
            source_turn_id:  ID del turn que lo solicitó.
            source_filename: Nombre del archivo resumido (si aplica).
            source_url:      URL resumida (si aplica).
            source_tool_id:  ID de la tool que generó el resumen.
            tags:            Tags semánticos.
            language:        Idioma del resumen.

        Returns:
            Artifact de tipo SUMMARY.
        """
        return cls(
            artifact_id=ArtifactId.generate(),
            artifact_type=ArtifactType.SUMMARY,
            display_name=display_name,
            content=ArtifactContent.from_markdown(content),
            metadata=ArtifactMetadata(
                source_filename=source_filename,
                source_url=source_url,
                source_tool_id=source_tool_id,
                tags=tags,
                language=language,
            ),
            session_id=session_id,
            source_task_id=source_task_id,
            source_turn_id=source_turn_id,
            created_at=datetime.now(tz=timezone.utc),
        )

    @classmethod
    def create_search_results(
        cls,
        *,
        display_name: str,
        results: list[dict[str, Any]],
        query: str,
        session_id: SessionId,
        source_task_id: TaskId,
        source_turn_id: TurnId,
        source_tool_id: str = "web_search",
    ) -> Artifact:
        """
        Factory: crea un artifact de tipo SEARCH_RESULTS.

        Args:
            display_name:   Nombre legible.
            results:        Lista de resultados de búsqueda.
            query:          Query que originó los resultados.
            session_id:     ID de la sesión.
            source_task_id: ID de la task.
            source_turn_id: ID del turn.
            source_tool_id: ID de la tool de búsqueda.

        Returns:
            Artifact de tipo SEARCH_RESULTS.
        """
        return cls(
            artifact_id=ArtifactId.generate(),
            artifact_type=ArtifactType.SEARCH_RESULTS,
            display_name=display_name,
            content=ArtifactContent.from_structured_list(results),
            metadata=ArtifactMetadata(
                source_tool_id=source_tool_id,
                item_count=len(results),
                extra={"query": query},
            ),
            session_id=session_id,
            source_task_id=source_task_id,
            source_turn_id=source_turn_id,
            created_at=datetime.now(tz=timezone.utc),
        )

    @classmethod
    def create_flashcards(
        cls,
        *,
        display_name: str,
        cards: list[dict[str, str]],
        session_id: SessionId,
        source_task_id: TaskId,
        source_turn_id: TurnId,
        source_filename: str | None = None,
        tags: tuple[str, ...] = (),
    ) -> Artifact:
        """
        Factory: crea un artifact de tipo FLASHCARDS.

        Args:
            display_name:    Nombre legible del deck.
            cards:           Lista de dicts con 'front' y 'back'.
            session_id:      ID de la sesión.
            source_task_id:  ID de la task.
            source_turn_id:  ID del turn.
            source_filename: Archivo de origen si aplica.
            tags:            Tags semánticos del contenido.

        Returns:
            Artifact de tipo FLASHCARDS.
        """
        return cls(
            artifact_id=ArtifactId.generate(),
            artifact_type=ArtifactType.FLASHCARDS,
            display_name=display_name,
            content=ArtifactContent.from_structured_list(cards),
            metadata=ArtifactMetadata(
                source_filename=source_filename,
                source_tool_id="flashcard_generate",
                item_count=len(cards),
                tags=tags,
            ),
            session_id=session_id,
            source_task_id=source_task_id,
            source_turn_id=source_turn_id,
            created_at=datetime.now(tz=timezone.utc),
        )

    @classmethod
    def create_comparison_table(
        cls,
        *,
        display_name: str,
        table_data: dict[str, Any],
        session_id: SessionId,
        source_task_id: TaskId,
        source_turn_id: TurnId,
        tags: tuple[str, ...] = (),
    ) -> Artifact:
        """
        Factory: crea un artifact de tipo COMPARISON_TABLE.

        Args:
            display_name:   Nombre legible de la tabla.
            table_data:     Dict con la estructura de la tabla (headers + rows).
            session_id:     ID de la sesión.
            source_task_id: ID de la task.
            source_turn_id: ID del turn.
            tags:           Tags semánticos.

        Returns:
            Artifact de tipo COMPARISON_TABLE.
        """
        return cls(
            artifact_id=ArtifactId.generate(),
            artifact_type=ArtifactType.COMPARISON_TABLE,
            display_name=display_name,
            content=ArtifactContent.from_json(table_data),
            metadata=ArtifactMetadata(
                source_tool_id="compare_options",
                tags=tags,
            ),
            session_id=session_id,
            source_task_id=source_task_id,
            source_turn_id=source_turn_id,
            created_at=datetime.now(tz=timezone.utc),
        )

    @classmethod
    def create_generated_file(
        cls,
        *,
        display_name: str,
        file_path: str,
        session_id: SessionId,
        source_task_id: TaskId,
        source_turn_id: TurnId,
        file_size_bytes: int | None = None,
        file_mime_type: str | None = None,
    ) -> Artifact:
        """
        Factory: crea un artifact de tipo GENERATED_FILE.

        Args:
            display_name:    Nombre legible del archivo.
            file_path:       Path absoluto del archivo generado.
            session_id:      ID de la sesión.
            source_task_id:  ID de la task.
            source_turn_id:  ID del turn.
            file_size_bytes: Tamaño del archivo en bytes.
            file_mime_type:  MIME type del archivo.

        Returns:
            Artifact de tipo GENERATED_FILE.
        """
        return cls(
            artifact_id=ArtifactId.generate(),
            artifact_type=ArtifactType.GENERATED_FILE,
            display_name=display_name,
            content=ArtifactContent.from_file_path(file_path),
            metadata=ArtifactMetadata(
                source_filename=file_path.split("/")[-1],
                source_tool_id="file_write",
                file_size_bytes=file_size_bytes,
                file_mime_type=file_mime_type,
            ),
            session_id=session_id,
            source_task_id=source_task_id,
            source_turn_id=source_turn_id,
            created_at=datetime.now(tz=timezone.utc),
        )

    @classmethod
    def create_analysis(
        cls,
        *,
        display_name: str,
        content: str,
        session_id: SessionId,
        source_task_id: TaskId,
        source_turn_id: TurnId,
        source_filename: str | None = None,
        source_tool_id: str | None = None,
        tags: tuple[str, ...] = (),
        language: str = "es",
    ) -> Artifact:
        """
        Factory: crea un artifact de tipo ANALYSIS.

        Args:
            display_name:    Nombre legible del análisis.
            content:         Texto del análisis (Markdown).
            session_id:      ID de la sesión.
            source_task_id:  ID de la task.
            source_turn_id:  ID del turn.
            source_filename: Archivo analizado si aplica.
            source_tool_id:  Tool que realizó el análisis.
            tags:            Tags semánticos.
            language:        Idioma del análisis.

        Returns:
            Artifact de tipo ANALYSIS.
        """
        return cls(
            artifact_id=ArtifactId.generate(),
            artifact_type=ArtifactType.ANALYSIS,
            display_name=display_name,
            content=ArtifactContent.from_markdown(content),
            metadata=ArtifactMetadata(
                source_filename=source_filename,
                source_tool_id=source_tool_id,
                tags=tags,
                language=language,
            ),
            session_id=session_id,
            source_task_id=source_task_id,
            source_turn_id=source_turn_id,
            created_at=datetime.now(tz=timezone.utc),
        )

    @classmethod
    def create_web_page(
        cls,
        *,
        display_name: str,
        content: str,
        url: str,
        session_id: SessionId,
        source_task_id: TaskId,
        source_turn_id: TurnId,
    ) -> Artifact:
        """
        Factory: crea un artifact de tipo WEB_PAGE.

        Args:
            display_name:   Nombre legible (generalmente el título de la página).
            content:        Texto extraído de la página.
            url:            URL de la página.
            session_id:     ID de la sesión.
            source_task_id: ID de la task.
            source_turn_id: ID del turn.

        Returns:
            Artifact de tipo WEB_PAGE.
        """
        return cls(
            artifact_id=ArtifactId.generate(),
            artifact_type=ArtifactType.WEB_PAGE,
            display_name=display_name,
            content=ArtifactContent.from_text(content),
            metadata=ArtifactMetadata(
                source_url=url,
                source_tool_id="web_fetch",
            ),
            session_id=session_id,
            source_task_id=source_task_id,
            source_turn_id=source_turn_id,
            created_at=datetime.now(tz=timezone.utc),
        )

    @classmethod
    def create_note(
        cls,
        *,
        display_name: str,
        content: str,
        session_id: SessionId,
        source_task_id: TaskId,
        source_turn_id: TurnId,
        tags: tuple[str, ...] = (),
    ) -> Artifact:
        """
        Factory: crea un artifact de tipo NOTE.

        Args:
            display_name:   Nombre legible de la nota.
            content:        Contenido de la nota (Markdown).
            session_id:     ID de la sesión.
            source_task_id: ID de la task.
            source_turn_id: ID del turn.
            tags:           Tags semánticos.

        Returns:
            Artifact de tipo NOTE.
        """
        return cls(
            artifact_id=ArtifactId.generate(),
            artifact_type=ArtifactType.NOTE,
            display_name=display_name,
            content=ArtifactContent.from_markdown(content),
            metadata=ArtifactMetadata(
                source_tool_id="notes_create",
                tags=tags,
            ),
            session_id=session_id,
            source_task_id=source_task_id,
            source_turn_id=source_turn_id,
            created_at=datetime.now(tz=timezone.utc),
        )


# =============================================================================
# ARTIFACT SUMMARY (para listados en UI)
# =============================================================================


@dataclass(frozen=True)
class ArtifactSummary:
    """
    Vista ligera de un Artifact para listados en la UI.

    No incluye el contenido completo — solo la metadata necesaria para
    mostrar el artifact en una lista y decidir si cargarlo completo.
    """

    artifact_id: ArtifactId
    artifact_type: ArtifactType
    display_name: str
    session_id: SessionId
    source_task_id: TaskId
    created_at: datetime
    content_size_chars: int
    estimated_tokens: int
    is_truncated: bool = False
    item_count: int | None = None
    source_filename: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def type_label(self) -> str:
        """Etiqueta legible del tipo de artifact para la UI."""
        labels: dict[ArtifactType, str] = {
            ArtifactType.SUMMARY: "Resumen",
            ArtifactType.SEARCH_RESULTS: "Resultados de búsqueda",
            ArtifactType.ANALYSIS: "Análisis",
            ArtifactType.COMPARISON_TABLE: "Tabla comparativa",
            ArtifactType.FLASHCARDS: "Flashcards",
            ArtifactType.QUIZ: "Quiz",
            ArtifactType.STUDY_PLAN: "Plan de estudio",
            ArtifactType.CONCEPT_MAP: "Mapa conceptual",
            ArtifactType.GENERATED_FILE: "Archivo generado",
            ArtifactType.NOTE: "Nota",
            ArtifactType.ACTION_LOG: "Registro de acciones",
            ArtifactType.EXTRACTED_TEXT: "Texto extraído",
            ArtifactType.WEB_PAGE: "Página web",
        }
        return labels.get(self.artifact_type, self.artifact_type.value)

    def to_display_dict(self) -> dict[str, Any]:
        """Serializa el summary para la UI."""
        return {
            "artifact_id": self.artifact_id.to_str(),
            "artifact_type": self.artifact_type.value,
            "type_label": self.type_label,
            "display_name": self.display_name,
            "created_at": self.created_at.isoformat(),
            "content_size_chars": self.content_size_chars,
            "estimated_tokens": self.estimated_tokens,
            "is_truncated": self.is_truncated,
            "item_count": self.item_count,
            "source_filename": self.source_filename,
            "tags": list(self.tags),
        }