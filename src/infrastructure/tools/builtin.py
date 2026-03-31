"""
Registro de tools built-in de HiperForge User.

Este módulo es el único punto donde se instancian y declaran todas las
tools concretas disponibles en V1. La función get_all_builtin_tools()
es llamada por startup.py durante el bootstrap para registrarlas en el
InMemoryToolRegistry.

Tools disponibles en V1:
  - filesystem: file_read, file_list, file_search, file_write
  - search:     web_search, web_fetch
  - study:      flashcard_generate, quiz_generate
  - documents:  document_read, document_summarize
  - notes:      notes_create

Cada tool se define como un ToolSchema con su ToolCapability, nivel de
riesgo, categoría, y plataforma. El executor es la instancia del ToolPort
concreto que ejecuta la acción real.

En esta versión inicial, las tools de filesystem y desktop están
deshabilitadas por defecto hasta que el usuario las apruebe en el primer uso.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forge_core.tools.protocol import (
    MutationType,
    Platform,
    RiskLevel,
    ToolCapability,
    ToolCategory,
    ToolSchema,
)

from src.ports.outbound.storage_port import SessionStorePort


def get_all_builtin_tools(
    allowed_base_path: Path,
    session_store: SessionStorePort,
) -> list[ToolSchema]:
    """
    Retorna los schemas de todas las tools built-in disponibles en V1.

    Las tools se construyen con acceso al allowed_base_path del usuario
    para que las tools de filesystem puedan validar paths permitidos.

    Args:
        allowed_base_path: Directorio base permitido (típicamente Path.home()).
        session_store:     Store de sesiones (para tools que necesitan contexto).

    Returns:
        Lista de ToolSchema listos para registrar en el InMemoryToolRegistry.
    """
    tools: list[ToolSchema] = []

    # ─── FILESYSTEM ──────────────────────────────────────────────────────────
    tools.append(_make_file_read_schema(allowed_base_path))
    tools.append(_make_file_list_schema(allowed_base_path))
    tools.append(_make_file_search_schema(allowed_base_path))
    tools.append(_make_file_write_schema(allowed_base_path))

    # ─── SEARCH ──────────────────────────────────────────────────────────────
    tools.append(_make_web_search_schema())
    tools.append(_make_web_fetch_schema())

    # ─── STUDY ───────────────────────────────────────────────────────────────
    tools.append(_make_flashcard_generate_schema())
    tools.append(_make_quiz_generate_schema())

    # ─── DOCUMENTS ───────────────────────────────────────────────────────────
    tools.append(_make_document_read_schema(allowed_base_path))
    tools.append(_make_document_summarize_schema())

    # ─── NOTES ───────────────────────────────────────────────────────────────
    tools.append(_make_notes_create_schema())

    return tools


# =============================================================================
# FILESYSTEM TOOLS
# =============================================================================


def _make_file_read_schema(base_path: Path) -> ToolSchema:
    """Lee el contenido de un archivo de texto del sistema del usuario."""
    return ToolSchema(
        tool_id="file_read",
        display_name="Leer archivo",
        description=(
            "Lee el contenido de un archivo de texto en el sistema del usuario. "
            "Compatible con .txt, .md, .py, .json, .csv, .yaml y formatos de texto plano."
        ),
        category=ToolCategory.FILESYSTEM,
        capability=ToolCapability(
            risk_classification=RiskLevel.LOW,
            mutation_type=MutationType.NONE,
            platform=Platform.ALL,
            requires_sandbox=False,
        ),
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Ruta absoluta o relativa al archivo a leer.",
                },
                "encoding": {
                    "type": "string",
                    "description": "Codificación del archivo. Por defecto: utf-8.",
                    "default": "utf-8",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Máximo de caracteres a retornar. Por defecto: 50000.",
                    "default": 50000,
                },
            },
            "required": ["file_path"],
        },
        enabled=True,
        executor=_FileReadExecutor(base_path),
    )


def _make_file_list_schema(base_path: Path) -> ToolSchema:
    """Lista los archivos de un directorio."""
    return ToolSchema(
        tool_id="file_list",
        display_name="Listar directorio",
        description=(
            "Lista los archivos y subdirectorios de una carpeta del sistema del usuario."
        ),
        category=ToolCategory.FILESYSTEM,
        capability=ToolCapability(
            risk_classification=RiskLevel.NONE,
            mutation_type=MutationType.NONE,
            platform=Platform.ALL,
            requires_sandbox=False,
        ),
        parameters={
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Ruta del directorio a listar.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Patrón glob para filtrar (ej: '*.py'). Opcional.",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Si True, lista subdirectorios también.",
                    "default": False,
                },
            },
            "required": ["directory"],
        },
        enabled=True,
        executor=_FileListExecutor(base_path),
    )


def _make_file_search_schema(base_path: Path) -> ToolSchema:
    """Busca archivos por nombre o contenido."""
    return ToolSchema(
        tool_id="file_search",
        display_name="Buscar archivos",
        description="Busca archivos por nombre o contenido de texto en el sistema del usuario.",
        category=ToolCategory.FILESYSTEM,
        capability=ToolCapability(
            risk_classification=RiskLevel.NONE,
            mutation_type=MutationType.NONE,
            platform=Platform.ALL,
            requires_sandbox=False,
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Texto a buscar en nombres de archivo o contenido.",
                },
                "search_dir": {
                    "type": "string",
                    "description": "Directorio donde buscar. Por defecto: home del usuario.",
                },
                "file_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extensiones a incluir (ej: ['pdf', 'docx']). Vacío = todas.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Máximo de resultados a retornar.",
                    "default": 20,
                },
            },
            "required": ["query"],
        },
        enabled=True,
        executor=_FileSearchExecutor(base_path),
    )


def _make_file_write_schema(base_path: Path) -> ToolSchema:
    """Escribe o crea un archivo de texto."""
    return ToolSchema(
        tool_id="file_write",
        display_name="Escribir archivo",
        description=(
            "Crea o sobreescribe un archivo de texto en el sistema del usuario. "
            "Requiere aprobación del usuario antes de ejecutarse."
        ),
        category=ToolCategory.FILESYSTEM,
        capability=ToolCapability(
            risk_classification=RiskLevel.MEDIUM,
            mutation_type=MutationType.WRITE,
            platform=Platform.ALL,
            requires_sandbox=False,
        ),
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Ruta del archivo a escribir.",
                },
                "content": {
                    "type": "string",
                    "description": "Contenido a escribir en el archivo.",
                },
                "mode": {
                    "type": "string",
                    "description": "'overwrite' para sobreescribir, 'append' para añadir al final.",
                    "default": "overwrite",
                },
            },
            "required": ["file_path", "content"],
        },
        enabled=True,
        executor=_FileWriteExecutor(base_path),
    )


# =============================================================================
# SEARCH TOOLS
# =============================================================================


def _make_web_search_schema() -> ToolSchema:
    """Busca información en la web."""
    return ToolSchema(
        tool_id="web_search",
        display_name="Buscar en la web",
        description=(
            "Realiza una búsqueda web y retorna los resultados más relevantes "
            "con títulos, URLs y snippets."
        ),
        category=ToolCategory.SEARCH,
        capability=ToolCapability(
            risk_classification=RiskLevel.NONE,
            mutation_type=MutationType.NONE,
            platform=Platform.ALL,
            requires_sandbox=False,
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Consulta de búsqueda.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Número máximo de resultados.",
                    "default": 10,
                },
                "language": {
                    "type": "string",
                    "description": "Idioma de los resultados (ej: 'es', 'en').",
                    "default": "es",
                },
            },
            "required": ["query"],
        },
        enabled=True,
        executor=_WebSearchExecutor(),
    )


def _make_web_fetch_schema() -> ToolSchema:
    """Obtiene el contenido de una URL específica."""
    return ToolSchema(
        tool_id="web_fetch",
        display_name="Obtener página web",
        description=(
            "Descarga y extrae el texto de una página web específica."
        ),
        category=ToolCategory.SEARCH,
        capability=ToolCapability(
            risk_classification=RiskLevel.LOW,
            mutation_type=MutationType.NONE,
            platform=Platform.ALL,
            requires_sandbox=False,
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL de la página a obtener.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Máximo de caracteres del contenido extraído.",
                    "default": 20000,
                },
            },
            "required": ["url"],
        },
        enabled=True,
        executor=_WebFetchExecutor(),
    )


# =============================================================================
# STUDY TOOLS
# =============================================================================


def _make_flashcard_generate_schema() -> ToolSchema:
    """Genera flashcards de estudio desde texto o documentos."""
    return ToolSchema(
        tool_id="flashcard_generate",
        display_name="Generar flashcards",
        description=(
            "Genera tarjetas de estudio (pregunta/respuesta) a partir de "
            "texto, notas, o el contenido de un documento."
        ),
        category=ToolCategory.STUDY,
        capability=ToolCapability(
            risk_classification=RiskLevel.NONE,
            mutation_type=MutationType.NONE,
            platform=Platform.ALL,
            requires_sandbox=False,
        ),
        parameters={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Texto del que generar las flashcards.",
                },
                "count": {
                    "type": "integer",
                    "description": "Número de flashcards a generar.",
                    "default": 15,
                },
                "difficulty": {
                    "type": "string",
                    "description": "Nivel de dificultad: 'basic', 'intermediate', 'advanced'.",
                    "default": "intermediate",
                },
                "language": {
                    "type": "string",
                    "description": "Idioma de las flashcards.",
                    "default": "es",
                },
            },
            "required": ["content"],
        },
        enabled=True,
        executor=_FlashcardGenerateExecutor(),
    )


def _make_quiz_generate_schema() -> ToolSchema:
    """Genera un quiz de opción múltiple desde texto o notas."""
    return ToolSchema(
        tool_id="quiz_generate",
        display_name="Generar quiz",
        description=(
            "Genera preguntas de opción múltiple con respuestas correctas e "
            "incorrectas a partir de un texto o tema."
        ),
        category=ToolCategory.STUDY,
        capability=ToolCapability(
            risk_classification=RiskLevel.NONE,
            mutation_type=MutationType.NONE,
            platform=Platform.ALL,
            requires_sandbox=False,
        ),
        parameters={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Texto del que generar el quiz.",
                },
                "question_count": {
                    "type": "integer",
                    "description": "Número de preguntas a generar.",
                    "default": 10,
                },
                "options_per_question": {
                    "type": "integer",
                    "description": "Opciones por pregunta (incluyendo la correcta).",
                    "default": 4,
                },
            },
            "required": ["content"],
        },
        enabled=True,
        executor=_QuizGenerateExecutor(),
    )


# =============================================================================
# DOCUMENT TOOLS
# =============================================================================


def _make_document_read_schema(base_path: Path) -> ToolSchema:
    """Lee y extrae texto de documentos PDF, DOCX, etc."""
    return ToolSchema(
        tool_id="document_read",
        display_name="Leer documento",
        description=(
            "Lee y extrae el texto de documentos PDF, DOCX, PPTX y otros formatos. "
            "Retorna el texto extraído para análisis o resumen."
        ),
        category=ToolCategory.DOCUMENTS,
        capability=ToolCapability(
            risk_classification=RiskLevel.LOW,
            mutation_type=MutationType.NONE,
            platform=Platform.ALL,
            requires_sandbox=False,
        ),
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Ruta del documento a leer.",
                },
                "pages": {
                    "type": "string",
                    "description": "Páginas a extraer (ej: '1-5', 'all'). Por defecto: 'all'.",
                    "default": "all",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Máximo de caracteres a retornar.",
                    "default": 50000,
                },
            },
            "required": ["file_path"],
        },
        enabled=True,
        executor=_DocumentReadExecutor(base_path),
    )


def _make_document_summarize_schema() -> ToolSchema:
    """Genera un resumen estructurado de un texto o documento."""
    return ToolSchema(
        tool_id="document_summarize",
        display_name="Resumir documento",
        description=(
            "Genera un resumen estructurado y conciso de un texto largo o documento."
        ),
        category=ToolCategory.DOCUMENTS,
        capability=ToolCapability(
            risk_classification=RiskLevel.NONE,
            mutation_type=MutationType.NONE,
            platform=Platform.ALL,
            requires_sandbox=False,
        ),
        parameters={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Texto a resumir.",
                },
                "style": {
                    "type": "string",
                    "description": "Estilo del resumen: 'bullet_points', 'paragraph', 'outline'.",
                    "default": "paragraph",
                },
                "max_length": {
                    "type": "string",
                    "description": "Longitud máxima: 'short' (~200 palabras), 'medium' (~500), 'long' (~1000).",
                    "default": "medium",
                },
                "language": {
                    "type": "string",
                    "description": "Idioma del resumen.",
                    "default": "es",
                },
            },
            "required": ["content"],
        },
        enabled=True,
        executor=_DocumentSummarizeExecutor(),
    )


# =============================================================================
# NOTES TOOLS
# =============================================================================


def _make_notes_create_schema() -> ToolSchema:
    """Crea una nota de texto estructurada."""
    return ToolSchema(
        tool_id="notes_create",
        display_name="Crear nota",
        description="Crea una nota o apunte a partir de la conversación o de texto proporcionado.",
        category=ToolCategory.PRODUCTIVITY,
        capability=ToolCapability(
            risk_classification=RiskLevel.NONE,
            mutation_type=MutationType.NONE,
            platform=Platform.ALL,
            requires_sandbox=False,
        ),
        parameters={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Título de la nota.",
                },
                "content": {
                    "type": "string",
                    "description": "Contenido de la nota en Markdown.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags para clasificar la nota.",
                },
            },
            "required": ["title", "content"],
        },
        enabled=True,
        executor=_NotesCreateExecutor(),
    )


# =============================================================================
# EXECUTORS — implementaciones mínimas de ToolPort
# =============================================================================
# En V1, los executors son implementaciones directas simples.
# En V2, cada uno se extraerá a su propio módulo en infrastructure/tools/


class _FileReadExecutor:
    """Ejecutor para file_read."""

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    async def execute(self, tool_input: Any) -> Any:
        from forge_core.tools.protocol import ToolOutput
        file_path = Path(tool_input.arguments.get("file_path", ""))
        max_chars = tool_input.arguments.get("max_chars", 50_000)
        encoding = tool_input.arguments.get("encoding", "utf-8")

        if not file_path.is_absolute():
            file_path = self._base_path / file_path

        if not file_path.exists():
            return ToolOutput.error(f"Archivo no encontrado: {file_path}")

        if not str(file_path).startswith(str(self._base_path)):
            return ToolOutput.error("Acceso denegado: ruta fuera del directorio permitido.")

        try:
            content = file_path.read_text(encoding=encoding, errors="replace")
            if len(content) > max_chars:
                content = content[:max_chars] + f"\n\n[...truncado a {max_chars} caracteres]"
            return ToolOutput.success(
                content=content,
                metadata={"file_path": str(file_path), "size_chars": len(content)},
            )
        except Exception as e:
            return ToolOutput.error(f"Error al leer el archivo: {e}")


class _FileListExecutor:
    """Ejecutor para file_list."""

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    async def execute(self, tool_input: Any) -> Any:
        from forge_core.tools.protocol import ToolOutput
        directory = Path(tool_input.arguments.get("directory", str(self._base_path)))
        pattern = tool_input.arguments.get("pattern", "*")
        recursive = tool_input.arguments.get("recursive", False)

        if not directory.is_absolute():
            directory = self._base_path / directory

        if not str(directory).startswith(str(self._base_path)):
            return ToolOutput.error("Acceso denegado: directorio fuera del permitido.")

        if not directory.exists() or not directory.is_dir():
            return ToolOutput.error(f"Directorio no encontrado: {directory}")

        try:
            if recursive:
                entries = list(directory.rglob(pattern))[:200]
            else:
                entries = list(directory.glob(pattern))[:200]

            result = []
            for entry in sorted(entries):
                result.append({
                    "name": entry.name,
                    "type": "dir" if entry.is_dir() else "file",
                    "path": str(entry),
                    "size_bytes": entry.stat().st_size if entry.is_file() else None,
                })

            import json
            return ToolOutput.success(
                content=json.dumps(result, ensure_ascii=False, indent=2),
                metadata={"directory": str(directory), "count": len(result)},
            )
        except Exception as e:
            return ToolOutput.error(f"Error al listar directorio: {e}")


class _FileSearchExecutor:
    """Ejecutor para file_search."""

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    async def execute(self, tool_input: Any) -> Any:
        from forge_core.tools.protocol import ToolOutput
        query = tool_input.arguments.get("query", "")
        search_dir = Path(tool_input.arguments.get("search_dir", str(self._base_path)))
        file_types = tool_input.arguments.get("file_types", [])
        max_results = tool_input.arguments.get("max_results", 20)

        if not str(search_dir).startswith(str(self._base_path)):
            return ToolOutput.error("Acceso denegado: directorio fuera del permitido.")

        try:
            results = []
            extensions = {f".{ext.lstrip('.')}" for ext in file_types} if file_types else None
            query_lower = query.lower()

            for path in search_dir.rglob("*"):
                if len(results) >= max_results:
                    break
                if not path.is_file():
                    continue
                if extensions and path.suffix.lower() not in extensions:
                    continue
                if query_lower in path.name.lower():
                    results.append({"name": path.name, "path": str(path), "match": "name"})

            import json
            return ToolOutput.success(
                content=json.dumps(results, ensure_ascii=False, indent=2),
                metadata={"query": query, "count": len(results)},
            )
        except Exception as e:
            return ToolOutput.error(f"Error en la búsqueda: {e}")


class _FileWriteExecutor:
    """Ejecutor para file_write."""

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    async def execute(self, tool_input: Any) -> Any:
        from forge_core.tools.protocol import ToolOutput
        file_path = Path(tool_input.arguments.get("file_path", ""))
        content = tool_input.arguments.get("content", "")
        mode = tool_input.arguments.get("mode", "overwrite")

        if not file_path.is_absolute():
            file_path = self._base_path / file_path

        if not str(file_path).startswith(str(self._base_path)):
            return ToolOutput.error("Acceso denegado: ruta fuera del directorio permitido.")

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            write_mode = "a" if mode == "append" else "w"
            file_path.write_text(content, encoding="utf-8") if write_mode == "w" else \
                open(file_path, "a", encoding="utf-8").write(content)
            return ToolOutput.success(
                content=f"Archivo guardado: {file_path}",
                metadata={"file_path": str(file_path), "mode": mode, "size_chars": len(content)},
            )
        except Exception as e:
            return ToolOutput.error(f"Error al escribir el archivo: {e}")


class _WebSearchExecutor:
    """Ejecutor para web_search — stub en V1 (requiere API key de búsqueda)."""

    async def execute(self, tool_input: Any) -> Any:
        from forge_core.tools.protocol import ToolOutput
        query = tool_input.arguments.get("query", "")
        # En V1, retorna un mensaje indicando que se necesita configurar la API
        return ToolOutput.success(
            content=(
                f"Búsqueda web para: '{query}'\n\n"
                "Esta función requiere configurar una API de búsqueda (SerpAPI, Brave, etc.). "
                "Configura FORGE_USER__SEARCH__API_KEY para habilitarla."
            ),
            metadata={"query": query, "stub": True},
        )


class _WebFetchExecutor:
    """Ejecutor para web_fetch."""

    async def execute(self, tool_input: Any) -> Any:
        from forge_core.tools.protocol import ToolOutput
        url = tool_input.arguments.get("url", "")
        max_chars = tool_input.arguments.get("max_chars", 20_000)

        try:
            import urllib.request
            import urllib.error
            headers = {"User-Agent": "Mozilla/5.0 (compatible; ForgeUser/1.0)"}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                raw = response.read().decode("utf-8", errors="replace")

            # Extracción básica de texto (remover HTML)
            import re
            text = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

            if len(text) > max_chars:
                text = text[:max_chars] + f"\n\n[...truncado a {max_chars} caracteres]"

            return ToolOutput.success(
                content=text,
                metadata={"url": url, "size_chars": len(text)},
            )
        except Exception as e:
            return ToolOutput.error(f"Error al obtener la página: {e}")


class _FlashcardGenerateExecutor:
    """Ejecutor para flashcard_generate — delega al LLM via prompt estructurado."""

    async def execute(self, tool_input: Any) -> Any:
        from forge_core.tools.protocol import ToolOutput
        content = tool_input.arguments.get("content", "")
        count = tool_input.arguments.get("count", 15)
        difficulty = tool_input.arguments.get("difficulty", "intermediate")
        language = tool_input.arguments.get("language", "es")

        # Retorna instrucción para que el LLM genere las flashcards
        # El ConversationCoordinator procesa esto como un artifact
        import json
        prompt_result = {
            "instruction": "generate_flashcards",
            "content_preview": content[:500],
            "count": count,
            "difficulty": difficulty,
            "language": language,
            "status": "pending_llm_generation",
        }
        return ToolOutput.success(
            content=json.dumps(prompt_result, ensure_ascii=False),
            metadata={"count": count, "difficulty": difficulty},
        )


class _QuizGenerateExecutor:
    """Ejecutor para quiz_generate."""

    async def execute(self, tool_input: Any) -> Any:
        from forge_core.tools.protocol import ToolOutput
        import json
        content = tool_input.arguments.get("content", "")
        question_count = tool_input.arguments.get("question_count", 10)
        options = tool_input.arguments.get("options_per_question", 4)

        result = {
            "instruction": "generate_quiz",
            "content_preview": content[:500],
            "question_count": question_count,
            "options_per_question": options,
            "status": "pending_llm_generation",
        }
        return ToolOutput.success(
            content=json.dumps(result, ensure_ascii=False),
            metadata={"question_count": question_count},
        )


class _DocumentReadExecutor:
    """Ejecutor para document_read."""

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    async def execute(self, tool_input: Any) -> Any:
        from forge_core.tools.protocol import ToolOutput
        file_path = Path(tool_input.arguments.get("file_path", ""))
        max_chars = tool_input.arguments.get("max_chars", 50_000)

        if not file_path.is_absolute():
            file_path = self._base_path / file_path

        if not str(file_path).startswith(str(self._base_path)):
            return ToolOutput.error("Acceso denegado: ruta fuera del directorio permitido.")

        if not file_path.exists():
            return ToolOutput.error(f"Archivo no encontrado: {file_path}")

        suffix = file_path.suffix.lower()
        try:
            if suffix == ".pdf":
                return await self._read_pdf(file_path, max_chars)
            elif suffix in {".txt", ".md", ".rst", ".csv"}:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                if len(content) > max_chars:
                    content = content[:max_chars] + "\n\n[...truncado]"
                return ToolOutput.success(content=content, metadata={"format": suffix})
            else:
                return ToolOutput.error(
                    f"Formato '{suffix}' no soportado. "
                    f"Usa: .pdf, .txt, .md, .csv"
                )
        except Exception as e:
            return ToolOutput.error(f"Error al leer el documento: {e}")

    async def _read_pdf(self, file_path: Path, max_chars: int) -> Any:
        from forge_core.tools.protocol import ToolOutput
        try:
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
            pages_text = []
            for page in reader.pages:
                pages_text.append(page.extract_text() or "")
            content = "\n\n".join(pages_text)
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n[...truncado]"
            return ToolOutput.success(
                content=content,
                metadata={"format": "pdf", "pages": len(reader.pages)},
            )
        except ImportError:
            return ToolOutput.error(
                "La librería 'pypdf' no está instalada. "
                "Instálala con: pip install pypdf"
            )


class _DocumentSummarizeExecutor:
    """Ejecutor para document_summarize."""

    async def execute(self, tool_input: Any) -> Any:
        from forge_core.tools.protocol import ToolOutput
        import json
        content = tool_input.arguments.get("content", "")
        style = tool_input.arguments.get("style", "paragraph")
        max_length = tool_input.arguments.get("max_length", "medium")
        language = tool_input.arguments.get("language", "es")

        result = {
            "instruction": "summarize",
            "content_preview": content[:500],
            "content_length": len(content),
            "style": style,
            "max_length": max_length,
            "language": language,
            "status": "pending_llm_generation",
        }
        return ToolOutput.success(
            content=json.dumps(result, ensure_ascii=False),
            metadata={"style": style, "max_length": max_length},
        )


class _NotesCreateExecutor:
    """Ejecutor para notes_create."""

    async def execute(self, tool_input: Any) -> Any:
        from forge_core.tools.protocol import ToolOutput
        import json
        title = tool_input.arguments.get("title", "")
        content = tool_input.arguments.get("content", "")
        tags = tool_input.arguments.get("tags", [])

        note = {
            "title": title,
            "content": content,
            "tags": tags,
            "created_at": __import__("datetime").datetime.now().isoformat(),
        }
        return ToolOutput.success(
            content=json.dumps(note, ensure_ascii=False, indent=2),
            metadata={"title": title, "tags": tags},
        )