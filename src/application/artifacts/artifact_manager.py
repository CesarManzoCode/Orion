"""
ArtifactManager — gestor del ciclo de vida de artifacts del agente.

El ArtifactManager es el servicio de aplicación responsable de crear,
persistir, recuperar y gestionar los artifacts producidos por el agente.

Responsabilidades:
  1. Crear artifacts tipados desde outputs de tools (resúmenes, flashcards, etc.)
  2. Persistir artifacts en el ArtifactStorePort
  3. Registrar referencias de artifacts en la Session
  4. Recuperar artifacts para la UI y el contexto del LLM
  5. Gestionar el ciclo de vida: creación → referenciación → exportación

Diseño:
  - El ArtifactManager no contiene lógica de negocio de los artifacts.
    La entidad Artifact tiene sus propias factories y validaciones.
  - El ArtifactManager es el coordinador entre la entidad Artifact,
    el ArtifactStorePort, y la Session.
  - Es stateless — todo el estado vive en el storage.
  - Los outputs de tools se transforman en artifacts en este servicio,
    no en las tools ni en el TaskExecutor.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from forge_core.errors.types import ForgeStorageError
from forge_core.observability.logging import get_logger

from src.domain.entities.artifact import (
    Artifact,
    ArtifactContent,
    ArtifactMetadata,
    ArtifactSummary,
    ArtifactType,
)
from src.domain.entities.session import ArtifactReference, Session
from src.domain.value_objects.identifiers import (
    ArtifactId,
    SessionId,
    TaskId,
    TurnId,
)
from src.ports.outbound.storage_port import (
    ArtifactQuery,
    ArtifactStorePort,
    PaginatedResult,
)


logger = get_logger(__name__, component="artifact_manager")


class ArtifactManager:
    """
    Gestor del ciclo de vida de artifacts del agente.

    Coordina la creación, persistencia y recuperación de artifacts.
    Actúa como fachada entre el TaskExecutor (que produce outputs)
    y el ArtifactStorePort (que los persiste).
    """

    def __init__(
        self,
        artifact_store: ArtifactStorePort,
    ) -> None:
        """
        Args:
            artifact_store: Puerto de persistencia de artifacts.
        """
        self._store = artifact_store

    # =========================================================================
    # CREACIÓN DE ARTIFACTS TIPADOS
    # =========================================================================

    async def create_summary(
        self,
        *,
        display_name: str,
        content: str,
        session: Session,
        task_id: TaskId,
        turn_id: TurnId,
        source_filename: str | None = None,
        source_url: str | None = None,
        source_tool_id: str | None = None,
        tags: tuple[str, ...] = (),
        language: str = "es",
    ) -> Artifact:
        """
        Crea y persiste un artifact de tipo SUMMARY.

        Args:
            display_name:    Nombre legible del resumen.
            content:         Texto del resumen (Markdown).
            session:         Session activa.
            task_id:         Task que produjo el resumen.
            turn_id:         Turn que lo solicitó.
            source_filename: Archivo resumido (si aplica).
            source_url:      URL resumida (si aplica).
            source_tool_id:  Tool que generó el resumen.
            tags:            Tags semánticos.
            language:        Idioma del resumen.

        Returns:
            Artifact creado y persistido.
        """
        artifact = Artifact.create_summary(
            display_name=display_name,
            content=content,
            session_id=session.session_id,
            source_task_id=task_id,
            source_turn_id=turn_id,
            source_filename=source_filename,
            source_url=source_url,
            source_tool_id=source_tool_id,
            tags=tags,
            language=language,
        )
        return await self._persist_and_register(artifact, session)

    async def create_flashcards(
        self,
        *,
        display_name: str,
        cards: list[dict[str, str]],
        session: Session,
        task_id: TaskId,
        turn_id: TurnId,
        source_filename: str | None = None,
        tags: tuple[str, ...] = (),
    ) -> Artifact:
        """
        Crea y persiste un artifact de tipo FLASHCARDS.

        Args:
            display_name:    Nombre del deck de flashcards.
            cards:           Lista de dicts con 'front' y 'back'.
            session:         Session activa.
            task_id:         Task que produjo las flashcards.
            turn_id:         Turn que lo solicitó.
            source_filename: Archivo fuente (si aplica).
            tags:            Tags semánticos del contenido.

        Returns:
            Artifact creado y persistido.
        """
        artifact = Artifact.create_flashcards(
            display_name=display_name,
            cards=cards,
            session_id=session.session_id,
            source_task_id=task_id,
            source_turn_id=turn_id,
            source_filename=source_filename,
            tags=tags,
        )
        return await self._persist_and_register(artifact, session)

    async def create_search_results(
        self,
        *,
        display_name: str,
        results: list[dict[str, Any]],
        query: str,
        session: Session,
        task_id: TaskId,
        turn_id: TurnId,
        source_tool_id: str = "web_search",
    ) -> Artifact:
        """
        Crea y persiste un artifact de tipo SEARCH_RESULTS.

        Args:
            display_name:   Nombre descriptivo.
            results:        Lista de resultados de búsqueda.
            query:          Query que originó los resultados.
            session:        Session activa.
            task_id:        Task que ejecutó la búsqueda.
            turn_id:        Turn que lo solicitó.
            source_tool_id: Tool de búsqueda usada.

        Returns:
            Artifact creado y persistido.
        """
        artifact = Artifact.create_search_results(
            display_name=display_name,
            results=results,
            query=query,
            session_id=session.session_id,
            source_task_id=task_id,
            source_turn_id=turn_id,
            source_tool_id=source_tool_id,
        )
        return await self._persist_and_register(artifact, session)

    async def create_analysis(
        self,
        *,
        display_name: str,
        content: str,
        session: Session,
        task_id: TaskId,
        turn_id: TurnId,
        source_filename: str | None = None,
        source_tool_id: str | None = None,
        tags: tuple[str, ...] = (),
        language: str = "es",
    ) -> Artifact:
        """
        Crea y persiste un artifact de tipo ANALYSIS.

        Args:
            display_name:    Nombre del análisis.
            content:         Texto del análisis (Markdown).
            session:         Session activa.
            task_id:         Task que realizó el análisis.
            turn_id:         Turn que lo solicitó.
            source_filename: Archivo analizado.
            source_tool_id:  Tool que realizó el análisis.
            tags:            Tags semánticos.
            language:        Idioma del análisis.

        Returns:
            Artifact creado y persistido.
        """
        artifact = Artifact.create_analysis(
            display_name=display_name,
            content=content,
            session_id=session.session_id,
            source_task_id=task_id,
            source_turn_id=turn_id,
            source_filename=source_filename,
            source_tool_id=source_tool_id,
            tags=tags,
            language=language,
        )
        return await self._persist_and_register(artifact, session)

    async def create_comparison_table(
        self,
        *,
        display_name: str,
        table_data: dict[str, Any],
        session: Session,
        task_id: TaskId,
        turn_id: TurnId,
        tags: tuple[str, ...] = (),
    ) -> Artifact:
        """Crea y persiste un artifact de tipo COMPARISON_TABLE."""
        artifact = Artifact.create_comparison_table(
            display_name=display_name,
            table_data=table_data,
            session_id=session.session_id,
            source_task_id=task_id,
            source_turn_id=turn_id,
            tags=tags,
        )
        return await self._persist_and_register(artifact, session)

    async def create_web_page(
        self,
        *,
        display_name: str,
        content: str,
        url: str,
        session: Session,
        task_id: TaskId,
        turn_id: TurnId,
    ) -> Artifact:
        """Crea y persiste un artifact de tipo WEB_PAGE."""
        artifact = Artifact.create_web_page(
            display_name=display_name,
            content=content,
            url=url,
            session_id=session.session_id,
            source_task_id=task_id,
            source_turn_id=turn_id,
        )
        return await self._persist_and_register(artifact, session)

    async def create_generated_file(
        self,
        *,
        display_name: str,
        file_path: str,
        session: Session,
        task_id: TaskId,
        turn_id: TurnId,
        file_size_bytes: int | None = None,
        file_mime_type: str | None = None,
    ) -> Artifact:
        """Crea y persiste un artifact de tipo GENERATED_FILE."""
        artifact = Artifact.create_generated_file(
            display_name=display_name,
            file_path=file_path,
            session_id=session.session_id,
            source_task_id=task_id,
            source_turn_id=turn_id,
            file_size_bytes=file_size_bytes,
            file_mime_type=file_mime_type,
        )
        return await self._persist_and_register(artifact, session)

    async def create_note(
        self,
        *,
        display_name: str,
        content: str,
        session: Session,
        task_id: TaskId,
        turn_id: TurnId,
        tags: tuple[str, ...] = (),
    ) -> Artifact:
        """Crea y persiste un artifact de tipo NOTE."""
        artifact = Artifact.create_note(
            display_name=display_name,
            content=content,
            session_id=session.session_id,
            source_task_id=task_id,
            source_turn_id=turn_id,
            tags=tags,
        )
        return await self._persist_and_register(artifact, session)

    async def create_from_tool_output(
        self,
        *,
        artifact_type: ArtifactType,
        display_name: str,
        raw_output: str,
        session: Session,
        task_id: TaskId,
        turn_id: TurnId,
        tool_id: str,
        metadata_extra: dict[str, Any] | None = None,
    ) -> Artifact:
        """
        Crea un artifact genérico desde el output crudo de una tool.

        Método de último recurso cuando no hay una factory específica para
        el tipo de artifact. Determina el content_type según artifact_type.

        Args:
            artifact_type:  Tipo de artifact a crear.
            display_name:   Nombre del artifact.
            raw_output:     Contenido crudo de la tool.
            session:        Session activa.
            task_id:        Task que lo produjo.
            turn_id:        Turn que lo solicitó.
            tool_id:        ID de la tool que lo generó.
            metadata_extra: Metadata adicional.

        Returns:
            Artifact creado y persistido.
        """
        content = ArtifactContent.from_markdown(raw_output)
        metadata = ArtifactMetadata(
            source_tool_id=tool_id,
            extra=metadata_extra or {},
        )

        artifact = Artifact(
            artifact_id=ArtifactId.generate(),
            artifact_type=artifact_type,
            display_name=display_name,
            content=content,
            metadata=metadata,
            session_id=session.session_id,
            source_task_id=task_id,
            source_turn_id=turn_id,
            created_at=datetime.now(tz=timezone.utc),
        )

        return await self._persist_and_register(artifact, session)

    # =========================================================================
    # RECUPERACIÓN Y BÚSQUEDA
    # =========================================================================

    async def get(self, artifact_id: ArtifactId) -> Artifact | None:
        """
        Carga un artifact completo por su ID.

        Args:
            artifact_id: ID del artifact.

        Returns:
            Artifact completo con contenido, o None si no existe.
        """
        try:
            return await self._store.load(artifact_id)
        except ForgeStorageError as e:
            logger.error("artifact_carga_fallo", artifact_id=artifact_id.to_str(), error=e)
            return None

    async def get_summary(
        self,
        artifact_id: ArtifactId,
    ) -> ArtifactSummary | None:
        """
        Carga solo el summary de un artifact (sin contenido completo).

        Args:
            artifact_id: ID del artifact.

        Returns:
            ArtifactSummary sin contenido, o None si no existe.
        """
        try:
            return await self._store.load_summary(artifact_id)
        except ForgeStorageError as e:
            logger.error("artifact_summary_fallo", artifact_id=artifact_id.to_str(), error=e)
            return None

    async def list_by_session(
        self,
        session_id: SessionId,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> PaginatedResult[ArtifactSummary]:
        """
        Lista los artifacts de una sesión.

        Args:
            session_id: ID de la sesión.
            limit:      Máximo de artifacts a retornar.
            offset:     Desplazamiento para paginación.

        Returns:
            PaginatedResult con ArtifactSummary ordenados por created_at desc.
        """
        try:
            return await self._store.list_by_session(
                session_id,
                limit=limit,
                offset=offset,
            )
        except ForgeStorageError as e:
            logger.error("artifact_listado_fallo", session_id=session_id.to_str(), error=e)
            from src.ports.outbound.storage_port import PaginatedResult
            return PaginatedResult(items=[], total_count=0, limit=limit, offset=offset)

    async def search(
        self,
        *,
        session_id: SessionId | None = None,
        artifact_type: ArtifactType | None = None,
        tags: list[str] | None = None,
        text_search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> PaginatedResult[ArtifactSummary]:
        """
        Busca artifacts según criterios múltiples.

        Args:
            session_id:    Filtrar por sesión.
            artifact_type: Filtrar por tipo.
            tags:          Filtrar por tags (AND).
            text_search:   Búsqueda de texto libre.
            limit:         Máximo de resultados.
            offset:        Desplazamiento para paginación.

        Returns:
            PaginatedResult con ArtifactSummary encontrados.
        """
        query = ArtifactQuery(
            session_id=session_id,
            artifact_type=artifact_type,
            tags=tags or [],
            text_search=text_search,
            limit=limit,
            offset=offset,
        )
        try:
            return await self._store.search(query)
        except ForgeStorageError as e:
            logger.error("artifact_busqueda_fallo", error=e)
            return PaginatedResult(items=[], total_count=0, limit=limit, offset=offset)

    # =========================================================================
    # EXPORTACIÓN
    # =========================================================================

    async def export(
        self,
        artifact_id: ArtifactId,
        session_id: SessionId,
        *,
        format: str = "markdown",
    ) -> bytes:
        """
        Exporta un artifact en el formato especificado.

        Args:
            artifact_id: ID del artifact a exportar.
            session_id:  ID de la sesión propietaria (para autorización).
            format:      'markdown' | 'json' | 'txt' | 'anki'.

        Returns:
            Bytes del archivo exportado.
        """
        artifact = await self.get(artifact_id)
        if artifact is None or artifact.session_id != session_id:
            return b""

        content = artifact.content.raw_content

        if format == "json":
            import json
            data = {
                "name": artifact.display_name,
                "type": artifact.artifact_type.value,
                "created_at": artifact.created_at.isoformat(),
                "content": content,
                "metadata": artifact.metadata.to_index_dict(),
            }
            return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")

        elif format == "anki" and artifact.artifact_type == ArtifactType.FLASHCARDS:
            return self._export_flashcards_anki(content)

        elif format == "txt":
            return content.encode("utf-8")

        else:  # markdown por defecto
            header = f"# {artifact.display_name}\n\n"
            meta = (
                f"*Tipo: {artifact.artifact_type.value} | "
                f"Creado: {artifact.created_at.strftime('%Y-%m-%d %H:%M')}*\n\n---\n\n"
            )
            return (header + meta + content).encode("utf-8")

    def _export_flashcards_anki(self, content: str) -> bytes:
        """
        Exporta flashcards en formato compatible con Anki (TSV).

        El formato Anki espera 'frente[TAB]reverso' por línea.

        Args:
            content: JSON serializado de las flashcards.

        Returns:
            Bytes en formato TSV para importar en Anki.
        """
        import json

        try:
            cards = json.loads(content)
            lines = []
            for card in cards:
                front = card.get("front", "").replace("\t", " ").replace("\n", "<br>")
                back = card.get("back", "").replace("\t", " ").replace("\n", "<br>")
                lines.append(f"{front}\t{back}")
            return "\n".join(lines).encode("utf-8")
        except (json.JSONDecodeError, AttributeError):
            return content.encode("utf-8")

    # =========================================================================
    # ELIMINACIÓN
    # =========================================================================

    async def delete(
        self,
        artifact_id: ArtifactId,
        session_id: SessionId,
    ) -> bool:
        """
        Elimina un artifact verificando que pertenezca a la sesión.

        Args:
            artifact_id: ID del artifact a eliminar.
            session_id:  ID de la sesión propietaria (autorización).

        Returns:
            True si fue eliminado, False si no existía o no pertenece a la sesión.
        """
        artifact = await self.get(artifact_id)
        if artifact is None or artifact.session_id != session_id:
            return False

        try:
            return await self._store.delete(artifact_id)
        except ForgeStorageError as e:
            logger.error("artifact_eliminacion_fallo", artifact_id=artifact_id.to_str(), error=e)
            return False

    # =========================================================================
    # HELPERS INTERNOS
    # =========================================================================

    async def _persist_and_register(
        self,
        artifact: Artifact,
        session: Session,
    ) -> Artifact:
        """
        Persiste el artifact y registra la referencia en la sesión.

        Args:
            artifact: Artifact a persistir.
            session:  Session activa donde se registra.

        Returns:
            El mismo artifact (inmutable, no se modifica).

        Raises:
            ForgeStorageError: Si no se puede persistir.
        """
        try:
            await self._store.save(artifact)
        except ForgeStorageError as e:
            logger.error(
                "artifact_persistencia_fallo",
                artifact_id=artifact.artifact_id.to_str(),
                artifact_type=artifact.artifact_type.value,
                error=e,
            )
            raise

        # Registrar referencia ligera en la sesión
        artifact_ref = ArtifactReference(
            artifact_id=artifact.artifact_id,
            artifact_type=artifact.artifact_type.value,
            display_name=artifact.display_name,
            created_at=artifact.created_at,
        )
        session.add_artifact(artifact_ref)

        logger.info(
            "artifact_creado",
            artifact_id=artifact.artifact_id.to_str(),
            artifact_type=artifact.artifact_type.value,
            session_id=session.session_id.to_str(),
            display_name=artifact.display_name,
            content_size=artifact.content.size_chars(),
        )

        return artifact