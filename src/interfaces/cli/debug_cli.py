#!/usr/bin/env python3
"""
CLI interactiva de Orion — agente conversacional.

Punto de entrada de desarrollo para probar el agente directamente desde
la terminal sin necesidad de la interfaz Tauri.

Diseño de boundaries:
  Esta CLI solo importa y usa:
    - build_container() del bootstrap (necesario para construir el sistema)
    - ConversationPort (el único contrato de entrada permitido para la UI)
    - Los tipos de request/response del conversation_port

  NO importa directamente:
    - ConversationCoordinator ni ningún servicio interno
    - SessionManager ni ningún servicio de aplicación
    - Entidades del dominio (Session, Task, etc.)

Arranque:
    python orion_cli.py

O con el entrypoint instalado:
    forge-user
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


def _ensure_project_root_in_path() -> None:
    """
    Añade el directorio raíz del proyecto al sys.path si no está ya.

    Necesario únicamente cuando se ejecuta el script directamente con
    `python orion_cli.py` desde el directorio raíz (no desde un entorno
    instalado con `pip install -e .`).

    Una vez que el proyecto esté instalado correctamente con:
        pip install -e .
    este bloque es innecesario y los entrypoints de pyproject.toml
    manejan el path correctamente.
    """
    project_root = Path(__file__).parent.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_project_root_in_path()


# Los únicos imports permitidos en una interfaz: bootstrap + puerto de entrada
from src.bootstrap.container import build_container, AppContainer  # noqa: E402
from src.ports.inbound.conversation_port import (  # noqa: E402
    ConversationPort,
    UserMessageRequest,
    AssistantResponseType,
    CreateSessionRequest,
)
from src.domain.value_objects.identifiers import UserId, SessionId  # noqa: E402
from forge_core.errors.types import (  # noqa: E402
    ForgeStorageError,
    ForgeLLMError,
    ForgePolicyError,
)


class OrionCLI:
    """
    CLI interactiva de Orion.

    Solo habla con el ConversationPort — no toca ningún servicio interno.
    """

    def __init__(self) -> None:
        self._container: AppContainer | None = None
        self._port: ConversationPort | None = None
        self._session_id: str | None = None
        self._user_id: UserId = UserId.generate()

    # =========================================================================
    # ARRANQUE
    # =========================================================================

    async def initialize(self) -> None:
        """
        Construye el container y restaura o crea la sesión activa.

        Intenta restaurar la última sesión del usuario. Si no existe,
        crea una nueva. El workaround de "siempre crear nueva sesión"
        fue eliminado — el bug de restauración está corregido en
        sqlite_storage.py::load_session().
        """
        print("\n🚀 Inicializando Orion...")

        self._container = await build_container()
        self._port = self._container.conversation_port

        session_id = await self._restore_or_create_session()
        self._session_id = session_id

        print("✅ Orion listo!")
        print()
        print("=" * 60)
        print("  💬  CLI INTERACTIVA DE ORION")
        print("=" * 60)
        print("  Escribe 'salir' para terminar | 'help' para comandos")
        print(f"  Sesión: {self._session_id[:36]}...")
        print("=" * 60)
        print()

    async def _restore_or_create_session(self) -> str:
        """
        Restaura la última sesión activa o crea una nueva.

        Returns:
            session_id de la sesión activa como string.
        """
        assert self._container is not None

        profile = await self._container.profile_store.load_default()
        if profile is not None:
            self._user_id = profile.user_id
            last_session = await self._container.session_manager.get_last_active_session(
                self._user_id
            )
            if last_session is not None:
                print(f"  📂 Sesión restaurada ({last_session.turn_count} turnos previos)")
                return last_session.session_id.to_str()

        # Crear perfil de usuario por defecto si no existe
        if profile is None:
            from src.domain.entities.user_profile import UserProfile
            profile = UserProfile.create_default()
            await self._container.profile_store.save(profile)
            self._user_id = profile.user_id

        # Crear nueva sesión
        request = CreateSessionRequest(user_id=self._user_id)
        session_id, _ = await self._container.conversation_port.create_session(request)
        print("  ✨ Nueva sesión creada")
        return session_id.to_str()

    # =========================================================================
    # BUCLE PRINCIPAL
    # =========================================================================

    async def run(self) -> None:
        """Bucle interactivo principal."""
        await self.initialize()

        while True:
            try:
                user_input = input("Tú: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\n👋 Hasta luego!")
                break

            if not user_input:
                continue

            if user_input.lower() in {"salir", "exit", "quit"}:
                print("\n👋 Hasta luego!")
                break
            if user_input.lower() == "help":
                self._show_help()
                continue
            if user_input.lower() == "info":
                await self._show_info()
                continue
            if user_input.lower() == "nueva":
                await self._new_session()
                continue

            print("\n⏳ Pensando...\n")
            response_text = await self._send_message(user_input)
            if response_text:
                print(f"Orion: {response_text}\n")

    # =========================================================================
    # MANEJO DE MENSAJES — errores tipados, sin detección por substring
    # =========================================================================

    async def _send_message(self, user_input: str) -> str:
        """
        Envía un mensaje al agente y retorna la respuesta de texto.

        Captura errores usando la jerarquía de excepciones tipadas del
        dominio. No detecta errores por contenido del string del mensaje
        de excepción.
        """
        assert self._port is not None
        assert self._session_id is not None

        request = UserMessageRequest(
            session_id=SessionId.from_str(self._session_id),
            message=user_input,
            attachments=[],
        )

        try:
            response = await self._port.send_message(request)

            if response.response_type == AssistantResponseType.ERROR:
                return f"⚠️ {response.error_message or 'Error procesando el mensaje.'}"

            return response.text_content or "(Sin respuesta)"

        except ForgeStorageError as exc:
            print(f"  [Storage] {exc}", file=sys.stderr)
            return "Error temporal del almacenamiento. Por favor intenta de nuevo."

        except ForgeLLMError as exc:
            print(f"  [LLM] {exc}", file=sys.stderr)
            return "Error al comunicarse con el modelo. Por favor intenta de nuevo."

        except ForgePolicyError as exc:
            return f"🔒 Acción no permitida: {exc}"

        except Exception as exc:
            import traceback
            traceback.print_exc()
            return f"❌ Error inesperado ({type(exc).__name__}). Revisa la consola."

    # =========================================================================
    # COMANDOS AUXILIARES
    # =========================================================================

    def _show_help(self) -> None:
        print()
        print("  📖 COMANDOS:")
        print("     help   — mostrar esta ayuda")
        print("     info   — información de la sesión actual")
        print("     nueva  — iniciar una sesión nueva")
        print("     salir  — terminar")
        print()
        print("  💡 EJEMPLOS:")
        print("     ¿Qué es el teorema de Bayes?")
        print("     Lista los archivos en ~/Documentos")
        print("     Genera flashcards sobre fotosíntesis")
        print()

    async def _show_info(self) -> None:
        """Muestra estadísticas de la sesión activa."""
        assert self._container is not None
        assert self._session_id is not None

        stats = await self._container.session_manager.get_session_stats(
            SessionId.from_str(self._session_id)
        )
        print()
        print("  📊 SESIÓN ACTIVA:")
        print(f"     ID:          {self._session_id[:36]}")
        print(f"     Estado:      {stats.get('status', '?')}")
        print(f"     Turnos:      {stats.get('turn_count', 0)}")
        print(f"     Tareas:      {stats.get('task_count', 0)}")
        print(f"     Artifacts:   {stats.get('artifact_count', 0)}")
        budget = stats.get("token_budget", {})
        if budget:
            ratio = budget.get("usage_ratio", 0)
            print(f"     Contexto:    {ratio:.0%} usado")
        print()

    async def _new_session(self) -> None:
        """Crea una nueva sesión."""
        assert self._container is not None

        request = CreateSessionRequest(user_id=self._user_id)
        session_id, _ = await self._container.conversation_port.create_session(request)
        self._session_id = session_id.to_str()
        print(f"  ✨ Nueva sesión: {self._session_id[:36]}")
        print()


# =============================================================================
# ENTRY POINT
# =============================================================================


async def _main() -> int:
    """Punto de entrada async."""
    cli = OrionCLI()
    try:
        await cli.run()
        return 0
    except Exception as exc:
        import traceback
        print(f"\n❌ Error fatal: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))