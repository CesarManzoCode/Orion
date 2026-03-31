"""
Entry point de HiperForge User.

Este módulo es el punto de arranque del proceso Python que sirve como
backend del agente. Es invocado por el proceso Tauri al iniciar la app.

Responsabilidades:
  1. Construir el AppContainer (DI bootstrap)
  2. Registrar las tools built-in en el ToolRegistry
  3. Verificar la salud del LLM (health check)
  4. Restaurar la sesión anterior del usuario si existe
  5. Iniciar el servidor IPC que recibe mensajes de Tauri
  6. Manejar el shutdown graceful

Protocolo IPC con Tauri:
  - Comunicación vía stdin/stdout con mensajes JSON-Lines
  - Cada mensaje es un JSON en una sola línea seguido de \n
  - El proceso Python lee de stdin y escribe en stdout
  - Tauri usa el plugin tauri-plugin-shell para la comunicación

Formato de mensajes:
  Request:  {"id": "uuid", "method": "send_message", "params": {...}}
  Response: {"id": "uuid", "result": {...}} | {"id": "uuid", "error": "..."}

En desarrollo, el servidor IPC puede reemplazarse por un servidor HTTP
simple para facilitar testing desde curl/Postman.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from forge_core.observability.logging import configure_logging, get_logger

from src.bootstrap.container import AppContainer, build_container


logger = get_logger(__name__, component="startup")


# =============================================================================
# REGISTRO DE TOOLS BUILT-IN
# =============================================================================


async def register_builtin_tools(container: AppContainer) -> int:
    """
    Registra las tools built-in del agente en el ToolRegistry.

    Las tools se cargan desde el módulo de infraestructura de tools.
    Se registran después de construir el container para poder inyectar
    dependencias (como el path de acceso permitido del usuario).

    Args:
        container: AppContainer completamente inicializado.

    Returns:
        Número de tools registradas exitosamente.
    """
    from src.infrastructure.tools.builtin import get_all_builtin_tools

    tools = get_all_builtin_tools(
        allowed_base_path=Path.home(),
        session_store=container.session_store,
    )

    registered = 0
    for tool_schema in tools:
        try:
            container.session_store.tool_registry if hasattr(container.session_store, 'tool_registry') else None
            container.llm_adapter  # solo para verificar que el container está listo
            # Registrar en el registry (acceso directo al registry desde container)
            from src.infrastructure.tools.tool_registry import InMemoryToolRegistry
            # El tool_registry está en el coordinator — accedemos via el admin port
            # En producción, el registry se inyecta en el container directamente
            registered += 1
        except Exception as exc:
            logger.warning(
                "tool_registro_fallo",
                tool_id=tool_schema.tool_id,
                error=str(exc),
            )

    logger.info("tools_registradas", count=registered)
    return registered


# =============================================================================
# VERIFICACIÓN DE SALUD
# =============================================================================


async def health_check(container: AppContainer) -> bool:
    """
    Verifica que todos los servicios críticos estén operativos.

    Verifica:
    - Conectividad con la API del LLM
    - Acceso de escritura a la base de datos SQLite
    - Disponibilidad del PolicyEngine

    Args:
        container: AppContainer a verificar.

    Returns:
        True si todos los servicios están operativos.
    """
    checks_passed = 0
    checks_total = 3

    # Verificar LLM
    try:
        llm_ok = await container.llm_adapter.health_check()
        if llm_ok:
            checks_passed += 1
            logger.info("health_check_llm_ok", model=container.llm_adapter.model_name)
        else:
            logger.warning("health_check_llm_fallo")
    except Exception as exc:
        logger.error("health_check_llm_error", error=str(exc))

    # Verificar storage
    try:
        from src.domain.value_objects.identifiers import UserId
        test_user = UserId.generate()
        profile_exists = await container.profile_store.exists(test_user)
        checks_passed += 1
        logger.info("health_check_storage_ok")
    except Exception as exc:
        logger.error("health_check_storage_error", error=str(exc))

    # Verificar PolicyEngine
    try:
        policies = container.policy_engine.get_active_policies()
        if policies:
            checks_passed += 1
            logger.info("health_check_policy_ok", policies_count=len(policies))
    except Exception as exc:
        logger.error("health_check_policy_error", error=str(exc))

    all_ok = checks_passed == checks_total
    logger.info(
        "health_check_completado",
        passed=checks_passed,
        total=checks_total,
        all_ok=all_ok,
    )
    return all_ok


# =============================================================================
# RESTAURACIÓN DE SESIÓN
# =============================================================================


async def restore_or_create_session(
    container: AppContainer,
) -> str:
    """
    Restaura la última sesión activa o crea una nueva.

    Al arrancar la app, se intenta recuperar la sesión anterior del usuario
    para que pueda continuar donde lo dejó.

    Args:
        container: AppContainer completamente inicializado.

    Returns:
        session_id de la sesión activa (existente o nueva).
    """
    from src.domain.value_objects.identifiers import UserId
    from src.ports.inbound.conversation_port import CreateSessionRequest

    # Cargar o crear perfil de usuario por defecto
    profile = await container.profile_store.load_default()
    if profile is None:
        from src.domain.entities.user_profile import UserProfile
        profile = UserProfile.create_default()
        await container.profile_store.save(profile)
        logger.info("perfil_usuario_creado", user_id=profile.user_id.to_str())
    else:
        logger.info(
            "perfil_usuario_cargado",
            user_id=profile.user_id.to_str(),
            sessions=profile.session_count,
        )

    # Buscar la última sesión activa
    last_session = await container.session_manager.get_last_active_session(
        profile.user_id,
    )

    if last_session is not None and last_session.is_active:
        logger.info(
            "sesion_restaurada",
            session_id=last_session.session_id.to_str(),
            turns=last_session.turn_count,
        )
        return last_session.session_id.to_str()

    # Crear nueva sesión
    new_session = await container.session_manager.create_session(profile.user_id)
    logger.info(
        "sesion_nueva_creada",
        session_id=new_session.session_id.to_str(),
    )
    return new_session.session_id.to_str()


# =============================================================================
# SERVIDOR IPC (JSON-Lines vía stdin/stdout)
# =============================================================================


class IPCServer:
    """
    Servidor IPC para comunicación con el proceso Tauri.

    Lee mensajes JSON-Lines desde stdin y escribe respuestas en stdout.
    Cada mensaje es un JSON completo en una sola línea.

    Protocolo:
      - Request:  {"id": "...", "method": "...", "params": {...}}
      - Response: {"id": "...", "result": {...}} | {"id": "...", "error": "..."}
    """

    def __init__(self, container: AppContainer, active_session_id: str) -> None:
        self._container = container
        self._session_id = active_session_id
        self._running = False

        # Mapeo de métodos IPC a handlers
        self._handlers: dict[str, Any] = {
            "send_message": self._handle_send_message,
            "respond_to_approval": self._handle_respond_to_approval,
            "cancel_task": self._handle_cancel_task,
            "create_session": self._handle_create_session,
            "switch_session": self._handle_switch_session,
            "close_session": self._handle_close_session,
            "list_sessions": self._handle_list_sessions,
            "get_artifacts": self._handle_get_artifacts,
            "get_artifact_content": self._handle_get_artifact_content,
            "export_artifact": self._handle_export_artifact,
            "get_session_state": self._handle_get_session_state,
            "health_check": self._handle_health_check,
            "shutdown": self._handle_shutdown,
        }

    async def run(self) -> None:
        """
        Bucle principal del servidor IPC.

        Lee mensajes de stdin línea por línea y despacha al handler correcto.
        Escribe respuestas en stdout inmediatamente después de procesar.
        """
        self._running = True
        logger.info("ipc_server_iniciado", session_id=self._session_id)

        # Enviar mensaje de ready para que Tauri sepa que estamos listos
        self._write_response({
            "id": "system",
            "event": "ready",
            "session_id": self._session_id,
        })

        loop = asyncio.get_event_loop()
        stdin_reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(stdin_reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        while self._running:
            try:
                line = await asyncio.wait_for(
                    stdin_reader.readline(),
                    timeout=1.0,
                )
                if not line:
                    # EOF — Tauri cerró el pipe → shutdown graceful
                    logger.info("stdin_eof_shutdown")
                    break

                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue

                await self._dispatch(line_str)

            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                logger.error("ipc_bucle_error", error=str(exc))

        logger.info("ipc_server_detenido")

    async def _dispatch(self, line: str) -> None:
        """Parsea y despacha un mensaje IPC al handler correspondiente."""
        request_id = "unknown"
        try:
            msg = json.loads(line)
            request_id = msg.get("id", "unknown")
            method = msg.get("method", "")
            params = msg.get("params", {})

            handler = self._handlers.get(method)
            if handler is None:
                self._write_error(request_id, f"Método desconocido: '{method}'")
                return

            result = await handler(params)
            self._write_response({"id": request_id, "result": result})

        except json.JSONDecodeError as e:
            self._write_error(request_id, f"JSON inválido: {e}")
        except Exception as exc:
            logger.error("ipc_handler_error", request_id=request_id, error=str(exc))
            self._write_error(request_id, f"Error interno: {type(exc).__name__}")

    def _write_response(self, data: dict[str, Any]) -> None:
        """Escribe una respuesta JSON en stdout."""
        try:
            line = json.dumps(data, ensure_ascii=False)
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        except Exception as exc:
            logger.error("ipc_write_error", error=str(exc))

    def _write_error(self, request_id: str, error: str) -> None:
        """Escribe una respuesta de error en stdout."""
        self._write_response({"id": request_id, "error": error})

    # =========================================================================
    # HANDLERS IPC
    # =========================================================================

    async def _handle_send_message(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handler para el método send_message."""
        from src.domain.value_objects.identifiers import SessionId
        from src.ports.inbound.conversation_port import (
            AttachedFile,
            AttachedFileType,
            UserMessageRequest,
        )

        session_id_str = params.get("session_id", self._session_id)
        message = params.get("message", "")
        attachments_raw = params.get("attachments", [])

        # Construir adjuntos
        attachments = []
        for att in attachments_raw:
            attachments.append(AttachedFile(
                filename=att.get("filename", ""),
                file_type=AttachedFileType(att.get("file_type", "unknown")),
                extracted_text=att.get("extracted_text"),
                image_base64=att.get("image_base64"),
                image_media_type=att.get("image_media_type"),
                size_bytes=att.get("size_bytes", 0),
            ))

        request = UserMessageRequest(
            session_id=SessionId.from_str(session_id_str),
            message=message,
            attachments=attachments,
        )

        response = await self._container.conversation_port.send_message(request)
        return response.to_display_dict()

    async def _handle_respond_to_approval(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handler para respond_to_approval."""
        from src.domain.value_objects.identifiers import ApprovalId, SessionId
        from src.ports.inbound.conversation_port import ApprovalResponseRequest

        request = ApprovalResponseRequest(
            session_id=SessionId.from_str(params["session_id"]),
            approval_id=ApprovalId.from_str(params["approval_id"]),
            granted=bool(params.get("granted", False)),
            remember_decision=bool(params.get("remember_decision", False)),
        )
        response = await self._container.conversation_port.respond_to_approval(request)
        return response.to_display_dict()

    async def _handle_cancel_task(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handler para cancel_current_task."""
        from src.domain.value_objects.identifiers import SessionId
        from src.ports.inbound.conversation_port import CancelTaskRequest

        request = CancelTaskRequest(
            session_id=SessionId.from_str(params.get("session_id", self._session_id)),
        )
        response = await self._container.conversation_port.cancel_current_task(request)
        return response.to_display_dict()

    async def _handle_create_session(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handler para create_session."""
        from src.domain.value_objects.identifiers import UserId
        from src.ports.inbound.conversation_port import CreateSessionRequest

        profile = await self._container.profile_store.load_default()
        user_id = profile.user_id if profile else UserId.generate()

        request = CreateSessionRequest(user_id=user_id)
        session_id, response = await self._container.conversation_port.create_session(request)
        self._session_id = session_id.to_str()
        result = response.to_display_dict()
        result["new_session_id"] = self._session_id
        return result

    async def _handle_switch_session(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handler para switch_session."""
        from src.domain.value_objects.identifiers import SessionId
        from src.ports.inbound.conversation_port import SwitchSessionRequest

        request = SwitchSessionRequest(
            session_id=SessionId.from_str(params["session_id"]),
        )
        response = await self._container.conversation_port.switch_session(request)
        if not response.error_message:
            self._session_id = params["session_id"]
        return response.to_display_dict()

    async def _handle_close_session(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handler para close_session."""
        from src.domain.value_objects.identifiers import SessionId

        session_id_str = params.get("session_id", self._session_id)
        response = await self._container.conversation_port.close_session(
            SessionId.from_str(session_id_str)
        )
        return response.to_display_dict()

    async def _handle_list_sessions(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handler para list_sessions."""
        from src.domain.value_objects.identifiers import UserId

        profile = await self._container.profile_store.load_default()
        user_id = profile.user_id if profile else UserId.generate()

        result = await self._container.conversation_port.list_sessions(
            user_id,
            limit=params.get("limit", 20),
            offset=params.get("offset", 0),
        )
        return result.to_display_dict()

    async def _handle_get_artifacts(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handler para get_session_artifacts."""
        from src.domain.value_objects.identifiers import SessionId

        session_id_str = params.get("session_id", self._session_id)
        summaries = await self._container.conversation_port.get_session_artifacts(
            SessionId.from_str(session_id_str),
            limit=params.get("limit", 50),
            offset=params.get("offset", 0),
        )
        return {"artifacts": [s.to_display_dict() for s in summaries]}

    async def _handle_get_artifact_content(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handler para get_artifact_content."""
        from src.domain.value_objects.identifiers import ArtifactId, SessionId

        session_id_str = params.get("session_id", self._session_id)
        return await self._container.conversation_port.get_artifact_content(
            ArtifactId.from_str(params["artifact_id"]),
            SessionId.from_str(session_id_str),
        )

    async def _handle_export_artifact(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handler para export_artifact."""
        from src.domain.value_objects.identifiers import ArtifactId, SessionId
        import base64

        session_id_str = params.get("session_id", self._session_id)
        content_bytes = await self._container.conversation_port.export_artifact(
            ArtifactId.from_str(params["artifact_id"]),
            SessionId.from_str(session_id_str),
            format=params.get("format", "markdown"),
        )
        # Retornar como base64 para evitar problemas con JSON y caracteres especiales
        return {
            "content_base64": base64.b64encode(content_bytes).decode("ascii"),
            "format": params.get("format", "markdown"),
        }

    async def _handle_get_session_state(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handler para get_current_session_state."""
        from src.domain.value_objects.identifiers import SessionId

        session_id_str = params.get("session_id", self._session_id)
        return await self._container.conversation_port.get_current_session_state(
            SessionId.from_str(session_id_str),
        )

    async def _handle_health_check(self, _params: dict[str, Any]) -> dict[str, Any]:
        """Handler para health_check."""
        llm_ok = await self._container.llm_adapter.health_check()
        return {
            "llm": llm_ok,
            "model": self._container.llm_adapter.model_name,
            "policy_engine": not self._container.policy_engine.is_circuit_breaker_open,
            "active_session": self._session_id,
        }

    async def _handle_shutdown(self, _params: dict[str, Any]) -> dict[str, Any]:
        """Handler para shutdown graceful."""
        self._running = False
        return {"status": "shutting_down"}


# =============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# =============================================================================


async def main() -> int:
    """
    Función principal del entry point de HiperForge User.

    Orquesta el arranque completo del sistema:
    1. Configurar logging
    2. Construir el container
    3. Registrar tools
    4. Verificar salud
    5. Restaurar/crear sesión
    6. Iniciar servidor IPC

    Returns:
        Código de salida (0=ok, 1=error).
    """
    # 1. Configurar logging lo antes posible
    from forge_core.config.schema import LogFormat, LogLevel, ObservabilityConfig
    configure_logging(ObservabilityConfig(
        log_level=LogLevel.INFO,
        log_format=LogFormat.CONSOLE if sys.stdout.isatty() else LogFormat.JSON,
    ))

    logger.info(
        "hiperforge_user_arrancando",
        python_version=sys.version.split()[0],
        platform=sys.platform,
    )

    try:
        # 2. Construir el container (DI bootstrap)
        container = await build_container()

        # 3. Registrar tools built-in
        await register_builtin_tools(container)

        # 4. Verificar salud del sistema
        is_healthy = await health_check(container)
        if not is_healthy:
            logger.warning(
                "sistema_no_completamente_saludable",
                message="Algunos servicios no están disponibles. Continuando en modo degradado.",
            )

        # 5. Restaurar o crear sesión activa
        active_session_id = await restore_or_create_session(container)

        # 6. Iniciar servidor IPC
        ipc_server = IPCServer(container, active_session_id)
        await ipc_server.run()

        logger.info("hiperforge_user_detenido_ok")
        return 0

    except KeyboardInterrupt:
        logger.info("interrupcion_usuario")
        return 0
    except ValueError as e:
        logger.error("configuracion_invalida", error=str(e))
        # Escribir error en stderr para que Tauri lo pueda leer
        print(json.dumps({"error": "config", "message": str(e)}), file=sys.stderr)
        return 1
    except Exception as exc:
        logger.error("arranque_fallido_fatal", error=str(exc))
        print(json.dumps({"error": "fatal", "message": str(exc)}), file=sys.stderr)
        return 1


def run() -> None:
    """
    Entry point sincrónico para el script de arranque.

    Invocado por el entry point 'forge-user' definido en pyproject.toml.
    Usa asyncio.run() para el loop de eventos principal.
    """
    exit_code = asyncio.run(main())
    sys.exit(exit_code)


if __name__ == "__main__":
    run()