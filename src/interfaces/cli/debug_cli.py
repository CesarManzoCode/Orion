"""
Debug CLI de HiperForge User.

Interfaz de línea de comandos para administración, diagnóstico y debugging
del sistema. Accede al AdminPort del sistema para obtener información técnica
que no está disponible en la UI principal de Tauri.

Uso:
    forge-user-debug sessions list
    forge-user-debug sessions inspect <session_id>
    forge-user-debug audit query --session <id> --hours 24
    forge-user-debug tools list
    forge-user-debug policy status
    forge-user-debug security report --session <id>

Solo disponible en entornos de desarrollo. No se instala en producción.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from src.bootstrap.container import build_container


app = typer.Typer(
    name="forge-user-debug",
    help="Debug CLI de administración para HiperForge User.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

sessions_app = typer.Typer(help="Gestión de sesiones.", no_args_is_help=True)
audit_app = typer.Typer(help="Consulta del audit log.", no_args_is_help=True)
tools_app = typer.Typer(help="Gestión de tools.", no_args_is_help=True)
policy_app = typer.Typer(help="Estado del PolicyEngine.", no_args_is_help=True)
security_app = typer.Typer(help="Reportes de seguridad.", no_args_is_help=True)

app.add_typer(sessions_app, name="sessions")
app.add_typer(audit_app, name="audit")
app.add_typer(tools_app, name="tools")
app.add_typer(policy_app, name="policy")
app.add_typer(security_app, name="security")

console = Console()


def _run(coro: Any) -> Any:
    """Ejecuta una coroutine de forma síncrona para el CLI."""
    return asyncio.run(coro)


async def _get_admin_port() -> Any:
    """Construye el container y retorna el AdminPort."""
    container = await build_container()
    return container.admin_port


# =============================================================================
# SESSIONS
# =============================================================================


@sessions_app.command("list")
def sessions_list(
    limit: int = typer.Option(20, "--limit", "-n", help="Máximo de sesiones a mostrar."),
    include_closed: bool = typer.Option(True, "--all/--active", help="Incluir sesiones cerradas."),
) -> None:
    """Lista las sesiones del sistema."""

    async def _run_async() -> None:
        admin = await _get_admin_port()
        sessions = await admin.get_session_list(limit=limit, include_closed=include_closed)

        table = Table(title="Sesiones", show_header=True, header_style="bold cyan")
        table.add_column("ID", style="dim", width=28)
        table.add_column("Estado", width=12)
        table.add_column("Turns", justify="right", width=6)
        table.add_column("Artifacts", justify="right", width=10)
        table.add_column("Última actividad", width=22)

        for s in sessions:
            status_color = {
                "active": "green",
                "paused": "yellow",
                "closed": "dim",
                "archived": "dim",
            }.get(s.get("status", ""), "white")

            table.add_row(
                s.get("session_id", "")[:26] + "..",
                f"[{status_color}]{s.get('status', '')}[/{status_color}]",
                str(s.get("turn_count", 0)),
                str(s.get("artifact_count", 0)),
                s.get("last_activity", "")[:19],
            )

        console.print(table)

    _run(_run_async())


@sessions_app.command("inspect")
def sessions_inspect(
    session_id: str = typer.Argument(..., help="ID de la sesión a inspeccionar."),
) -> None:
    """Muestra información técnica detallada de una sesión."""

    async def _run_async() -> None:
        from src.domain.value_objects.identifiers import SessionId
        admin = await _get_admin_port()
        info = await admin.inspect_session(SessionId.from_str(session_id))

        console.print(Panel(
            json.dumps(info, indent=2, ensure_ascii=False, default=str),
            title=f"Sesión: {session_id[:20]}...",
            border_style="cyan",
        ))

    _run(_run_async())


# =============================================================================
# AUDIT
# =============================================================================


@audit_app.command("query")
def audit_query(
    session_id: Optional[str] = typer.Option(None, "--session", "-s", help="Filtrar por sesión."),
    tool_id: Optional[str] = typer.Option(None, "--tool", "-t", help="Filtrar por tool."),
    last_hours: int = typer.Option(24, "--hours", "-h", help="Últimas N horas."),
    limit: int = typer.Option(50, "--limit", "-n", help="Máximo de entradas."),
    security_only: bool = typer.Option(False, "--security", help="Solo eventos de seguridad."),
) -> None:
    """Consulta el audit log con filtros."""

    async def _run_async() -> None:
        admin = await _get_admin_port()
        entries = await admin.query_audit_log(
            session_id=session_id,
            tool_id=tool_id,
            last_hours=last_hours,
            limit=limit,
        )

        if security_only:
            entries = [e for e in entries if e.get("is_security_event")]

        if not entries:
            console.print("[dim]No se encontraron entradas en el audit log.[/dim]")
            return

        table = Table(title="Audit Log", show_header=True, header_style="bold magenta")
        table.add_column("Timestamp", width=20)
        table.add_column("Tool", width=20)
        table.add_column("Decisión", width=16)
        table.add_column("Riesgo", width=10)
        table.add_column("OK", width=4)
        table.add_column("ms", justify="right", width=8)

        for entry in entries:
            decision = entry.get("policy_decision", "")
            decision_color = {
                "allow": "green",
                "deny": "red",
                "require_approval": "yellow",
            }.get(decision, "white")

            is_sec = entry.get("is_security_event", False)
            ok = entry.get("success")
            ok_str = "[green]✓[/green]" if ok else ("[red]✗[/red]" if ok is False else "-")

            table.add_row(
                entry.get("occurred_at", "")[:19],
                f"{'[bold red]' if is_sec else ''}{entry.get('tool_id', '-')[:18]}{'[/bold red]' if is_sec else ''}",
                f"[{decision_color}]{decision}[/{decision_color}]",
                entry.get("risk_level", "-"),
                ok_str,
                str(round(entry.get("duration_ms") or 0, 1)),
            )

        console.print(table)
        console.print(f"[dim]{len(entries)} entradas mostradas[/dim]")

    _run(_run_async())


# =============================================================================
# TOOLS
# =============================================================================


@tools_app.command("list")
def tools_list(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filtrar por categoría."),
    include_disabled: bool = typer.Option(False, "--all", help="Incluir tools deshabilitadas."),
) -> None:
    """Lista las tools registradas en el sistema."""

    async def _run_async() -> None:
        admin = await _get_admin_port()
        tools = await admin.get_tool_list(
            category=category,
            include_disabled=include_disabled,
        )

        table = Table(title="Tools registradas", show_header=True, header_style="bold blue")
        table.add_column("ID", width=24)
        table.add_column("Categoría", width=14)
        table.add_column("Riesgo", width=10)
        table.add_column("Habilitada", width=10)
        table.add_column("Saludable", width=10)
        table.add_column("Invocaciones", justify="right", width=13)
        table.add_column("Fallos", justify="right", width=8)

        for t in tools:
            risk = t.get("risk_level", "NONE")
            risk_color = {
                "NONE": "green", "LOW": "cyan", "MEDIUM": "yellow",
                "HIGH": "red", "CRITICAL": "bold red",
            }.get(risk, "white")

            table.add_row(
                t.get("tool_id", ""),
                t.get("category", ""),
                f"[{risk_color}]{risk}[/{risk_color}]",
                "[green]✓[/green]" if t.get("enabled") else "[red]✗[/red]",
                "[green]✓[/green]" if t.get("healthy") else "[red]✗[/red]",
                str(t.get("invocations", 0)),
                str(t.get("failures", 0)),
            )

        console.print(table)

    _run(_run_async())


# =============================================================================
# POLICY
# =============================================================================


@policy_app.command("status")
def policy_status() -> None:
    """Muestra el estado del PolicyEngine y las políticas activas."""

    async def _run_async() -> None:
        admin = await _get_admin_port()
        status = await admin.get_policy_status()

        # Circuit breaker
        cb = status.get("circuit_breaker", {})
        cb_status = "[red]ABIERTO[/red]" if cb.get("is_open") else "[green]cerrado[/green]"

        console.print(Panel(
            f"Circuit breaker: {cb_status}  |  "
            f"Evaluaciones: [bold]{status.get('total_evaluations', 0)}[/bold]  |  "
            f"Denials: [bold red]{status.get('total_denials', 0)}[/bold red]  |  "
            f"Eventos de seguridad: [bold yellow]{status.get('total_security_events', 0)}[/bold yellow]",
            title="Estado del PolicyEngine",
            border_style="magenta",
        ))

        # Políticas activas
        policies = status.get("policies", [])
        if policies:
            table = Table(title="Políticas activas", header_style="bold")
            table.add_column("Nombre", width=42)
            table.add_column("Prioridad", justify="right", width=10)
            table.add_column("Habilitada", width=10)

            for p in sorted(policies, key=lambda x: x.get("priority", 0), reverse=True):
                table.add_row(
                    p.get("name", ""),
                    str(p.get("priority", 0)),
                    "[green]✓[/green]" if p.get("enabled") else "[red]✗[/red]",
                )
            console.print(table)

    _run(_run_async())


# =============================================================================
# SECURITY
# =============================================================================


@security_app.command("report")
def security_report(
    session_id: Optional[str] = typer.Option(None, "--session", "-s", help="Sesión específica."),
) -> None:
    """Muestra el reporte de seguridad del PolicyEngine."""

    async def _run_async() -> None:
        admin = await _get_admin_port()
        report = await admin.get_security_report(session_id)

        console.print(Panel(
            json.dumps(report, indent=2, ensure_ascii=False, default=str),
            title="Reporte de seguridad",
            border_style="red",
        ))

    _run(_run_async())


@security_app.command("reset")
def security_reset(
    session_id: str = typer.Argument(..., help="ID de la sesión a reiniciar."),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Confirmar sin prompt."),
) -> None:
    """Reinicia el contexto de seguridad de una sesión (sale de lockdown)."""
    if not confirm:
        typer.confirm(
            f"¿Reiniciar seguridad de la sesión {session_id[:20]}...?",
            abort=True,
        )

    async def _run_async() -> None:
        admin = await _get_admin_port()
        result = await admin.reset_session_security(session_id)
        console.print(f"[green]✓[/green] {result.get('message', 'Seguridad reiniciada.')}")

    _run(_run_async())


# =============================================================================
# MÉTRICAS GLOBALES
# =============================================================================


@app.command("metrics")
def metrics() -> None:
    """Muestra un resumen de métricas globales del sistema."""

    async def _run_async() -> None:
        admin = await _get_admin_port()
        m = await admin.get_metrics_summary()

        console.print(Panel(
            json.dumps(m, indent=2, ensure_ascii=False, default=str),
            title="Métricas del sistema",
            border_style="blue",
        ))

    _run(_run_async())


@app.command("config")
def show_config() -> None:
    """Muestra la configuración activa del sistema (sin secrets)."""

    async def _run_async() -> None:
        admin = await _get_admin_port()
        cfg = await admin.get_config()
        console.print(Panel(
            json.dumps(cfg, indent=2, ensure_ascii=False, default=str),
            title="Configuración activa",
            border_style="cyan",
        ))

    _run(_run_async())


# Importación necesaria para el tipo Any en _run
from typing import Any  # noqa: E402

if __name__ == "__main__":
    app()