"""
Value objects y reglas de clasificación de riesgo del dominio HiperForge User.

Este módulo define la lógica de negocio pura de evaluación de riesgo específica
al producto HiperForge User. Complementa el RiskLevel genérico de forge_core
con reglas de dominio concretas: qué paths son sensibles en Linux y Windows,
qué patrones de argumentos indican intención maliciosa, y cómo combinar el
riesgo base de una tool con el riesgo específico de un input.

Separación de responsabilidades respecto a forge_core:
  - forge_core/tools/protocol.py: define RiskLevel y RiskAssessment genéricos
  - forge_core/policy/framework.py: define el framework de evaluación de políticas
  - este módulo: implementa las REGLAS CONCRETAS de riesgo del producto
    (qué paths bloquear, qué patrones detectar, cómo calcular riesgo efectivo)

La lógica de este módulo es usada por:
  - src/application/policy_engine/policy_engine.py: para classify_risk()
  - src/domain/policies/: para definir condiciones de políticas concretas
  - src/infrastructure/security/: para validación de paths y sanitización

Principios de diseño:
  1. Todo el código es puro — sin I/O, sin estado global mutable.
  2. Las reglas son datos explícitos (listas, patrones), no magic strings.
  3. Las funciones de clasificación son deterministas y fast (< 1ms).
  4. Los falsos positivos (bloquear algo seguro) son preferibles a los
     falsos negativos (permitir algo peligroso) — fail-closed.
  5. Las reglas están documentadas con la razón de seguridad de cada una.

Jerarquía de tipos:

  SensitivePathPattern     — patrón de path con clasificación de riesgo
  InputRiskSignal          — señal de riesgo detectada en un input
  InputRiskAnalysis        — resultado completo del análisis de un input
  DomainRiskClassifier     — clasificador de riesgo del dominio (funciones puras)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from forge_core.tools.protocol import RiskAssessment, RiskLevel


# =============================================================================
# PATRONES DE PATHS SENSIBLES
# =============================================================================


class PathSensitivity(Enum):
    """
    Nivel de sensibilidad de un path del sistema de archivos.

    Determina qué nivel de riesgo se asigna cuando una tool intenta acceder
    a ese path, independientemente del nivel de riesgo base de la tool.
    """

    ALLOWED = auto()
    """
    El path está explícitamente permitido. No eleva el riesgo base.
    Ejemplo: ~/Documents, ~/Downloads, ~/Desktop.
    """

    SENSITIVE = auto()
    """
    El path es sensible y eleva el riesgo a HIGH.
    Requiere aprobación explícita del usuario.
    Ejemplo: ~/.config, ~/.local, archivos de configuración de usuario.
    """

    BLOCKED = auto()
    """
    El path está bloqueado. Eleva el riesgo a CRITICAL.
    PolicyEngine retorna DENY sin posibilidad de override.
    Ejemplo: ~/.ssh, /etc/shadow, credenciales del sistema.
    """


@dataclass(frozen=True)
class SensitivePathPattern:
    """
    Patrón de path con su clasificación de sensibilidad.

    Encapsula un patrón glob o regex con la razón de seguridad por la que
    ese path tiene una clasificación especial. La razón es fundamental para
    el audit log y para que los desarrolladores entiendan por qué una regla
    existe.
    """

    pattern: str
    """
    Patrón de matching del path. Puede ser:
    - Glob simple: '**/.ssh/*'
    - Prefijo de directorio: '/etc'
    - Nombre de archivo: 'id_rsa'
    El matching se hace case-insensitive en Windows, case-sensitive en Linux.
    """

    sensitivity: PathSensitivity
    """Nivel de sensibilidad de este patrón."""

    reason: str
    """
    Razón de seguridad por la que este path tiene clasificación especial.
    Documentación viva de las decisiones de seguridad.
    """

    platform: str = "all"
    """
    Plataforma donde aplica este patrón: 'linux', 'windows', 'darwin', 'all'.
    Permite definir reglas específicas por SO.
    """


# =============================================================================
# CATÁLOGO DE PATRONES SENSIBLES
# =============================================================================

# Patrones BLOQUEADOS en Linux/macOS — acceso siempre denegado
_BLOCKED_PATTERNS_UNIX: tuple[SensitivePathPattern, ...] = (
    SensitivePathPattern(
        pattern="**/.ssh",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Directorio SSH con claves privadas. Compromiso = acceso remoto no autorizado.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/.ssh/*",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Archivos SSH: claves privadas, authorized_keys, known_hosts.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/.gnupg",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Directorio GPG con claves de firma y cifrado.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/.gnupg/*",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Claves y anillos GPG del usuario.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="/etc/shadow",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Hashes de contraseñas del sistema. Compromiso = escalada de privilegios.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="/etc/passwd",
        sensitivity=PathSensitivity.BLOCKED,
        reason=(
            "Base de datos de usuarios del sistema. "
            "Lectura revela usuarios, UIDs y shells disponibles."
        ),
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="/etc/sudoers",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Configuración de sudo. Lectura revela qué usuarios tienen privilegios de root.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="/etc/sudoers.d/*",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Configuración adicional de sudo.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="/proc/**",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Sistema de archivos virtual del kernel. Acceso puede revelar info del proceso.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="/sys/**",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Sistema de archivos virtual del kernel. Escritura puede dañar el hardware.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/*id_rsa*",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Archivo de clave privada RSA SSH.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/*id_ed25519*",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Archivo de clave privada ED25519 SSH.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/*id_ecdsa*",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Archivo de clave privada ECDSA SSH.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/*.pem",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Archivo PEM — puede contener claves privadas o certificados.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/*.key",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Archivo de clave criptográfica genérica.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/*.p12",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Archivo PKCS#12 — contiene clave privada y certificado.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/*.pfx",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Archivo PFX (PKCS#12) — contiene clave privada y certificado.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/.netrc",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Archivo de credenciales de red (FTP, HTTP). Puede contener passwords en texto claro.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/.aws/credentials",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Credenciales AWS. Compromiso = acceso a infraestructura cloud.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/.aws/config",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Configuración AWS con posibles referencias a credenciales.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/.gcloud/**",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Credenciales Google Cloud SDK.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/.azure/**",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Credenciales Azure CLI.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/.docker/config.json",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Configuración Docker con tokens de registry (puede contener auth base64).",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/.kube/config",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Configuración kubectl con credenciales de cluster Kubernetes.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/*token*",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Archivo cuyo nombre indica que contiene un token de autenticación.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/*secret*",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Archivo cuyo nombre indica que contiene información secreta.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/*credential*",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Archivo cuyo nombre indica que contiene credenciales.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/.env",
        sensitivity=PathSensitivity.BLOCKED,
        reason=(
            "Archivo de variables de entorno — frecuentemente contiene "
            "API keys, passwords y secrets de aplicación."
        ),
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/.env.*",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Variantes de archivos .env (.env.local, .env.production, etc.).",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/keychain*",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Archivo o directorio de keychain — almacén de credenciales del sistema.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="/boot/**",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Sistema de arranque. Modificación puede hacer el sistema inbooteable.",
        platform="linux",
    ),
)

# Patrones BLOQUEADOS en Windows
_BLOCKED_PATTERNS_WINDOWS: tuple[SensitivePathPattern, ...] = (
    SensitivePathPattern(
        pattern="C:\\Windows\\System32\\**",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Directorio del sistema Windows. Modificación puede romper el sistema.",
        platform="windows",
    ),
    SensitivePathPattern(
        pattern="C:\\Windows\\SysWOW64\\**",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Directorio del sistema Windows 32-bit. Modificación puede romper el sistema.",
        platform="windows",
    ),
    SensitivePathPattern(
        pattern="**\\AppData\\Roaming\\Microsoft\\Credentials\\**",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Almacén de credenciales Windows DPAPI.",
        platform="windows",
    ),
    SensitivePathPattern(
        pattern="**\\AppData\\Local\\Microsoft\\Credentials\\**",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Almacén de credenciales local Windows DPAPI.",
        platform="windows",
    ),
    SensitivePathPattern(
        pattern="**\\.ssh\\**",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Directorio SSH con claves privadas.",
        platform="windows",
    ),
    SensitivePathPattern(
        pattern="**\\*.pem",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Archivo PEM — puede contener claves privadas.",
        platform="windows",
    ),
    SensitivePathPattern(
        pattern="**\\*.key",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Archivo de clave criptográfica.",
        platform="windows",
    ),
    SensitivePathPattern(
        pattern="**\\*.p12",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Archivo PKCS#12.",
        platform="windows",
    ),
    SensitivePathPattern(
        pattern="**\\*.pfx",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Archivo PFX (PKCS#12).",
        platform="windows",
    ),
    SensitivePathPattern(
        pattern="**\\.aws\\credentials",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Credenciales AWS.",
        platform="windows",
    ),
    SensitivePathPattern(
        pattern="**\\*token*",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Archivo que puede contener tokens de autenticación.",
        platform="windows",
    ),
    SensitivePathPattern(
        pattern="**\\*secret*",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Archivo que puede contener información secreta.",
        platform="windows",
    ),
    SensitivePathPattern(
        pattern="**\\.env",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Archivo de variables de entorno con posibles secrets.",
        platform="windows",
    ),
    SensitivePathPattern(
        pattern="**\\.env.*",
        sensitivity=PathSensitivity.BLOCKED,
        reason="Variantes de archivos .env.",
        platform="windows",
    ),
)

# Patrones SENSIBLES (no bloqueados, pero elevan riesgo a HIGH) — cross-platform
_SENSITIVE_PATTERNS_ALL: tuple[SensitivePathPattern, ...] = (
    SensitivePathPattern(
        pattern="**/.config/**",
        sensitivity=PathSensitivity.SENSITIVE,
        reason="Directorio de configuración de usuario. Puede contener settings sensibles.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/.local/share/**",
        sensitivity=PathSensitivity.SENSITIVE,
        reason="Datos locales de aplicaciones de usuario.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="/etc/**",
        sensitivity=PathSensitivity.SENSITIVE,
        reason="Directorio de configuración del sistema. Requiere revisión cuidadosa.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="/var/log/**",
        sensitivity=PathSensitivity.SENSITIVE,
        reason="Logs del sistema. Pueden contener información sensible de aplicaciones.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/.bash_history",
        sensitivity=PathSensitivity.SENSITIVE,
        reason="Historial de comandos bash. Revela comandos ejecutados por el usuario.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/.zsh_history",
        sensitivity=PathSensitivity.SENSITIVE,
        reason="Historial de comandos zsh.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/.profile",
        sensitivity=PathSensitivity.SENSITIVE,
        reason="Perfil de shell — puede exportar variables de entorno sensibles.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/.bashrc",
        sensitivity=PathSensitivity.SENSITIVE,
        reason="Configuración bash — puede contener aliases y variables sensibles.",
        platform="linux",
    ),
    SensitivePathPattern(
        pattern="**/.zshrc",
        sensitivity=PathSensitivity.SENSITIVE,
        reason="Configuración zsh.",
        platform="linux",
    ),
)

# Consolidar todos los patrones en una sola tupla ordenada
# (BLOCKED primero para short-circuit en el clasificador)
ALL_SENSITIVE_PATTERNS: tuple[SensitivePathPattern, ...] = (
    *_BLOCKED_PATTERNS_UNIX,
    *_BLOCKED_PATTERNS_WINDOWS,
    *_SENSITIVE_PATTERNS_ALL,
)


# =============================================================================
# SEÑALES DE RIESGO EN INPUTS
# =============================================================================


class RiskSignalType(Enum):
    """
    Tipo de señal de riesgo detectada en el input de una tool.

    Las señales no son conclusivas por sí solas — el clasificador las
    combina para producir un InputRiskAnalysis. Una sola señal puede no
    elevar el riesgo; la combinación de varias puede hacerlo.
    """

    PATH_TRAVERSAL = auto()
    """
    Intento de path traversal: '../../../etc/passwd', '..\\..\\Windows'.
    Señal de alta prioridad — siempre eleva a CRITICAL.
    """

    SENSITIVE_PATH = auto()
    """
    El path accedido coincide con un patrón sensible conocido.
    Eleva a HIGH o CRITICAL según el patrón.
    """

    SHELL_METACHARACTER = auto()
    """
    Presencia de metacaracteres de shell en argumentos de string:
    ';', '&&', '||', '`', '$(...)', '|', '>', '<', etc.
    Indica posible command injection.
    """

    NULL_BYTE = auto()
    """
    Presencia de bytes nulos ('\x00') en paths o argumentos.
    Técnica clásica de null byte injection para bypass de validaciones.
    """

    UNICODE_TRICK = auto()
    """
    Presencia de caracteres Unicode que pueden confundirse con caracteres
    ASCII (homoglyphs, RTL override, zero-width spaces).
    Técnica de evasión de filtros de seguridad.
    """

    EXCESSIVE_LENGTH = auto()
    """
    Un argumento de string excede la longitud máxima razonable.
    Puede indicar buffer overflow attempt o exfiltración de datos grandes.
    """

    SYSTEM_DIRECTORY = auto()
    """
    Acceso a un directorio del sistema operativo fuera de la allowlist.
    Elevación de riesgo basada en la localización del path.
    """

    HIDDEN_FILE = auto()
    """
    Acceso a archivos o directorios ocultos (prefijo '.').
    Frecuentemente contienen configuración y datos sensibles.
    """

    NETWORK_PATH = auto()
    """
    El path contiene una referencia a recurso de red (UNC path en Windows,
    URL file:// o smb://).
    """


@dataclass(frozen=True)
class InputRiskSignal:
    """
    Señal de riesgo detectada en el análisis de un input de tool.

    Cada señal describe un hallazgo específico que contribuye al risk level
    final del input. El conjunto de señales provee evidencia auditeable
    de por qué una acción fue clasificada con determinado riesgo.
    """

    signal_type: RiskSignalType
    """Tipo de señal detectada."""

    severity: RiskLevel
    """Severidad de esta señal específica."""

    description: str
    """Descripción legible de qué se detectó y por qué es relevante."""

    field_name: str | None = None
    """
    Nombre del campo del input donde se detectó la señal.
    None si la señal aplica al input completo.
    """

    evidence: str | None = None
    """
    Fragmento del input que activó la señal (truncado y sanitizado).
    Se incluye en el audit log para análisis forense.
    NUNCA incluir el input completo si puede contener datos sensibles.
    """


@dataclass(frozen=True)
class InputRiskAnalysis:
    """
    Resultado completo del análisis de riesgo de un input de tool.

    El DomainRiskClassifier produce un InputRiskAnalysis para cada invocación
    de tool. El PolicyEngine lo usa junto con el riesgo base de la tool para
    calcular el RiskAssessment final.
    """

    input_risk: RiskLevel
    """
    Nivel de riesgo determinado por el análisis del input.
    RiskLevel.NONE si el input no presenta señales de riesgo.
    """

    signals: tuple[InputRiskSignal, ...]
    """
    Señales de riesgo detectadas en el input.
    Tupla vacía si no hay señales (input limpio).
    """

    is_clean: bool
    """True si no se detectaron señales de riesgo."""

    summary: str
    """
    Resumen legible del análisis para el audit log.
    No incluye el contenido completo del input.
    """

    @classmethod
    def clean(cls) -> InputRiskAnalysis:
        """Factory: crea un análisis limpio (sin señales de riesgo)."""
        return cls(
            input_risk=RiskLevel.NONE,
            signals=(),
            is_clean=True,
            summary="Input analizado sin señales de riesgo detectadas.",
        )

    @classmethod
    def with_signals(
        cls,
        signals: list[InputRiskSignal],
    ) -> InputRiskAnalysis:
        """
        Factory: crea un análisis con señales de riesgo.

        El input_risk se calcula como el máximo de las severidades de todas
        las señales detectadas.

        Args:
            signals: Lista de señales detectadas (no vacía).

        Returns:
            InputRiskAnalysis con el riesgo máximo de las señales.
        """
        if not signals:
            return cls.clean()

        max_risk = max(s.severity for s in signals)
        signal_types = [s.signal_type.name for s in signals]
        summary = (
            f"Detectadas {len(signals)} señal(es) de riesgo: "
            f"{', '.join(signal_types)}. "
            f"Riesgo del input: {max_risk.name}."
        )

        return cls(
            input_risk=max_risk,
            signals=tuple(signals),
            is_clean=False,
            summary=summary,
        )


# =============================================================================
# CLASIFICADOR DE RIESGO DEL DOMINIO
# =============================================================================


class DomainRiskClassifier:
    """
    Clasificador de riesgo específico del dominio HiperForge User.

    Implementa todas las reglas de clasificación de riesgo del producto:
    análisis de paths, detección de señales maliciosas en inputs, y cálculo
    del riesgo efectivo combinando el riesgo base de la tool con el análisis
    del input concreto.

    Todas las funciones son puras (sin efectos secundarios, sin estado mutable).
    La instancia puede usarse como singleton o crearse por request — ambos
    patrones son seguros.

    Uso:
        classifier = DomainRiskClassifier()
        analysis = classifier.analyze_input(tool_id="file_read", arguments=args)
        if not analysis.is_clean:
            logger.warning("riesgo_detectado", signals=analysis.signals)
    """

    # Patrones de path traversal — detectan secuencias '../' y variantes
    _PATH_TRAVERSAL_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"\.\.[/\\]", re.IGNORECASE),          # ../ o ..\
        re.compile(r"\.\.[/\\]\.\.[/\\]"),                 # ../../
        re.compile(r"%2e%2e[%2f%5c]", re.IGNORECASE),    # URL-encoded ../
        re.compile(r"%252e%252e[%252f%255c]", re.IGNORECASE),  # doble URL-encoded
        re.compile(r"\.\.%[25][fc2f5c]", re.IGNORECASE), # mixed encoding
        re.compile(r"\x00"),                               # null byte
    )

    # Metacaracteres de shell que indican posible command injection
    _SHELL_METACHAR_PATTERN: re.Pattern[str] = re.compile(
        r"[;&|`$<>]"
        r"|&&|\|\|"
        r"|\$\("
        r"|\$\{"
        r"|`[^`]*`",
        re.IGNORECASE,
    )

    # Caracteres Unicode potencialmente peligrosos (confusables, control chars)
    _UNICODE_DANGEROUS_PATTERN: re.Pattern[str] = re.compile(
        r"[\u202a-\u202e]"   # RTL/LTR override
        r"|[\u2066-\u2069]"  # directional isolates
        r"|\u200b"           # zero-width space
        r"|\u200c"           # zero-width non-joiner
        r"|\u200d"           # zero-width joiner
        r"|\ufeff"           # BOM/zero-width no-break space
        r"|\u00ad",          # soft hyphen
    )

    # Longitud máxima razonable para un argumento de string
    _MAX_ARGUMENT_LENGTH: int = 10_000

    def __init__(self) -> None:
        """
        Inicializa el clasificador con el conjunto activo de patrones.

        Determina la plataforma actual para aplicar solo los patrones
        relevantes al SO en ejecución.
        """
        self._platform = self._detect_platform()
        self._active_patterns = self._build_active_patterns()

    @staticmethod
    def _detect_platform() -> str:
        """
        Detecta la plataforma actual del sistema operativo.

        Returns:
            'linux', 'windows', 'darwin', o 'unknown'.
        """
        platform_map = {
            "linux": "linux",
            "win32": "windows",
            "darwin": "darwin",
        }
        return platform_map.get(sys.platform, "unknown")

    def _build_active_patterns(self) -> list[SensitivePathPattern]:
        """
        Construye la lista de patrones activos para la plataforma actual.

        Filtra los patrones del catálogo global para incluir solo los que
        aplican a la plataforma actual o son cross-platform ('all').

        Returns:
            Lista de SensitivePathPattern activos, ordenados con BLOCKED primero.
        """
        active = [
            p for p in ALL_SENSITIVE_PATTERNS
            if p.platform == "all"
            or p.platform == self._platform
        ]
        # Ordenar: BLOCKED primero (para short-circuit), luego SENSITIVE
        return sorted(
            active,
            key=lambda p: (
                0 if p.sensitivity == PathSensitivity.BLOCKED else 1
            ),
        )

    def classify_path(self, path_str: str) -> PathSensitivity:
        """
        Clasifica un path según los patrones de sensibilidad del sistema.

        Evalúa el path contra todos los patrones activos (BLOCKED primero).
        Retorna la clasificación del primer patrón que coincida.
        Si ningún patrón coincide, el path se considera ALLOWED.

        La normalización del path se realiza antes del matching para evitar
        evasión mediante dobles slashes, mixed separators, o paths relativos.

        Args:
            path_str: String del path a clasificar.

        Returns:
            PathSensitivity del path.
        """
        if not path_str:
            return PathSensitivity.ALLOWED

        # Normalizar el path a su forma canónica
        normalized = self._normalize_path(path_str)
        normalized_lower = normalized.lower()

        for pattern in self._active_patterns:
            if self._matches_pattern(normalized_lower, pattern.pattern.lower()):
                return pattern.sensitivity

        return PathSensitivity.ALLOWED

    def get_path_block_reason(self, path_str: str) -> str | None:
        """
        Retorna la razón por la que un path está bloqueado o es sensible.

        Args:
            path_str: String del path a consultar.

        Returns:
            Razón de la clasificación, o None si el path es ALLOWED.
        """
        normalized = self._normalize_path(path_str)
        normalized_lower = normalized.lower()

        for pattern in self._active_patterns:
            if pattern.sensitivity != PathSensitivity.ALLOWED:
                if self._matches_pattern(normalized_lower, pattern.pattern.lower()):
                    return pattern.reason

        return None

    def analyze_input(
        self,
        tool_id: str,
        arguments: dict[str, Any],
    ) -> InputRiskAnalysis:
        """
        Analiza el input completo de una invocación de tool en busca de señales
        de riesgo.

        Examina todos los argumentos del input buscando:
        - Path traversal attempts
        - Paths que coinciden con patrones sensibles
        - Shell metacaracteres (command injection indicators)
        - Null bytes
        - Caracteres Unicode peligrosos
        - Argumentos excesivamente largos

        Args:
            tool_id:   ID de la tool (para contexto en el análisis).
            arguments: Diccionario de argumentos a analizar.

        Returns:
            InputRiskAnalysis con las señales detectadas y el riesgo resultante.
        """
        signals: list[InputRiskSignal] = []

        for field_name, value in arguments.items():
            if isinstance(value, str):
                field_signals = self._analyze_string_argument(
                    field_name=field_name,
                    value=value,
                )
                signals.extend(field_signals)

            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        item_signals = self._analyze_string_argument(
                            field_name=f"{field_name}[{i}]",
                            value=item,
                        )
                        signals.extend(item_signals)

        if not signals:
            return InputRiskAnalysis.clean()

        return InputRiskAnalysis.with_signals(signals)

    def _analyze_string_argument(
        self,
        field_name: str,
        value: str,
    ) -> list[InputRiskSignal]:
        """
        Analiza un argumento de tipo string en busca de señales de riesgo.

        Args:
            field_name: Nombre del campo en el schema de la tool.
            value:      Valor del argumento a analizar.

        Returns:
            Lista de InputRiskSignal detectadas. Vacía si el argumento es limpio.
        """
        signals: list[InputRiskSignal] = []

        # 1. Null bytes — siempre CRITICAL (null byte injection)
        if "\x00" in value:
            signals.append(InputRiskSignal(
                signal_type=RiskSignalType.NULL_BYTE,
                severity=RiskLevel.CRITICAL,
                description=(
                    "Byte nulo (\\x00) detectado en el argumento. "
                    "Técnica clásica de null byte injection."
                ),
                field_name=field_name,
                evidence="[CONTIENE NULL BYTE]",
            ))
            # Null byte es suficiente para CRITICAL — no seguir analizando este campo
            return signals

        # 2. Path traversal — siempre CRITICAL
        for pattern in self._PATH_TRAVERSAL_PATTERNS:
            if pattern.search(value):
                signals.append(InputRiskSignal(
                    signal_type=RiskSignalType.PATH_TRAVERSAL,
                    severity=RiskLevel.CRITICAL,
                    description=(
                        "Secuencia de path traversal detectada. "
                        "Intento de acceso a paths fuera del directorio base."
                    ),
                    field_name=field_name,
                    evidence=self._safe_truncate(value, 50),
                ))
                return signals  # CRITICAL — no continuar

        # 3. Caracteres Unicode peligrosos — HIGH
        if self._UNICODE_DANGEROUS_PATTERN.search(value):
            signals.append(InputRiskSignal(
                signal_type=RiskSignalType.UNICODE_TRICK,
                severity=RiskLevel.HIGH,
                description=(
                    "Caracteres Unicode de control o confusión detectados. "
                    "Posible intento de evasión de filtros mediante homoglyphs o "
                    "caracteres de control bidireccional."
                ),
                field_name=field_name,
                evidence="[CONTIENE UNICODE PELIGROSO]",
            ))

        # 4. Shell metacaracteres — HIGH (solo en campos que pueden ser comandos/paths)
        _path_like_fields = frozenset({
            "path", "file_path", "filepath", "directory", "dir",
            "command", "cmd", "args", "arguments", "query",
            "url", "uri", "source", "destination", "dest",
            "app", "application", "executable",
        })
        field_lower = field_name.lower()
        if any(pf in field_lower for pf in _path_like_fields):
            if self._SHELL_METACHAR_PATTERN.search(value):
                signals.append(InputRiskSignal(
                    signal_type=RiskSignalType.SHELL_METACHARACTER,
                    severity=RiskLevel.HIGH,
                    description=(
                        "Metacaracter(es) de shell detectado(s) en campo de path/comando. "
                        "Posible intento de command injection."
                    ),
                    field_name=field_name,
                    evidence=self._safe_truncate(value, 50),
                ))

        # 5. Análisis de path sensible (solo si el campo parece un path)
        _path_field_indicators = frozenset({
            "path", "file", "dir", "directory", "folder",
            "source", "destination", "dest", "target",
        })
        if any(ind in field_lower for ind in _path_field_indicators):
            path_sensitivity = self.classify_path(value)
            if path_sensitivity == PathSensitivity.BLOCKED:
                block_reason = self.get_path_block_reason(value) or "Path bloqueado por política."
                signals.append(InputRiskSignal(
                    signal_type=RiskSignalType.SENSITIVE_PATH,
                    severity=RiskLevel.CRITICAL,
                    description=f"Path bloqueado: {block_reason}",
                    field_name=field_name,
                    evidence=self._safe_truncate(value, 80),
                ))
            elif path_sensitivity == PathSensitivity.SENSITIVE:
                sens_reason = self.get_path_block_reason(value) or "Path sensible."
                signals.append(InputRiskSignal(
                    signal_type=RiskSignalType.SENSITIVE_PATH,
                    severity=RiskLevel.HIGH,
                    description=f"Path sensible: {sens_reason}",
                    field_name=field_name,
                    evidence=self._safe_truncate(value, 80),
                ))

            # 6. Archivos ocultos (prefijo '.') — LOW
            path_obj = Path(value)
            if path_obj.name.startswith(".") and path_sensitivity == PathSensitivity.ALLOWED:
                signals.append(InputRiskSignal(
                    signal_type=RiskSignalType.HIDDEN_FILE,
                    severity=RiskLevel.LOW,
                    description=(
                        f"El archivo '{path_obj.name}' es un archivo oculto. "
                        "Los archivos ocultos frecuentemente contienen configuración."
                    ),
                    field_name=field_name,
                    evidence=path_obj.name,
                ))

        # 7. Longitud excesiva — MEDIUM
        if len(value) > self._MAX_ARGUMENT_LENGTH:
            signals.append(InputRiskSignal(
                signal_type=RiskSignalType.EXCESSIVE_LENGTH,
                severity=RiskLevel.MEDIUM,
                description=(
                    f"El argumento tiene {len(value)} caracteres, "
                    f"superando el máximo razonable de {self._MAX_ARGUMENT_LENGTH}."
                ),
                field_name=field_name,
                evidence=f"longitud={len(value)}",
            ))

        return signals

    def build_risk_assessment(
        self,
        tool_id: str,
        base_risk: RiskLevel,
        input_analysis: InputRiskAnalysis,
    ) -> RiskAssessment:
        """
        Construye el RiskAssessment completo combinando el riesgo base de la
        tool con el análisis del input concreto.

        Es el puente entre el análisis del dominio y el RiskAssessment del
        protocolo de tools que el PolicyEngine consume.

        Args:
            tool_id:        ID de la tool siendo evaluada.
            base_risk:      Riesgo base declarado por la tool en su ToolCapability.
            input_analysis: Resultado del análisis del input concreto.

        Returns:
            RiskAssessment completo listo para el PolicyEngine.
        """
        effective_risk = max(base_risk, input_analysis.input_risk)

        if effective_risk == RiskLevel.CRITICAL:
            # Construir razón de bloqueo desde las señales
            if input_analysis.signals:
                blocking_signals = [
                    s.description for s in input_analysis.signals
                    if s.severity == RiskLevel.CRITICAL
                ]
                reason = " | ".join(blocking_signals) if blocking_signals else (
                    "Input clasificado como riesgo crítico por las políticas del sistema."
                )
            else:
                reason = (
                    f"La tool '{tool_id}' está clasificada con riesgo crítico "
                    "y no puede ejecutarse."
                )
            return RiskAssessment.block(
                tool_id=tool_id,
                base_risk=base_risk,
                reason=reason,
            )

        elif effective_risk >= RiskLevel.HIGH or input_analysis.input_risk >= RiskLevel.MEDIUM:
            elevated_by = [s.description for s in input_analysis.signals]
            justification = input_analysis.summary

            return RiskAssessment.require_approval(
                tool_id=tool_id,
                base_risk=base_risk,
                effective_risk=effective_risk,
                input_risk=input_analysis.input_risk,
                justification=justification,
                elevated_by=elevated_by,
            )

        else:
            return RiskAssessment.allow_direct(
                tool_id=tool_id,
                base_risk=effective_risk,
            )

    @staticmethod
    def _normalize_path(path_str: str) -> str:
        """
        Normaliza un path a su forma canónica para comparación segura.

        Maneja paths de Linux, Windows y mixed separators. La normalización
        previene evasión mediante variaciones de formato equivalentes:
        /etc//shadow, /etc/./shadow, /etc/subdir/../shadow.

        Args:
            path_str: Path a normalizar.

        Returns:
            String normalizado. Si la normalización falla, retorna el original
            en minúsculas (fail-safe — el patrón matching puede ser menos preciso
            pero no bloqueará falsamente).
        """
        try:
            if sys.platform == "win32":
                normalized = str(PureWindowsPath(path_str))
            else:
                # Usar PurePosixPath para normalización básica,
                # luego Path().resolve() si es un path absoluto existente
                normalized = str(PurePosixPath(path_str))
            return normalized
        except (ValueError, TypeError):
            return path_str

    @staticmethod
    def _matches_pattern(path_lower: str, pattern_lower: str) -> bool:
        """
        Verifica si un path (en minúsculas) coincide con un patrón glob.

        Implementa un subset de glob matching suficiente para los patrones
        de seguridad de este módulo:
        - '**' coincide con cualquier secuencia de directorios
        - '*' coincide con cualquier secuencia dentro de un directorio
        - El resto es matching literal

        Args:
            path_lower:    Path normalizado en minúsculas.
            pattern_lower: Patrón en minúsculas.

        Returns:
            True si el path coincide con el patrón.
        """
        from fnmatch import fnmatch

        # Normalizar separadores para comparación cross-platform
        path_normalized = path_lower.replace("\\", "/")
        pattern_normalized = pattern_lower.replace("\\", "/")

        # Matching directo con el path completo
        if fnmatch(path_normalized, pattern_normalized):
            return True

        # Matching contra cada componente del path (para patrones como '**/.ssh')
        # Construir todos los subpaths posibles
        parts = path_normalized.split("/")
        for i in range(len(parts)):
            subpath = "/".join(parts[i:])
            if fnmatch(subpath, pattern_normalized.lstrip("*/")):
                return True

        # Para patrones con '**', verificar si algún segmento del path
        # coincide con la parte después del '**'
        if "**" in pattern_normalized:
            suffix = pattern_normalized.split("**/", 1)[-1]
            if fnmatch(path_normalized, f"*/{suffix}") or fnmatch(path_normalized, suffix):
                return True

        return False

    @staticmethod
    def _safe_truncate(value: str, max_length: int) -> str:
        """
        Trunca un string de forma segura para incluir en evidencia de audit.

        Nunca incluye más de max_length caracteres para evitar que el audit
        log registre el contenido completo de un input potencialmente large.

        Args:
            value:      String a truncar.
            max_length: Longitud máxima del resultado.

        Returns:
            String truncado con indicador de truncación si fue necesario.
        """
        if len(value) <= max_length:
            return repr(value)
        return repr(value[:max_length]) + "...[truncado]"