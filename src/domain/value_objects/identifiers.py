"""
Identificadores tipados del dominio HiperForge User.

Este módulo define todos los tipos de identificador (IDs) del sistema como
value objects del dominio. Usar strings crudos como IDs es una fuente clásica
de bugs — pasar un SessionId donde se espera un TaskId compila y falla en runtime.
Los tipos wrapper eliminan esta clase entera de errores en tiempo de diseño.

Cada identificador hereda de BaseId, que encapsula un ULID (Universally Unique
Lexicographically Sortable Identifier). Los ULIDs tienen propiedades superiores
a los UUIDs para este sistema:

  - Ordenables lexicográficamente por tiempo de creación: los IDs más recientes
    son lexicográficamente mayores. Esto permite ordenar entidades por creación
    sin columna de timestamp adicional.
  - Globalmente únicos: 128 bits (48 timestamp + 80 random). La probabilidad
    de colisión es negligible incluso a escala global.
  - URL-safe: 26 caracteres en Base32 Crockford, sin caracteres especiales.
    Seguros para usar en rutas de API, nombres de archivo, y como claves de DB.
  - Humanos-legibles: más cortos que UUIDs con guiones y reconocibles por su
    prefijo de tipo (sess_, task_, etc.).

Formato de los IDs en este sistema:
    {prefijo}_{ulid_base32}
    Ejemplo: sess_01HZXK3B2YJGN8RQVWP4M5TC6

Diseño de type safety:
  - BaseId es inmutable (slots + __hash__ basado en valor).
  - Cada subtipo solo acepta strings con su prefijo correcto.
  - La comparación entre tipos diferentes siempre retorna False (no son iguales
    un SessionId y un TaskId aunque tengan el mismo valor interno).
  - Los IDs son hashables y pueden usarse como keys de dict y elementos de set.
  - Serialización/deserialización vía to_str() y from_str() con validación.

Uso:
    from src.domain.value_objects.identifiers import SessionId, TaskId

    session_id = SessionId.generate()        # crea uno nuevo
    task_id = TaskId.generate()

    # Type safety en funciones:
    def get_session(sid: SessionId) -> Session: ...
    get_session(task_id)  # mypy/pyright error — correcto

    # Serialización para storage:
    stored = session_id.to_str()             # "sess_01HZXK3B2YJGN8RQVWP4M5TC6"
    restored = SessionId.from_str(stored)    # valida prefijo y formato

    # Comparación:
    assert session_id == SessionId.from_str(stored)
    assert session_id != task_id            # tipos diferentes → siempre False
"""

from __future__ import annotations

import re
import time
import os
import struct
from typing import ClassVar, TypeVar

T = TypeVar("T", bound="BaseId")

# Alfabeto Base32 Crockford (sin I, L, O, U para evitar confusión visual)
_CROCKFORD_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_CROCKFORD_DECODE: dict[str, int] = {c: i for i, c in enumerate(_CROCKFORD_ALPHABET)}

# Regex de validación de ULID (26 caracteres Base32 Crockford, case-insensitive)
_ULID_PATTERN = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$", re.IGNORECASE)


def _generate_ulid() -> str:
    """
    Genera un ULID (Universally Unique Lexicographically Sortable Identifier).

    Implementación directa sin dependencia de librería externa para garantizar
    control total sobre el formato y evitar dependencias opcionales en el dominio.

    Estructura del ULID (128 bits):
      - 48 bits: timestamp en milisegundos desde epoch Unix
      - 80 bits: componente aleatorio (os.urandom para criptografía-grade)

    El resultado se codifica en Base32 Crockford (26 caracteres, case-insensitive).

    Returns:
        String de 26 caracteres en Base32 Crockford representando el ULID.
    """
    # Timestamp: 48 bits = milisegundos desde epoch
    timestamp_ms = int(time.time() * 1000)

    # Componente aleatorio: 10 bytes (80 bits) de os.urandom
    random_bytes = os.urandom(10)

    # Combinar en 16 bytes (128 bits): 6 bytes timestamp + 10 bytes random
    timestamp_bytes = struct.pack(">Q", timestamp_ms)[2:]  # últimos 6 bytes de uint64
    raw = timestamp_bytes + random_bytes  # 16 bytes total

    # Convertir 16 bytes a 128 bits y codificar en Base32 Crockford (26 chars)
    # Cada carácter codifica 5 bits → 26 * 5 = 130 bits ≥ 128 bits
    # Usamos aritmética de enteros para la conversión bit a bit
    value = int.from_bytes(raw, "big")

    # Codificar los 128 bits más bajos en 26 caracteres (130 bits de espacio)
    chars: list[str] = []
    for _ in range(26):
        chars.append(_CROCKFORD_ALPHABET[value & 0x1F])
        value >>= 5

    # El primer carácter debe ser ≤ '7' (128 bits, primer char de 3 bits)
    # Ajustar para que el timestamp ocupe los bits más altos (big-endian)
    return "".join(reversed(chars))


def _validate_ulid(value: str) -> bool:
    """
    Valida que un string sea un ULID Base32 Crockford válido.

    Args:
        value: String a validar.

    Returns:
        True si el string tiene 26 caracteres en el alfabeto Crockford.
    """
    return bool(_ULID_PATTERN.match(value.upper()))


# =============================================================================
# CLASE BASE
# =============================================================================


class BaseId:
    """
    Value object base para todos los identificadores del dominio.

    Implementa identidad por valor (value semantics): dos instancias con el
    mismo valor string son iguales, independientemente de su identidad de objeto.
    La igualdad incluye el tipo — un SessionId y un TaskId con el mismo valor
    ULID nunca son iguales.

    Atributos de clase que las subclases DEBEN definir:
        PREFIX: str — prefijo del tipo de ID (ej: "sess", "task").
                      Debe ser corto, descriptivo y único en el sistema.

    Implementación interna:
        _value: str — el ULID completo con prefijo (ej: "sess_01HZ...").
    """

    # Las subclases definen este atributo con su prefijo específico
    PREFIX: ClassVar[str] = ""

    # Optimización de memoria: __slots__ en lugar de __dict__
    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        """
        Inicializa el ID con un valor ya validado.

        No usar directamente — usar generate() o from_str() en su lugar.
        El constructor asume que el valor fue validado por from_str().

        Args:
            value: String completo con prefijo (ej: "sess_01HZXK3B2YJGN8RQVWP4M5TC6").
        """
        object.__setattr__(self, "_value", value)

    def __setattr__(self, name: str, value: object) -> None:
        """Impide la mutación del ID después de la creación (inmutabilidad)."""
        raise AttributeError(
            f"{type(self).__name__} es inmutable. "
            f"No se puede modificar el atributo '{name}'."
        )

    def __delattr__(self, name: str) -> None:
        """Impide la eliminación de atributos."""
        raise AttributeError(
            f"{type(self).__name__} es inmutable. "
            f"No se puede eliminar el atributo '{name}'."
        )

    @classmethod
    def generate(cls: type[T]) -> T:
        """
        Genera un nuevo ID único.

        Crea un ULID nuevo y lo combina con el PREFIX de la subclase.
        Cada llamada a generate() produce un ID globalmente único.

        Returns:
            Nueva instancia del tipo de ID con un ULID recién generado.

        Example:
            session_id = SessionId.generate()
            # session_id.to_str() → "sess_01HZXK3B2YJGN8RQVWP4M5TC6"
        """
        ulid = _generate_ulid()
        value = f"{cls.PREFIX}_{ulid}"
        return cls(value)

    @classmethod
    def from_str(cls: type[T], value: str) -> T:
        """
        Crea un ID a partir de su representación string.

        Valida que el string tenga el formato correcto (prefijo + ULID).
        Útil para deserializar IDs almacenados en base de datos o recibidos
        de la UI.

        Args:
            value: String con formato "{PREFIX}_{ULID}" (case-insensitive para ULID).

        Returns:
            Instancia del tipo de ID.

        Raises:
            ValueError: Si el string no tiene el formato esperado para este tipo.

        Example:
            session_id = SessionId.from_str("sess_01HZXK3B2YJGN8RQVWP4M5TC6")
        """
        if not isinstance(value, str):
            raise ValueError(
                f"El valor debe ser un string, se recibió {type(value).__name__}."
            )

        # Validar prefijo
        expected_prefix = f"{cls.PREFIX}_"
        if not value.upper().startswith(expected_prefix.upper()):
            raise ValueError(
                f"ID inválido para {cls.__name__}: se esperaba prefijo '{cls.PREFIX}_', "
                f"se recibió '{value[:len(expected_prefix)]}'."
            )

        # Extraer y validar el ULID
        ulid_part = value[len(expected_prefix):]
        if not _validate_ulid(ulid_part):
            raise ValueError(
                f"ID inválido para {cls.__name__}: el componente ULID '{ulid_part}' "
                f"no es un ULID válido. Debe ser 26 caracteres Base32 Crockford."
            )

        # Normalizar a mayúsculas para consistencia interna
        normalized = f"{cls.PREFIX}_{ulid_part.upper()}"
        return cls(normalized)

    @classmethod
    def from_str_unsafe(cls: type[T], value: str) -> T:
        """
        Crea un ID desde string sin validación completa.

        SOLO para uso interno en el ORM/storage cuando el valor viene de la
        base de datos y se garantiza que fue validado al escribirse.
        No usar en código de aplicación o dominio.

        Args:
            value: String del ID (se asume válido).

        Returns:
            Instancia del tipo de ID sin validación.
        """
        return cls(value)

    def to_str(self) -> str:
        """
        Retorna la representación string del ID.

        Returns:
            String en formato "{PREFIX}_{ULID}" con ULID en mayúsculas.
        """
        return self._value  # type: ignore[return-value]

    def ulid_part(self) -> str:
        """
        Retorna solo el componente ULID sin el prefijo.

        Útil para logging donde el prefijo es implícito por el contexto.

        Returns:
            String de 26 caracteres ULID.
        """
        return self._value.split("_", 1)[1]  # type: ignore[union-attr]

    def __str__(self) -> str:
        return self._value  # type: ignore[return-value]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._value!r})"

    def __eq__(self, other: object) -> bool:
        """
        Igualdad por valor Y por tipo.

        Un SessionId y un TaskId con el mismo ULID NO son iguales.
        Esto es intencional — los tipos diferentes representan entidades diferentes.
        """
        if type(self) is not type(other):
            return False
        return self._value == other._value  # type: ignore[union-attr]

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """
        Hash basado en tipo y valor.

        Incluye el tipo en el hash para que SessionId y TaskId con el mismo
        ULID tengan hashes diferentes — crucial para uso como claves de dict.
        """
        return hash((type(self).__name__, self._value))

    def __lt__(self, other: object) -> bool:
        """
        Comparación lexicográfica.

        Como ULIDs son lexicográficamente ordenables por tiempo de creación,
        ID1 < ID2 implica que ID1 fue creado antes que ID2 (con alta probabilidad).

        Solo permite comparación entre IDs del mismo tipo.

        Raises:
            TypeError: Si se intenta comparar IDs de tipos diferentes.
        """
        if type(self) is not type(other):
            raise TypeError(
                f"No se pueden comparar {type(self).__name__} y {type(other).__name__}. "
                f"Solo se pueden comparar IDs del mismo tipo."
            )
        return self._value < other._value  # type: ignore[operator,union-attr]

    def __le__(self, other: object) -> bool:
        return self == other or self.__lt__(other)

    def __gt__(self, other: object) -> bool:
        if type(self) is not type(other):
            raise TypeError(
                f"No se pueden comparar {type(self).__name__} y {type(other).__name__}."
            )
        return self._value > other._value  # type: ignore[operator,union-attr]

    def __ge__(self, other: object) -> bool:
        return self == other or self.__gt__(other)

    def __bool__(self) -> bool:
        """Un ID siempre es truthy — nunca puede ser vacío si fue construido correctamente."""
        return True


# =============================================================================
# IDENTIFICADORES DEL DOMINIO HIPERFORGE USER
# =============================================================================


class SessionId(BaseId):
    """
    Identificador único de una sesión conversacional.

    Una sesión encapsula toda la conversación del usuario con el agente,
    incluyendo el historial de turns, artifacts producidos y tareas ejecutadas.
    Las sesiones persisten entre reinicios de la aplicación.

    Formato: sess_{ULID}
    Ejemplo: sess_01HZXK3B2YJGN8RQVWP4M5TC6
    """

    PREFIX: ClassVar[str] = "sess"


class TaskId(BaseId):
    """
    Identificador único de una tarea ejecutada por el agente.

    Una tarea representa una intención del usuario que requiere ejecución
    de una o más tools. Puede incluir un plan multi-step. Las tareas están
    siempre asociadas a una sesión.

    Formato: task_{ULID}
    Ejemplo: task_01HZXM8D3FJKN5SQVWP9T2UC7
    """

    PREFIX: ClassVar[str] = "task"


class PlanId(BaseId):
    """
    Identificador único de un plan de ejecución multi-step.

    Un plan es la secuencia de pasos que el LightweightPlanner genera para
    resolver una tarea compleja. Solo existe cuando la tarea requiere
    planificación (no para tool calls únicos).

    Formato: plan_{ULID}
    Ejemplo: plan_01HZXN9E4GKLN6TRWQ0U3VD8
    """

    PREFIX: ClassVar[str] = "plan"


class StepId(BaseId):
    """
    Identificador único de un paso dentro de un plan.

    Cada PlannedStep dentro de un Plan tiene su propio StepId. Los pasos
    se ejecutan secuencialmente (con posibles dependencias entre ellos).

    Formato: step_{ULID}
    Ejemplo: step_01HZXP0F5HMMNP7USXR1V4WE9
    """

    PREFIX: ClassVar[str] = "step"


class InvocationId(BaseId):
    """
    Identificador único de una invocación de tool.

    Cada vez que el sistema invoca una tool (independientemente de si es
    parte de un plan o directa), se genera un InvocationId. Correlaciona
    la invocación con su ToolCall del LLM, el audit entry, y el ToolOutput.

    Formato: invk_{ULID}
    Ejemplo: invk_01HZXQ1G6JNNQ8VTYR2W5XF0
    """

    PREFIX: ClassVar[str] = "invk"


class ArtifactId(BaseId):
    """
    Identificador único de un artifact producido por el agente.

    Los artifacts son los productos del trabajo del agente: resúmenes,
    reportes, archivos generados, resultados de búsqueda, flashcards, etc.
    Persisten en el ArtifactStore y son accesibles durante toda la sesión.

    Formato: artf_{ULID}
    Ejemplo: artf_01HZXR2H7KPPRS9WUZS3Y6G1
    """

    PREFIX: ClassVar[str] = "artf"


class TurnId(BaseId):
    """
    Identificador único de un turno de conversación.

    Un turno representa un par request/response en la conversación:
    el mensaje del usuario y la respuesta del asistente (con sus tool calls
    y resultados incluidos). Los turns forman el historial de la sesión.

    Formato: turn_{ULID}
    Ejemplo: turn_01HZXS3J8LQQST0XV0T4Z7H2
    """

    PREFIX: ClassVar[str] = "turn"


class ApprovalId(BaseId):
    """
    Identificador único de una solicitud de aprobación.

    Cuando el agente necesita ejecutar una acción de riesgo MEDIUM o superior,
    crea una ApprovalRequest con este ID. El usuario ve la solicitud y aprueba
    o deniega. El ID correlaciona la solicitud con el audit entry.

    Formato: aprv_{ULID}
    Ejemplo: aprv_01HZXT4K9MRRTU1YW1U5A8I3
    """

    PREFIX: ClassVar[str] = "aprv"


class UserId(BaseId):
    """
    Identificador único del usuario.

    En V1, hay exactamente un usuario por instalación (single-user local).
    El UserId está preparado para sistemas multi-usuario en V3 donde
    diferentes perfiles coexistirán con sus propias sesiones y permisos.

    Formato: user_{ULID}
    Ejemplo: user_01HZXU5L0NSSUVP2ZXV6B9J4
    """

    PREFIX: ClassVar[str] = "user"


class MemoryEntryId(BaseId):
    """
    Identificador único de una entrada de memoria a largo plazo.

    Las entradas de memoria persisten información del usuario entre sesiones:
    preferencias inferidas, hechos recordados, contexto de estudio, etc.
    Se gestionan por el MemoryManager y persisten en el storage local.

    Formato: memo_{ULID}
    Ejemplo: memo_01HZXV6M1PTTUW3A0YW7C0K5
    """

    PREFIX: ClassVar[str] = "memo"


class RequestId(BaseId):
    """
    Identificador único de una request al LLM.

    Se genera por cada invocación al LLMPort y se propaga como header
    en las requests HTTP al provider (cuando está soportado). Permite
    correlacionar logs de Forge con logs del provider para debugging.

    Formato: req_{ULID}
    Ejemplo: req_01HZXW7N2QUUVX4B1ZX8D1L6
    """

    PREFIX: ClassVar[str] = "req"


class TraceId(BaseId):
    """
    Identificador de traza para correlación de operaciones distribuidas.

    Agrupa todas las operaciones de una request de usuario de extremo a
    extremo: desde el mensaje del usuario hasta la respuesta final.
    Se propaga por el sistema via ContextVar del módulo de logging.

    Formato: trc_{ULID}
    Ejemplo: trc_01HZXX8P3RVVWY5C2AY9E2M7
    """

    PREFIX: ClassVar[str] = "trc"


# =============================================================================
# REGISTRO DE TIPOS (para deserialización genérica)
# =============================================================================

# Mapa de prefijo → clase de ID. Usado por from_unknown_str() para
# deserializar IDs sin conocer su tipo de antemano (ej: en el ORM).
_ID_REGISTRY: dict[str, type[BaseId]] = {
    SessionId.PREFIX: SessionId,
    TaskId.PREFIX: TaskId,
    PlanId.PREFIX: PlanId,
    StepId.PREFIX: StepId,
    InvocationId.PREFIX: InvocationId,
    ArtifactId.PREFIX: ArtifactId,
    TurnId.PREFIX: TurnId,
    ApprovalId.PREFIX: ApprovalId,
    UserId.PREFIX: UserId,
    MemoryEntryId.PREFIX: MemoryEntryId,
    RequestId.PREFIX: RequestId,
    TraceId.PREFIX: TraceId,
}


def from_unknown_str(value: str) -> BaseId:
    """
    Deserializa un ID sin conocer su tipo de antemano.

    Detecta el tipo a partir del prefijo y retorna la instancia tipada correcta.
    Útil en el ORM cuando la columna almacena IDs de diferentes tipos, o en
    código genérico que trabaja con IDs polimórficos.

    Args:
        value: String con formato "{PREFIX}_{ULID}".

    Returns:
        Instancia del tipo de ID correspondiente al prefijo.

    Raises:
        ValueError: Si el prefijo no corresponde a ningún tipo registrado,
                    o si el ULID no es válido.

    Example:
        id_obj = from_unknown_str("sess_01HZXK3B2YJGN8RQVWP4M5TC6")
        assert isinstance(id_obj, SessionId)
    """
    if "_" not in value:
        raise ValueError(
            f"Formato de ID inválido: '{value}'. "
            f"Se esperaba formato '{{PREFIX}}_{{ULID}}'."
        )

    prefix, _ = value.split("_", 1)
    prefix_lower = prefix.lower()

    id_class = _ID_REGISTRY.get(prefix_lower)
    if id_class is None:
        registered = sorted(_ID_REGISTRY.keys())
        raise ValueError(
            f"Prefijo de ID desconocido: '{prefix}'. "
            f"Prefijos registrados: {registered}."
        )

    return id_class.from_str(value)


def is_valid_id(value: str) -> bool:
    """
    Verifica si un string es un ID válido del sistema (cualquier tipo).

    Args:
        value: String a verificar.

    Returns:
        True si el string es un ID válido de cualquier tipo registrado.
    """
    try:
        from_unknown_str(value)
        return True
    except (ValueError, AttributeError):
        return False