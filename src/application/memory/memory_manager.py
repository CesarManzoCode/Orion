"""
MemoryManager — gestor de memoria del agente HiperForge User.

El MemoryManager es el servicio de aplicación responsable de toda la gestión
de memoria del agente: qué recuerda de la conversación actual (memoria
a corto plazo), qué persiste entre sesiones (memoria a largo plazo), y
cómo se compacta el historial cuando el context window se llena.

Tipos de memoria implementados en V1:

  Memoria de corto plazo (in-session):
    - El historial de turns se mantiene en la Session y se carga desde SQLite.
    - El ContextBuilder selecciona qué turns incluir en el context window.
    - Cuando el historial excede el 85% del budget → compactación.

  Memoria de largo plazo (cross-session):
    - Hechos recordados explícitamente por el usuario.
    - Preferencias inferidas del comportamiento.
    - Almacenados en SQLite, recuperados por relevancia (keyword en V1).
    - Límite: MAX_LONG_TERM_FACTS entradas (LRU cuando se supera).

  Compactación:
    - El LLM genera un resumen de los N turns más antiguos.
    - El resumen reemplaza esos turns en el context window futuro.
    - Los turns compactados se marcan en la Session pero no se eliminan de SQLite.

Diseño:
  - El MemoryManager coordina con SessionManager (aplicar compactación)
    y con UserLLMPort (generar el resumen de compactación).
  - Es stateless entre llamadas — el estado vive en Session y SQLite.
  - La compactación es progresiva: solo compacta los turns necesarios,
    no toda la conversación de una vez.
  - Los hechos de largo plazo se recuperan por keyword matching en V1.
    En V2 se añade búsqueda por embeddings.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from forge_core.errors.types import ForgeLLMError, ForgeStorageError
from forge_core.observability.logging import get_logger

from src.application.context.context_builder import ContextBuilder
from src.application.session.session_manager import SessionManager
from src.domain.entities.session import Session
from src.domain.value_objects.identifiers import SessionId, TurnId, UserId
from src.domain.value_objects.token_budget import TokenEstimator
from src.ports.outbound.llm_port import UserLLMPort
from src.ports.outbound.storage_port import SessionStorePort


logger = get_logger(__name__, component="memory_manager")


# Máximo de hechos de largo plazo por usuario
_MAX_LONG_TERM_FACTS = 500

# Máximo de chars del resumen de compactación
_MAX_COMPACTION_SUMMARY_CHARS = 2_500

# Máximo de hechos a recuperar por turno para el contexto
_MAX_FACTS_PER_TURN = 5


class LongTermFact:
    """
    Hecho persistido en la memoria a largo plazo del usuario.

    Los hechos son observaciones del agente sobre el usuario que
    se guardan entre sesiones para personalizar la experiencia.

    Ejemplos:
    - "El usuario estudia Biología en la universidad"
    - "El usuario prefiere resúmenes con ejemplos concretos"
    - "El profesor de Matemáticas se llama Dr. García"
    - "El usuario trabaja en proyectos de Python"
    """

    __slots__ = (
        "fact_id",
        "content",
        "category",
        "source_session_id",
        "created_at",
        "last_accessed_at",
        "access_count",
        "tags",
        "confidence",
    )

    def __init__(
        self,
        fact_id: str,
        content: str,
        *,
        category: str = "general",
        source_session_id: str = "",
        tags: list[str] | None = None,
        confidence: float = 1.0,
    ) -> None:
        """
        Args:
            fact_id:           ID único del hecho (ULID).
            content:           Descripción del hecho en lenguaje natural.
            category:          Categoría del hecho ('preference', 'fact', 'context').
            source_session_id: ID de la sesión donde se aprendió el hecho.
            tags:              Tags para búsqueda y clasificación.
            confidence:        Confianza en el hecho [0.0, 1.0].
        """
        self.fact_id = fact_id
        self.content = content
        self.category = category
        self.source_session_id = source_session_id
        self.created_at = datetime.now(tz=timezone.utc)
        self.last_accessed_at = self.created_at
        self.access_count = 0
        self.tags = tags or []
        self.confidence = confidence

    def touch(self) -> None:
        """Actualiza last_accessed_at y incrementa access_count."""
        self.last_accessed_at = datetime.now(tz=timezone.utc)
        self.access_count += 1

    def matches_query(self, query: str) -> bool:
        """
        Verifica si el hecho es relevante para una query (keyword matching).

        En V1, usa búsqueda por palabras clave simples.
        En V2, se reemplazará con búsqueda por embeddings.

        Args:
            query: Texto a buscar en el hecho.

        Returns:
            True si el hecho contiene palabras clave de la query.
        """
        query_lower = query.lower()
        content_lower = self.content.lower()

        # Matching directo en el contenido
        if any(word in content_lower for word in query_lower.split() if len(word) > 3):
            return True

        # Matching en tags
        if any(tag.lower() in query_lower for tag in self.tags):
            return True

        return False

    def to_dict(self) -> dict[str, Any]:
        """Serializa el hecho para storage."""
        return {
            "fact_id": self.fact_id,
            "content": self.content,
            "category": self.category,
            "source_session_id": self.source_session_id,
            "created_at": self.created_at.isoformat(),
            "last_accessed_at": self.last_accessed_at.isoformat(),
            "access_count": self.access_count,
            "tags": self.tags,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LongTermFact:
        """Reconstruye un hecho desde el storage."""
        fact = cls(
            fact_id=data["fact_id"],
            content=data["content"],
            category=data.get("category", "general"),
            source_session_id=data.get("source_session_id", ""),
            tags=data.get("tags", []),
            confidence=data.get("confidence", 1.0),
        )
        if "created_at" in data:
            fact.created_at = datetime.fromisoformat(data["created_at"])
        if "last_accessed_at" in data:
            fact.last_accessed_at = datetime.fromisoformat(data["last_accessed_at"])
        fact.access_count = data.get("access_count", 0)
        return fact


class LongTermMemoryStore:
    """
    Store en memoria de los hechos de largo plazo del usuario.

    En V1 se almacena en SQLite vía la tabla de memoria del schema.
    En V2 se añade búsqueda por embeddings con un vector store local.

    Este store es simplificado en V1: carga todos los hechos del usuario
    al inicializar y los mantiene en memoria durante la sesión.
    El LRU (Least Recently Used) se aplica al superar el límite.
    """

    def __init__(self, user_id: UserId, max_facts: int = _MAX_LONG_TERM_FACTS) -> None:
        self._user_id = user_id
        self._max_facts = max_facts
        self._facts: dict[str, LongTermFact] = {}
        self._loaded = False

    def add_fact(self, fact: LongTermFact) -> None:
        """
        Añade un hecho a la memoria de largo plazo.

        Si el store está lleno, elimina el hecho menos usado recientemente (LRU).

        Args:
            fact: Hecho a añadir.
        """
        # Verificar si ya existe un hecho muy similar (deduplicación básica)
        for existing in self._facts.values():
            if self._is_duplicate(fact.content, existing.content):
                # Actualizar el existing con la nueva información
                existing.touch()
                return

        # LRU: eliminar el menos usado si está lleno
        if len(self._facts) >= self._max_facts:
            lru_id = min(
                self._facts.keys(),
                key=lambda fid: (
                    self._facts[fid].last_accessed_at,
                    self._facts[fid].access_count,
                ),
            )
            del self._facts[lru_id]

        self._facts[fact.fact_id] = fact

    def search(self, query: str, max_results: int = _MAX_FACTS_PER_TURN) -> list[LongTermFact]:
        """
        Busca hechos relevantes para una query.

        Implementación V1: keyword matching simple.
        Devuelve los hechos más relevantes ordenados por acceso reciente.

        Args:
            query:       Texto para buscar hechos relevantes.
            max_results: Máximo de hechos a retornar.

        Returns:
            Lista de hechos relevantes, ordenados por último acceso.
        """
        matching = [
            fact for fact in self._facts.values()
            if fact.matches_query(query)
        ]

        # Ordenar por relevancia: más accedidos recientemente primero
        matching.sort(
            key=lambda f: (f.last_accessed_at, f.access_count),
            reverse=True,
        )

        results = matching[:max_results]
        for fact in results:
            fact.touch()

        return results

    def get_all(self) -> list[LongTermFact]:
        """Retorna todos los hechos almacenados."""
        return list(self._facts.values())

    def remove(self, fact_id: str) -> bool:
        """Elimina un hecho por su ID."""
        if fact_id in self._facts:
            del self._facts[fact_id]
            return True
        return False

    def load_from_dicts(self, facts_data: list[dict[str, Any]]) -> None:
        """Carga hechos desde datos serializados (desde SQLite)."""
        for data in facts_data:
            try:
                fact = LongTermFact.from_dict(data)
                self._facts[fact.fact_id] = fact
            except Exception as exc:
                logger.warning("hecho_largo_plazo_carga_fallo", error=str(exc))
        self._loaded = True

    def to_dicts(self) -> list[dict[str, Any]]:
        """Serializa todos los hechos para storage."""
        return [fact.to_dict() for fact in self._facts.values()]

    @staticmethod
    def _is_duplicate(content1: str, content2: str) -> bool:
        """
        Verifica si dos contenidos son suficientemente similares para
        considerarlos duplicados (deduplicación básica por palabras clave).
        """
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        # Similaridad Jaccard simple
        if not words1 or not words2:
            return False
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return (intersection / union) > 0.75


class MemoryManager:
    """
    Gestor de memoria del agente HiperForge User.

    Coordina la memoria a corto plazo (historial de la sesión) y a largo
    plazo (hechos persistidos entre sesiones). También gestiona la
    compactación del historial cuando el context window se agota.

    El MemoryManager es stateful en cuanto a los hechos de largo plazo
    (mantiene un LongTermMemoryStore por usuario en memoria), pero
    stateless para las operaciones de sesión.
    """

    def __init__(
        self,
        llm_port: UserLLMPort,
        session_manager: SessionManager,
        session_store: SessionStorePort,
        *,
        memory_store: Any = None,
        profile_store: Any = None,
        long_term_memory_enabled: bool = True,
        max_long_term_facts: int = _MAX_LONG_TERM_FACTS,
    ) -> None:
        """
        Args:
            llm_port:                 Puerto LLM para generar resúmenes.
            session_manager:          SessionManager para persistir compactación.
            session_store:            Store de sesiones para cargar historial.
            memory_store:             Store SQLite para hechos de largo plazo.
            profile_store:            Store de perfiles (no usado directamente aquí).
            long_term_memory_enabled: Habilitar memoria a largo plazo.
            max_long_term_facts:      Máximo de hechos a largo plazo.
        """
        self._llm = llm_port
        self._session_manager = session_manager
        self._session_store = session_store
        self._memory_store = memory_store
        self._long_term_enabled = long_term_memory_enabled
        self._max_facts = max_long_term_facts

        # Caché en memoria de stores de largo plazo por usuario
        self._user_stores: dict[str, LongTermMemoryStore] = {}

    # =========================================================================
    # COMPACTACIÓN DEL HISTORIAL
    # =========================================================================

    async def compact_if_needed(
        self,
        session: Session,
    ) -> bool:
        """
        Compacta el historial de la sesión si supera el umbral configurado.

        Verifica el estado del budget de tokens y, si el historial excede
        el threshold, genera un resumen de los turns más antiguos y los
        reemplaza en el contexto.

        Args:
            session: Session activa a verificar.

        Returns:
            True si se realizó compactación, False si no fue necesaria.
        """
        compaction_decision = session._token_budget.should_compact(
            current_history_tokens=session.current_history_tokens,
            total_turns=session.turn_count,
        )

        if not compaction_decision.needs_compaction:
            return False

        logger.info(
            "compactacion_iniciando",
            session_id=session.session_id.to_str(),
            turns_to_compact=compaction_decision.turns_to_compact,
            urgency=round(compaction_decision.urgency, 2),
            tokens_to_free=compaction_decision.tokens_to_free,
        )

        try:
            await self._perform_compaction(
                session=session,
                turns_to_compact=compaction_decision.turns_to_compact,
            )
            return True

        except Exception as exc:
            logger.error(
                "compactacion_fallo",
                session_id=session.session_id.to_str(),
                error=exc,
            )
            return False

    async def force_compact(
        self,
        session: Session,
        *,
        turns_count: int | None = None,
    ) -> bool:
        """
        Fuerza la compactación del historial independientemente del threshold.

        Útil cuando el ConversationCoordinator necesita liberar espacio
        urgentemente antes de una llamada al LLM.

        Args:
            session:     Session a compactar.
            turns_count: Número de turns a compactar.
                         None usa la decisión automática del budget.

        Returns:
            True si la compactación fue exitosa.
        """
        if turns_count is None:
            decision = session._token_budget.should_compact(
                current_history_tokens=session.current_history_tokens,
                total_turns=session.turn_count,
            )
            turns_count = max(decision.turns_to_compact, 2)

        if session.turn_count < 2:
            return False

        turns_count = min(turns_count, session.turn_count - 1)

        try:
            await self._perform_compaction(session, turns_to_compact=turns_count)
            return True
        except Exception as exc:
            logger.error("compactacion_forzada_fallo", error=exc)
            return False

    async def _perform_compaction(
        self,
        session: Session,
        *,
        turns_to_compact: int,
    ) -> None:
        """
        Ejecuta la compactación: carga turns, genera resumen con el LLM,
        y aplica el resultado al aggregate Session.

        Args:
            session:          Session activa.
            turns_to_compact: Número de turns más antiguos a compactar.

        Raises:
            ForgeLLMError:   Si el LLM no puede generar el resumen.
            ForgeStorageError: Si no se puede persistir el resultado.
        """
        # 1. Cargar los turns más antiguos para compactar
        turns_data = await self._session_manager.load_all_turns_for_compaction(
            session,
            turns_count=turns_to_compact,
        )

        if not turns_data:
            logger.warning(
                "compactacion_sin_turns",
                session_id=session.session_id.to_str(),
            )
            return

        # 2. Construir el CompactionContext
        from src.application.context.context_builder import ContextBuilder
        compaction_context = await _build_compaction_context_from_turns(
            session=session,
            turns_data=turns_data,
            max_summary_tokens=session._token_budget.compaction_summary_tokens,
        )

        # 3. Llamar al LLM para generar el resumen
        try:
            summary_text = await self._llm.generate_compaction_summary(
                compaction_context
            )
        except ForgeLLMError as e:
            logger.error(
                "compactacion_llm_fallo",
                session_id=session.session_id.to_str(),
                error=e,
            )
            # Fallback: generar un resumen básico sin LLM
            summary_text = self._generate_fallback_summary(turns_data)

        # 4. Truncar el resumen si es demasiado largo
        if len(summary_text) > _MAX_COMPACTION_SUMMARY_CHARS:
            summary_text = summary_text[:_MAX_COMPACTION_SUMMARY_CHARS] + "..."

        # 5. Extraer los IDs de turns compactados
        compacted_turn_ids = [
            TurnId.from_str(t["turn_id"])
            for t in turns_data
            if "turn_id" in t
        ]

        # 6. Estimar tokens liberados
        freed_tokens = sum(
            TokenEstimator.estimate_message(
                "user",
                t.get("user_message", "") + " " + t.get("assistant_response", ""),
            )
            for t in turns_data
        )
        summary_tokens = TokenEstimator.estimate(summary_text)
        net_freed = max(0, freed_tokens - summary_tokens)

        # 7. Aplicar la compactación al aggregate y persistir
        await self._session_manager.apply_compaction(
            session,
            compaction_summary=summary_text,
            compacted_turn_ids=compacted_turn_ids,
            freed_tokens=net_freed,
        )

        logger.info(
            "compactacion_completada",
            session_id=session.session_id.to_str(),
            turns_compacted=len(compacted_turn_ids),
            tokens_freed=net_freed,
            summary_length=len(summary_text),
        )

    @staticmethod
    def _generate_fallback_summary(turns_data: list[dict[str, Any]]) -> str:
        """
        Genera un resumen básico sin LLM como fallback de compactación.

        Se usa cuando el LLM no está disponible pero la sesión necesita
        compactar para continuar.

        Args:
            turns_data: Datos de turns a resumir.

        Returns:
            Resumen básico basado en los mensajes del usuario.
        """
        if not turns_data:
            return "Conversación anterior sin detalles disponibles."

        user_messages = [
            t.get("user_message", "")
            for t in turns_data
            if t.get("user_message")
        ]

        if not user_messages:
            return f"Sección anterior de {len(turns_data)} turnos sin detalles disponibles."

        summary_parts = [
            f"[Resumen automático de {len(turns_data)} turnos anteriores]"
        ]

        # Incluir primeros y últimos mensajes del usuario como preview
        preview_messages = user_messages[:2] + (user_messages[-1:] if len(user_messages) > 2 else [])
        for msg in preview_messages:
            truncated = msg[:200] + ("..." if len(msg) > 200 else "")
            summary_parts.append(f"• {truncated}")

        return "\n".join(summary_parts)

    # =========================================================================
    # MEMORIA A LARGO PLAZO
    # =========================================================================

    async def get_relevant_context(
        self,
        user_id: UserId,
        query: str,
    ) -> list[LongTermFact]:
        """
        Recupera hechos de largo plazo relevantes para la query actual.

        Se llama al inicio de cada turno para enriquecer el contexto del LLM
        con información persistida de sesiones anteriores.

        Args:
            user_id: ID del usuario.
            query:   Texto del mensaje actual para buscar hechos relevantes.

        Returns:
            Lista de hechos relevantes (máximo _MAX_FACTS_PER_TURN).
        """
        if not self._long_term_enabled:
            return []

        store = await self._get_user_store(user_id)
        if not store:
            return []

        return store.search(query, max_results=_MAX_FACTS_PER_TURN)

    async def store_fact(
        self,
        user_id: UserId,
        content: str,
        *,
        category: str = "fact",
        session_id: str = "",
        tags: list[str] | None = None,
        confidence: float = 1.0,
    ) -> str:
        """
        Almacena un hecho en la memoria de largo plazo del usuario.

        El agente llama a este método cuando detecta información relevante
        para persistir (hechos explícitos mencionados por el usuario,
        preferencias, contexto de estudio, etc.).

        Args:
            user_id:    ID del usuario.
            content:    Descripción del hecho en lenguaje natural.
            category:   'fact', 'preference', 'context'.
            session_id: ID de la sesión fuente.
            tags:       Tags para categorización y búsqueda.
            confidence: Confianza en la veracidad del hecho [0.0, 1.0].

        Returns:
            ID del hecho almacenado.
        """
        if not self._long_term_enabled:
            return ""

        from src.domain.value_objects.identifiers import MemoryEntryId
        fact_id = MemoryEntryId.generate().to_str()

        fact = LongTermFact(
            fact_id=fact_id,
            content=content,
            category=category,
            source_session_id=session_id,
            tags=tags,
            confidence=confidence,
        )

        store = await self._get_user_store(user_id)
        if store:
            store.add_fact(fact)
            await self._persist_user_store(user_id, store)

        logger.debug(
            "hecho_almacenado",
            user_id=user_id.to_str(),
            category=category,
            content_preview=content[:80],
        )

        return fact_id

    async def remove_fact(
        self,
        user_id: UserId,
        fact_id: str,
    ) -> bool:
        """
        Elimina un hecho de la memoria de largo plazo.

        Args:
            user_id: ID del usuario.
            fact_id: ID del hecho a eliminar.

        Returns:
            True si existía y fue eliminado.
        """
        store = await self._get_user_store(user_id)
        if not store:
            return False

        removed = store.remove(fact_id)
        if removed:
            await self._persist_user_store(user_id, store)

        return removed

    async def get_all_facts(
        self,
        user_id: UserId,
    ) -> list[LongTermFact]:
        """
        Retorna todos los hechos de largo plazo del usuario.

        Usado por el debug CLI y para exportación del historial de memoria.

        Args:
            user_id: ID del usuario.

        Returns:
            Lista de todos los hechos almacenados.
        """
        store = await self._get_user_store(user_id)
        if not store:
            return []
        return store.get_all()

    async def extract_facts_from_turn(
        self,
        user_id: UserId,
        session_id: str,
        user_message: str,
        assistant_response: str,
    ) -> list[str]:
        """
        Extrae hechos potencialmente persistibles de un turno de conversación.

        Analiza el turno buscando patrones que indican información relevante
        para recordar: nombres propios, preferencias, contexto de estudio, etc.

        En V1, usa patrones de texto simples.
        En V2, usa el LLM para extracción más sofisticada.

        Args:
            user_id:            ID del usuario.
            session_id:         ID de la sesión.
            user_message:       Mensaje del usuario.
            assistant_response: Respuesta del asistente.

        Returns:
            Lista de IDs de hechos almacenados (vacía si no se extrajo nada).
        """
        if not self._long_term_enabled:
            return []

        extracted_fact_ids: list[str] = []

        # Patrones simples de extracción de hechos
        facts_to_store = self._extract_fact_patterns(
            user_message=user_message,
            session_id=session_id,
        )

        for fact_content, category, tags in facts_to_store:
            fact_id = await self.store_fact(
                user_id=user_id,
                content=fact_content,
                category=category,
                session_id=session_id,
                tags=tags,
            )
            if fact_id:
                extracted_fact_ids.append(fact_id)

        return extracted_fact_ids

    def _extract_fact_patterns(
        self,
        user_message: str,
        session_id: str,
    ) -> list[tuple[str, str, list[str]]]:
        """
        Extrae hechos usando patrones de texto simples.

        Detecta frases que indican información importante:
        - "Me llamo X", "Soy X" → nombre/identidad
        - "Mi profesor se llama X" → contexto académico
        - "Estudio X" → área de estudio
        - "Prefiero X", "Me gusta X" → preferencias

        Args:
            user_message: Mensaje del usuario a analizar.
            session_id:   ID de la sesión.

        Returns:
            Lista de tuplas (content, category, tags).
        """
        facts: list[tuple[str, str, list[str]]] = []
        msg_lower = user_message.lower()

        # Patrones de identidad
        identity_patterns = [
            ("me llamo ", "name"),
            ("mi nombre es ", "name"),
            ("soy ", "identity"),
        ]
        for pattern, tag in identity_patterns:
            if pattern in msg_lower:
                idx = msg_lower.find(pattern)
                end = min(idx + len(pattern) + 50, len(user_message))
                fragment = user_message[idx:end].split(".")[0].split(",")[0]
                if len(fragment) > len(pattern) + 2:
                    facts.append((
                        f"El usuario menciona: '{fragment.strip()}'",
                        "fact",
                        [tag, "identity"],
                    ))

        # Patrones de estudio
        study_patterns = [
            "estudio ", "estudié ", "mi materia ", "mi profesor",
            "mi clase de ", "en la universidad", "en el colegio",
        ]
        for pattern in study_patterns:
            if pattern in msg_lower:
                # Extraer el fragmento relevante
                idx = msg_lower.find(pattern)
                fragment = user_message[idx:idx + 100].split(".")[0]
                if len(fragment) > len(pattern) + 3:
                    facts.append((
                        f"Contexto de estudio: '{fragment.strip()[:100]}'",
                        "context",
                        ["study", "academic"],
                    ))
                break  # un hecho por turno de este tipo

        # Patrones de preferencia
        preference_patterns = [
            "prefiero ", "me gusta más ", "no me gusta ",
            "siempre quiero ", "necesito que ",
        ]
        for pattern in preference_patterns:
            if pattern in msg_lower:
                idx = msg_lower.find(pattern)
                fragment = user_message[idx:idx + 80].split(".")[0]
                if len(fragment) > len(pattern) + 5:
                    facts.append((
                        f"Preferencia del usuario: '{fragment.strip()[:80]}'",
                        "preference",
                        ["preference"],
                    ))
                break

        return facts

    async def build_long_term_context_string(
        self,
        user_id: UserId,
        current_query: str,
    ) -> str:
        """
        Construye el string de contexto de largo plazo para el system prompt.

        Recupera los hechos más relevantes y los formatea como texto
        para incluir en el contexto del LLM.

        Args:
            user_id:       ID del usuario.
            current_query: Query actual para selección de hechos relevantes.

        Returns:
            String con los hechos formateados, vacío si no hay hechos.
        """
        if not self._long_term_enabled:
            return ""

        relevant_facts = await self.get_relevant_context(user_id, current_query)

        if not relevant_facts:
            return ""

        facts_text = "\n".join(f"• {fact.content}" for fact in relevant_facts)
        return f"[Información recordada sobre el usuario]\n{facts_text}"

    async def clear_long_term_memory(
        self,
        user_id: UserId,
    ) -> int:
        """
        Elimina toda la memoria de largo plazo del usuario.

        Solo se llama cuando el usuario explícitamente solicita
        borrar su historial de memoria.

        Args:
            user_id: ID del usuario.

        Returns:
            Número de hechos eliminados.
        """
        store = await self._get_user_store(user_id)
        if not store:
            return 0

        facts_count = len(store.get_all())
        user_id_str = user_id.to_str()

        # Limpiar el store en memoria
        self._user_stores.pop(user_id_str, None)

        # Persistir el estado vacío
        await self._persist_user_store(user_id, LongTermMemoryStore(user_id))

        logger.info(
            "memoria_largo_plazo_limpiada",
            user_id=user_id_str,
            facts_removed=facts_count,
        )

        return facts_count

    # =========================================================================
    # HELPERS DE ALMACENAMIENTO DE LARGO PLAZO
    # =========================================================================

    async def _get_user_store(
        self,
        user_id: UserId,
    ) -> LongTermMemoryStore | None:
        """
        Obtiene o carga el store de largo plazo del usuario desde SQLite.

        Se mantiene en caché en memoria durante la vida del MemoryManager.

        Args:
            user_id: ID del usuario.

        Returns:
            LongTermMemoryStore del usuario, o None si hay error de storage.
        """
        user_id_str = user_id.to_str()

        if user_id_str in self._user_stores:
            return self._user_stores[user_id_str]

        store = LongTermMemoryStore(
            user_id=user_id,
            max_facts=self._max_facts,
        )

        # Cargar hechos persistidos desde SQLite
        if self._memory_store is not None:
            try:
                facts_data = await self._memory_store.load_facts(user_id_str)
                store.load_from_dicts(facts_data)
            except Exception as exc:
                logger.error("memoria_carga_fallo", user_id=user_id_str, error=str(exc))

        self._user_stores[user_id_str] = store
        return store

    async def _persist_user_store(
        self,
        user_id: UserId,
        store: LongTermMemoryStore,
    ) -> None:
        """
        Persiste el store de largo plazo del usuario en SQLite.

        Args:
            user_id: ID del usuario.
            store:   Store a persistir.
        """
        if self._memory_store is None:
            return

        try:
            await self._memory_store.save_facts(user_id.to_str(), store.to_dicts())
            logger.debug(
                "memoria_largo_plazo_persistida",
                user_id=user_id.to_str(),
                facts_count=len(store.get_all()),
            )
        except Exception as exc:
            logger.error(
                "memoria_persistencia_fallo",
                user_id=user_id.to_str(),
                error=str(exc),
            )

    # =========================================================================
    # MÉTRICAS Y DIAGNÓSTICO
    # =========================================================================

    async def get_memory_stats(
        self,
        user_id: UserId,
        session: Session | None = None,
    ) -> dict[str, Any]:
        """
        Retorna estadísticas de memoria para diagnóstico y el debug CLI.

        Args:
            user_id: ID del usuario.
            session: Session activa (opcional, para stats de corto plazo).

        Returns:
            Dict con estadísticas completas de memoria.
        """
        stats: dict[str, Any] = {
            "long_term_enabled": self._long_term_enabled,
        }

        # Stats de largo plazo
        store = self._user_stores.get(user_id.to_str())
        if store:
            all_facts = store.get_all()
            stats["long_term"] = {
                "total_facts": len(all_facts),
                "max_facts": self._max_facts,
                "categories": {},
            }
            for fact in all_facts:
                cat = fact.category
                stats["long_term"]["categories"][cat] = (
                    stats["long_term"]["categories"].get(cat, 0) + 1
                )
        else:
            stats["long_term"] = {"total_facts": 0, "loaded": False}

        # Stats de corto plazo
        if session:
            allocation = session._token_budget.compute_allocation()
            snapshot = session.get_token_usage_snapshot()
            compaction_decision = session._token_budget.should_compact(
                current_history_tokens=session.current_history_tokens,
                total_turns=session.turn_count,
            )
            stats["short_term"] = {
                "total_turns": session.turn_count,
                "active_turns": snapshot.turns_included,
                "compacted_turns": session.compacted_turns_count,
                "has_compaction_summary": session.compaction_summary is not None,
                "current_history_tokens": session.current_history_tokens,
                "available_history_tokens": allocation.available_for_history,
                "usage_ratio": round(
                    allocation.usage_ratio(session.current_history_tokens), 3
                ),
                "needs_compaction": compaction_decision.needs_compaction,
                "compaction_urgency": round(compaction_decision.urgency, 3),
            }

        return stats


# =============================================================================
# HELPERS FUERA DE LA CLASE
# =============================================================================


async def _build_compaction_context_from_turns(
    session: Session,
    turns_data: list[dict[str, Any]],
    max_summary_tokens: int,
) -> Any:
    """
    Construye un CompactionContext desde datos de turns crudos.

    Función auxiliar que evita la dependencia circular entre MemoryManager
    y ContextBuilder — construye el contexto de compactación directamente.

    Args:
        session:             Session activa.
        turns_data:          Datos de turns a compactar.
        max_summary_tokens:  Tokens máximos del resumen.

    Returns:
        CompactionContext para el UserLLMPort.
    """
    from forge_core.llm.protocol import LLMMessage
    from src.ports.outbound.llm_port import CompactionContext

    # Construir mensajes del historial a compactar
    turns_messages: list[LLMMessage] = []
    for turn in turns_data:
        user_msg = turn.get("user_message", "")
        assistant_msg = turn.get("assistant_response", "")
        if user_msg:
            turns_messages.append(LLMMessage.user(user_msg))
        if assistant_msg:
            turns_messages.append(LLMMessage.assistant(assistant_msg))

    return CompactionContext(
        session_id=session.session_id,
        turns_to_compact=turns_messages,
        max_summary_tokens=max_summary_tokens,
        language="es",
    )