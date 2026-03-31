"""
Entidad Conversation - Representa una conversación individual.
Una conversación encapsula un turno completo de interacción:
- Mensaje del usuario → Respuesta del agente
- Clasificación de intención
- Decisión de routing
- Artifacts involucrados
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from src.domain.value_objects.identifiers import ConversationId, SessionId, UserId
from src.domain.value_objects.intent import IntentClassification
@dataclass(frozen=True)
class Conversation:
    """Una conversación individual: mensaje del usuario → respuesta del agente."""
    conversation_id: ConversationId
    session_id: SessionId
    user_id: UserId
    turn_number: int
    user_message: str
    assistant_response: str
    intent: IntentClassification
    routing_decision: str = "direct_reply"
    artifacts_involved: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    @classmethod
    def create(
        cls,
        session_id: SessionId,
        user_id: UserId,
        turn_number: int,
        user_message: str,
        assistant_response: str,
        intent: IntentClassification,
        **kwargs: Any,
    ) -> Conversation:
        """Factory method para crear una conversación."""
        return cls(
            conversation_id=ConversationId.generate(),
            session_id=session_id,
            user_id=user_id,
            turn_number=turn_number,
            user_message=user_message,
            assistant_response=assistant_response,
            intent=intent,
            **kwargs,
        )
