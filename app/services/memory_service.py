"""
Memory Service - Conversation history management.
"""
from typing import List, Dict, Optional
from datetime import datetime
import uuid
import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from app.models.document import Conversation, ConversationMessage
from app.core.logging import app_logger as logger
from app.core.config import settings


class MemoryService:
    """
    Manages conversation history and context retrieval.
    """

    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    async def get_or_create_conversation(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Conversation:
        """
        Get existing conversation or create new one.

        Args:
            session_id: Optional session ID
            user_id: Optional user ID

        Returns:
            Conversation object
        """
        if session_id:
            # Try to get existing conversation
            result = await self.db.execute(
                select(Conversation).where(Conversation.id == session_id)
            )
            conversation = result.scalar_one_or_none()

            if conversation:
                logger.info(f"Retrieved existing conversation: {session_id}")
                return conversation

        # Create new conversation
        session_id = session_id or str(uuid.uuid4())

        conversation = Conversation(
            id=session_id,
            user_id=user_id,
            message_count=0
        )

        self.db.add(conversation)
        await self.db.commit()
        await self.db.refresh(conversation)

        logger.info(f"Created new conversation: {session_id}")
        return conversation

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tokens_used: Optional[int] = None,
        sources: Optional[List[Dict]] = None,
        confidence_score: Optional[float] = None
    ) -> ConversationMessage:
        """
        Add a message to conversation history.

        Args:
            session_id: Conversation session ID
            role: 'user' or 'assistant'
            content: Message content
            tokens_used: Token count (for assistant messages)
            sources: Source documents (for assistant messages)
            confidence_score: Confidence score (for assistant messages)

        Returns:
            ConversationMessage object
        """
        logger.info(f"Adding {role} message to session: {session_id}")

        message = ConversationMessage(
            session_id=session_id,
            role=role,
            content=content,
            tokens_used=tokens_used,
            sources=json.dumps(sources) if sources else None,
            confidence_score=confidence_score
        )

        self.db.add(message)

        # Update conversation message count
        result = await self.db.execute(
            select(Conversation).where(Conversation.id == session_id)
        )
        conversation = result.scalar_one_or_none()

        if conversation:
            conversation.message_count += 1
            conversation.updated_at = datetime.utcnow()

        await self.db.commit()
        await self.db.refresh(message)

        logger.info(f"Message added (ID: {message.id})")
        return message

    async def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Get conversation history for a session.

        Args:
            session_id: Conversation session ID
            limit: Maximum number of messages (default: MAX_CONVERSATION_HISTORY)

        Returns:
            List of message dictionaries
        """
        limit = limit or settings.MAX_CONVERSATION_HISTORY

        logger.info(f"Retrieving last {limit} messages for session: {session_id}")

        result = await self.db.execute(
            select(ConversationMessage)
            .where(ConversationMessage.session_id == session_id)
            .order_by(desc(ConversationMessage.timestamp))
            .limit(limit)
        )

        messages = result.scalars().all()

        # Reverse to get chronological order
        messages = list(reversed(messages))

        history = [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp
            }
            for msg in messages
        ]

        logger.info(f"Retrieved {len(history)} messages")
        return history

    async def get_last_n_messages(
        self,
        session_id: str,
        n: int = 3
    ) -> List[Dict]:
        """
        Get last N messages from conversation.

        Args:
            session_id: Conversation session ID
            n: Number of messages to retrieve

        Returns:
            List of last N messages
        """
        return await self.get_conversation_history(session_id, limit=n)

    async def clear_conversation(self, session_id: str) -> bool:
        """
        Clear all messages in a conversation.

        Args:
            session_id: Conversation session ID

        Returns:
            True if successful
        """
        logger.info(f"Clearing conversation: {session_id}")

        try:
            # Delete all messages
            await self.db.execute(
                select(ConversationMessage).where(
                    ConversationMessage.session_id == session_id
                )
            )

            # Update conversation
            result = await self.db.execute(
                select(Conversation).where(Conversation.id == session_id)
            )
            conversation = result.scalar_one_or_none()

            if conversation:
                conversation.message_count = 0

            await self.db.commit()
            logger.info("Conversation cleared")
            return True

        except Exception as e:
            logger.error(f"Failed to clear conversation: {str(e)}")
            return False

    async def list_conversations(
        self,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Conversation]:
        """
        List all conversations, optionally filtered by user.

        Args:
            user_id: Optional user ID filter
            limit: Maximum number of conversations

        Returns:
            List of Conversation objects
        """
        logger.info("Listing conversations")

        query = select(Conversation).order_by(desc(Conversation.updated_at)).limit(limit)

        if user_id:
            query = query.where(Conversation.user_id == user_id)

        result = await self.db.execute(query)
        conversations = result.scalars().all()

        logger.info(f"Found {len(conversations)} conversations")
        return conversations

    def check_follow_up_question(
        self,
        query: str,
        conversation_history: List[Dict]
    ) -> bool:
        """
        Check if query is likely a follow-up question.

        Args:
            query: User query
            conversation_history: Previous messages

        Returns:
            True if likely a follow-up question
        """
        if not conversation_history:
            return False

        # Simple heuristics for follow-up detection
        follow_up_indicators = [
            "it", "that", "this", "they", "them",
            "what about", "how about", "tell me more",
            "and", "also", "more", "further",
            "continue", "elaborate"
        ]

        query_lower = query.lower()

        # Check if query starts with follow-up indicators
        for indicator in follow_up_indicators:
            if query_lower.startswith(indicator) or f" {indicator} " in query_lower:
                logger.info(f"Detected follow-up question (indicator: '{indicator}')")
                return True

        # Check if query is very short (< 5 words) - often follow-ups
        if len(query.split()) < 5 and len(conversation_history) > 0:
            logger.info("Detected potential follow-up (short query with history)")
            return True

        return False


def format_conversation_for_llm(messages: List[Dict]) -> List[Dict]:
    """
    Format conversation history for LLM consumption.

    Args:
        messages: List of message dictionaries

    Returns:
        Formatted messages for LLM
    """
    return [
        {
            "role": msg["role"],
            "content": msg["content"]
        }
        for msg in messages
    ]
