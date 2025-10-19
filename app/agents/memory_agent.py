from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.agents.base_agent import BaseAgent
from app.services.memory_service import MemoryService
from app.core.config import settings


class MemoryAgent(BaseAgent):

    def __init__(self, db_session: AsyncSession):
        super().__init__(name="MemoryAgent")
        self.memory_service = MemoryService(db_session)

    async def execute(
        self,
        session_id: str,
        action: str = "retrieve",
        **kwargs
    ) -> Dict[str, Any]:
        
        self.log_execution(f"Executing action: {action}", f"session={session_id}")

        try:
            if action == "retrieve":
                return await self._retrieve_history(session_id, **kwargs)
            elif action == "store":
                return await self._store_message(session_id, **kwargs)
            elif action == "check_followup":
                return await self._check_followup(session_id, **kwargs)
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}"
                }

        except Exception as e:
            self.log_error(f"Memory operation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _retrieve_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        
        limit = limit or settings.MAX_CONVERSATION_HISTORY

        self.log_execution("Retrieving conversation history", f"limit={limit}")

        # Get conversation history
        history = await self.memory_service.get_conversation_history(
            session_id=session_id,
            limit=limit
        )

        return {
            "success": True,
            "history": history,
            "count": len(history),
            "session_id": session_id
        }

    async def _store_message(
        self,
        session_id: str,
        role: str,
        content: str,
        **kwargs
    ) -> Dict[str, Any]:
        
        self.log_execution("Storing message", f"role={role}")

        # Ensure conversation exists
        await self.memory_service.get_or_create_conversation(session_id=session_id)

        # Store message
        message = await self.memory_service.add_message(
            session_id=session_id,
            role=role,
            content=content,
            tokens_used=kwargs.get("tokens_used"),
            sources=kwargs.get("sources"),
            confidence_score=kwargs.get("confidence_score")
        )

        return {
            "success": True,
            "message_id": message.id,
            "session_id": session_id
        }

    async def _check_followup(
        self,
        session_id: str,
        query: str
    ) -> Dict[str, Any]:
        
        self.log_execution("Checking if follow-up question", f"query='{query[:30]}...'")

        # Get recent history
        history = await self.memory_service.get_conversation_history(
            session_id=session_id,
            limit=5
        )

        # Check for follow-up indicators
        is_followup = self.memory_service.check_follow_up_question(query, history)

        return {
            "success": True,
            "is_followup": is_followup,
            "history_count": len(history),
            "context": history[-2:] if is_followup and len(history) >= 2 else []
        }

    async def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> str:
        
        conversation = await self.memory_service.get_or_create_conversation(
            session_id=session_id,
            user_id=user_id
        )
        return conversation.id
