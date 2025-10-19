from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.search_agent import SearchAgent
from app.agents.memory_agent import MemoryAgent
from app.agents.answer_agent import AnswerAgent
from app.core.logging import app_logger as logger


class QAOrchestrator:
    

    def __init__(self, db_session: AsyncSession):
        
        self.search_agent = SearchAgent()
        self.memory_agent = MemoryAgent(db_session)
        self.answer_agent = AnswerAgent()
        logger.info("QAOrchestrator initialized with 3 agents")

    async def process_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        include_sources: bool = True,
        top_k: int = 5
    ) -> Dict[str, Any]:
        
        logger.info(f"Processing query: '{query[:100]}...'")

        try:
            # Step 1: Memory Agent - Get or create session
            session_id = await self.memory_agent.get_or_create_session(
                session_id=session_id,
                user_id=user_id
            )

            logger.info(f"Using session: {session_id}")

            # Step 2: Memory Agent - Retrieve conversation history
            memory_result = await self.memory_agent.execute(
                session_id=session_id,
                action="retrieve"
            )

            conversation_history = memory_result.get("history", [])
            logger.info(f"Retrieved {len(conversation_history)} previous messages")

            # Step 3: Memory Agent - Check if follow-up question
            followup_result = await self.memory_agent.execute(
                session_id=session_id,
                action="check_followup",
                query=query
            )

            is_followup = followup_result.get("is_followup", False)
            logger.info(f"Is follow-up question: {is_followup}")

            # Step 4: Search Agent - Perform hybrid search
            search_result = await self.search_agent.execute(
                query=query,
                top_k=top_k
            )

            if not search_result.get("success"):
                return {
                    "success": False,
                    "error": "Search failed",
                    "details": search_result.get("error")
                }

            search_results = search_result.get("results", [])
            logger.info(f"Search returned {len(search_results)} results")

            # Step 5: Answer Agent - Generate answer
            answer_result = await self.answer_agent.execute(
                query=query,
                search_results=search_results,
                conversation_history=conversation_history if is_followup else None
            )

            if not answer_result.get("success"):
                return {
                    "success": False,
                    "error": "Answer generation failed",
                    "details": answer_result.get("error")
                }

            # Step 6: Memory Agent - Store user query
            await self.memory_agent.execute(
                session_id=session_id,
                action="store",
                role="user",
                content=query
            )

            # Step 7: Memory Agent - Store assistant response
            await self.memory_agent.execute(
                session_id=session_id,
                action="store",
                role="assistant",
                content=answer_result["answer"],
                tokens_used=answer_result.get("tokens_used"),
                sources=answer_result.get("sources"),
                confidence_score=answer_result.get("confidence_score")
            )

            logger.info("Q&A processing completed successfully")

            # Build final response
            response = {
                "success": True,
                "answer": answer_result["answer"],
                "session_id": session_id,
                "tokens_used": answer_result.get("tokens_used"),
                "confidence_score": answer_result.get("confidence_score"),
                "is_followup": is_followup,
                "search_results_count": len(search_results)
            }

            if include_sources:
                response["sources"] = answer_result.get("sources", [])

            return response

        except Exception as e:
            logger.error(f"Orchestrator failed: {str(e)}")
            return {
                "success": False,
                "error": "Processing failed",
                "details": str(e)
            }

    async def get_conversation_history(
        self,
        session_id: str,
        limit: int = 20
    ) -> Dict[str, Any]:
        
        result = await self.memory_agent.execute(
            session_id=session_id,
            action="retrieve",
            limit=limit
        )

        return result

    async def clear_conversation(self, session_id: str) -> bool:
        
        try:
            await self.memory_agent.memory_service.clear_conversation(session_id)
            logger.info(f"Cleared conversation: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear conversation: {str(e)}")
            return False
