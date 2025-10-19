from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
from typing import List

from app.models.schemas import (
    QARequest,
    QAResponse,
    SourceDocument,
    ConversationHistoryResponse,
    ConversationHistoryMessage,
    ConversationListResponse
)
from app.agents.orchestrator import QAOrchestrator
from app.core.database import get_db
from app.core.logging import app_logger as logger
from app.models.document import Conversation
from sqlalchemy import select, desc


router = APIRouter(prefix="/qa", tags=["Q&A"])


@router.post("/ask", response_model=QAResponse)
async def ask_question(
    request: QARequest,
    db: AsyncSession = Depends(get_db)
):
    
    logger.info(f"Received Q&A request: '{request.query[:100]}...'")

    try:
        orchestrator = QAOrchestrator(db)

        result = await orchestrator.process_query(
            query=request.query,
            session_id=request.session_id,
            include_sources=request.include_sources,
            top_k=request.top_k or 5
        )

        if not result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"Q&A processing failed: {result.get('details', 'Unknown error')}"
            )

        # Format sources
        sources = None
        if request.include_sources and "sources" in result:
            sources = [
                SourceDocument(**source)
                for source in result["sources"]
            ]

        response = QAResponse(
            answer=result["answer"],
            session_id=result["session_id"],
            sources=sources,
            timestamp=datetime.utcnow(),
            tokens_used=result.get("tokens_used"),
            confidence_score=result.get("confidence_score")
        )

        logger.info(f"Q&A request completed successfully (session: {response.session_id})")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Q&A request failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {str(e)}"
        )


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    
    logger.info(f"Listing conversations (limit={limit})")

    try:
        result = await db.execute(
            select(Conversation)
            .order_by(desc(Conversation.updated_at))
            .limit(limit)
        )
        conversations = result.scalars().all()

        return ConversationListResponse(
            conversations=[conv.to_dict() for conv in conversations],
            total=len(conversations)
        )

    except Exception as e:
        logger.error(f"Failed to list conversations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list conversations: {str(e)}"
        )


@router.get("/conversations/{session_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    session_id: str,
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    
    logger.info(f"Retrieving conversation history: {session_id}")

    try:
        # Get conversation
        result = await db.execute(
            select(Conversation).where(Conversation.id == session_id)
        )
        conversation = result.scalar_one_or_none()

        if not conversation:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation not found: {session_id}"
            )

        # Get conversation history
        orchestrator = QAOrchestrator(db)
        history_result = await orchestrator.get_conversation_history(
            session_id=session_id,
            limit=limit
        )

        if not history_result.get("success"):
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve conversation history"
            )

        # Format messages
        messages = [
            ConversationHistoryMessage(
                role=msg["role"],
                content=msg["content"],
                timestamp=msg["timestamp"],
                sources=None  # TODO: Add sources if needed
            )
            for msg in history_result.get("history", [])
        ]

        return ConversationHistoryResponse(
            session_id=session_id,
            messages=messages,
            message_count=len(messages),
            created_at=conversation.created_at,
            updated_at=conversation.updated_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get conversation history: {str(e)}"
        )


@router.delete("/conversations/{session_id}")
async def clear_conversation(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    
    logger.info(f"Clearing conversation: {session_id}")

    try:
        orchestrator = QAOrchestrator(db)
        success = await orchestrator.clear_conversation(session_id)

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to clear conversation"
            )

        return {
            "success": True,
            "message": f"Conversation {session_id} cleared successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear conversation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear conversation: {str(e)}"
        )


@router.get("/health")
async def qa_health_check():
    
    return {
        "status": "healthy",
        "service": "Q&A API",
        "agents": ["SearchAgent", "MemoryAgent", "AnswerAgent"],
        "llm": "Groq (llama3-8b-8192)"
    }
