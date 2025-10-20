from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from pydantic import BaseModel

from app.services.qdrant_service import qdrant_service
from app.core.logging import app_logger as logger


router = APIRouter(prefix="/search", tags=["search"])


class SearchQuery(BaseModel):
    query: str
    limit: int = 10
    document_id: Optional[str] = None
    hybrid: bool = True


class SearchResponse(BaseModel):
    query: str
    found: int
    results: list
    search_time_ms: int


@router.post("/", response_model=SearchResponse)
async def search_documents(search: SearchQuery):
    logger.info(f"Search query: '{search.query}'")

    try:
        # Build filter
        filter_by = None
        if search.document_id:
            filter_by = f"document_id:={search.document_id}"

        # Execute search using Qdrant
        results = qdrant_service.search(
            query=search.query,
            limit=search.limit,
            filter_by=filter_by,
            hybrid_search=search.hybrid
        )

        return SearchResponse(
            query=search.query,
            found=results.get("found", 0),
            results=results.get("hits", []),
            search_time_ms=results.get("search_time_ms", 0)
        )

    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/", response_model=SearchResponse)
async def search_documents_get(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Number of results"),
    document_id: Optional[str] = Query(None, description="Filter by document ID"),
    hybrid: bool = Query(True, description="Use hybrid search")
):
    search_query = SearchQuery(
        query=q,
        limit=limit,
        document_id=document_id,
        hybrid=hybrid
    )
    return await search_documents(search_query)


@router.get("/stats")
async def get_search_stats():
    """Get Qdrant collection statistics."""
    try:
        stats = qdrant_service.get_collection_stats()
        return {
            "status": "success",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))