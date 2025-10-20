from typing import Dict, Any, List
from app.agents.base_agent import BaseAgent
from app.services.qdrant_service import qdrant_service
from app.services.reranker import reranker_service
from app.core.config import settings
import time


class SearchAgent(BaseAgent):
    """
    Search agent using Qdrant for hybrid search (BM25 + Vector).

    Qdrant handles server-side fusion of keyword and semantic search,
    eliminating the need for client-side reranking.
    """

    def __init__(self):
        super().__init__(name="SearchAgent")

    async def execute(
        self,
        query: str,
        top_k: int = None,
        filter_by: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute hybrid search using Qdrant.

        Flow:
        1. Call Qdrant hybrid search (BM25 + vector with RRF fusion)
        2. Results are already ranked by Qdrant server-side
        3. Format and return results

        Args:
            query: Search query
            top_k: Number of results to return
            filter_by: Filter expression (e.g., "document_id:=abc123")

        Returns:
            Search results with scores
        """
        self.log_execution("Executing hybrid search", f"query='{query[:50]}...'")

        top_k = top_k or settings.TOP_K_SEARCH_RESULTS

        try:
            search_start = time.time()

            # Fetch more results for reranking (2x top_k)
            # Reranker will select the best top_k from these
            fetch_limit = top_k * 2

            # Execute hybrid search (Qdrant handles BM25 + vector fusion server-side)
            search_results = qdrant_service.search(
                query=query,
                limit=fetch_limit,
                filter_by=filter_by,
                hybrid_search=True  # Enable hybrid search (BM25 + vector)
            )

            search_time_ms = (time.time() - search_start) * 1000

            # Extract hits
            hits = search_results.get("hits", [])

            if not hits:
                self.log_execution("No results found", f"query='{query}'")
                return {
                    "success": True,
                    "results": [],
                    "count": 0,
                    "message": "No relevant documents found"
                }

            # Format results
            formatted_results = []
            for hit in hits:
                payload = hit.get("payload", {})
                formatted_results.append({
                    "document": {
                        "id": hit.get("id"),
                        "text": payload.get("text"),
                        "document_id": payload.get("document_id"),
                        "filename": payload.get("filename"),
                        "file_type": payload.get("file_type"),
                        "chunk_index": payload.get("chunk_index"),
                        "char_count": payload.get("char_count"),
                        "word_count": payload.get("word_count"),
                    },
                    "score": hit.get("score", 0),  # Combined hybrid score from Qdrant
                    "hybrid_search_info": {
                        "search_type": "qdrant_hybrid",
                        "fusion_method": "RRF",
                        "vector_weight": settings.HYBRID_SEARCH_WEIGHT_VECTOR,
                        "keyword_weight": settings.HYBRID_SEARCH_WEIGHT_KEYWORD
                    }
                })

            # Apply cross-encoder reranking for better accuracy
            self.log_execution("Cross-encoder reranking", f"reranking {len(formatted_results)} results")
            rerank_start = time.time()

            reranked_results = reranker_service.rerank(
                query=query,
                results=formatted_results,
                top_k=top_k
            )

            rerank_time_ms = (time.time() - rerank_start) * 1000
            total_time_ms = search_time_ms + rerank_time_ms

            self.log_execution(
                "Reranking complete",
                f"took {rerank_time_ms:.0f}ms, top score: {reranked_results[0].get('rerank_score', 0):.3f}"
            )

            # Use reranked results
            formatted_results = reranked_results

            self.log_execution(
                "Search completed",
                f"Found {len(formatted_results)} results in {total_time_ms:.0f}ms (search: {search_time_ms:.0f}ms + rerank: {rerank_time_ms:.0f}ms)"
            )

            return {
                "success": True,
                "results": formatted_results,
                "count": len(formatted_results),
                "query": query,
                "search_time_ms": total_time_ms,
                "rerank_time_ms": rerank_time_ms
            }

        except Exception as e:
            self.log_error(f"Search failed: {str(e)}")
            return {
                "success": False,
                "results": [],
                "count": 0,
                "error": str(e)
            }

    def expand_query(self, query: str) -> List[str]:
        """
        Query expansion (currently disabled, can be implemented later).

        Potential expansions:
        - Synonyms
        - Stemming variations
        - Entity extraction
        """
        queries = [query]
        return queries
