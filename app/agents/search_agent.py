from typing import Dict, Any, List
from app.agents.base_agent import BaseAgent
from app.services.typesense_service import typesense_service
from app.services.embeddings import embedding_service
from app.utils.similarity import cosine_similarity, hybrid_score
from app.core.config import settings
import time


class SearchAgent(BaseAgent):

    def __init__(self):
        super().__init__(name="SearchAgent")

    async def execute(
        self,
        query: str,
        top_k: int = None,
        filter_by: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        self.log_execution("Executing hybrid search", f"query='{query[:50]}...'")

        top_k = top_k or settings.TOP_K_SEARCH_RESULTS

        try:
            keyword_limit = max(top_k * 4, 15)  # At least 15 results for reranking

            self.log_execution("Keyword search", f"fetching {keyword_limit} results")

            search_results = typesense_service.search(
                query=query,
                limit=keyword_limit,
                filter_by=filter_by,
                hybrid_search=False  # Fast keyword-only search
            )

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

            # Step 2: Client-side semantic reranking (adds semantic understanding)
            self.log_execution("Semantic reranking", f"reranking {len(hits)} results")
            rerank_start = time.time()

            reranked_hits = self._semantic_rerank(query, hits, top_k)

            rerank_time_ms = (time.time() - rerank_start) * 1000
            self.log_execution("Reranking complete", f"took {rerank_time_ms:.0f}ms")

            # Step 3: Format results
            formatted_results = []
            for hit in reranked_hits:
                doc = hit.get("document", {})
                formatted_results.append({
                    "document": doc,
                    "text_match_score": hit.get("text_match", 0),
                    "semantic_score": hit.get("semantic_score", 0),
                    "hybrid_score": hit.get("hybrid_score", 0),
                    "hybrid_search_info": hit.get("hybrid_search_info", {})
                })

            self.log_execution(
                "Search completed",
                f"Found {len(formatted_results)} results"
            )

            return {
                "success": True,
                "results": formatted_results,
                "count": len(formatted_results),
                "query": query,
                "search_time_ms": search_results.get("search_time_ms", 0)
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
        queries = [query]

        return queries

    def _semantic_rerank(
        self,
        query: str,
        hits: List[Dict],
        top_k: int
    ) -> List[Dict]:
        try:
            # Generate embedding for query
            query_embedding = embedding_service.encode_single(query)

            # Compute semantic scores for each result
            for hit in hits:
                doc = hit.get("document", {})
                doc_embedding = doc.get("embedding", [])

                if doc_embedding and len(doc_embedding) > 0:
                    # Compute cosine similarity
                    semantic_sim = cosine_similarity(query_embedding, doc_embedding)

                    # Normalize to 0-1 range (cosine is -1 to 1)
                    semantic_score = (semantic_sim + 1) / 2

                    # Get keyword score (normalize text_match from Typesense)
                    text_match_score = hit.get("text_match", 0)
                    # Typesense text_match is very large, normalize it
                    keyword_score = min(text_match_score / 1e12, 1.0) if text_match_score > 0 else 0.0

                    # Compute hybrid score (70% semantic, 30% keyword)
                    combined_score = hybrid_score(
                        keyword_score=keyword_score,
                        semantic_score=semantic_score,
                        keyword_weight=settings.HYBRID_SEARCH_WEIGHT_KEYWORD,
                        semantic_weight=settings.HYBRID_SEARCH_WEIGHT_VECTOR
                    )

                    # Add scores to hit
                    hit["semantic_score"] = semantic_score
                    hit["hybrid_score"] = combined_score
                else:
                    # No embedding available, use keyword score only
                    hit["semantic_score"] = 0.0
                    hit["hybrid_score"] = hit.get("text_match", 0) / 1e12

            # Sort by hybrid score (descending)
            reranked_hits = sorted(
                hits,
                key=lambda x: x.get("hybrid_score", 0),
                reverse=True
            )

            # Return top-K
            return reranked_hits[:top_k]

        except Exception as e:
            self.log_error(f"Semantic reranking failed: {str(e)}")
            # Fallback to keyword-only ranking
            return hits[:top_k]

    def rerank_results(
        self,
        results: List[Dict],
        query: str
    ) -> List[Dict]:
        return results
