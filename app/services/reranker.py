"""
Cross-encoder reranking service for improving search result quality.

Cross-encoders provide more accurate relevance scoring than bi-encoders
by processing query-document pairs jointly.
"""
from typing import List, Dict
from sentence_transformers import CrossEncoder
from app.core.logging import app_logger as logger
from app.core.config import settings


class RerankerService:
    """
    Reranking service using cross-encoder models.

    Cross-encoders are more accurate than bi-encoders for relevance scoring
    because they process the query and document together, allowing for
    better understanding of their relationship.

    Trade-off: Slower than bi-encoders (can't pre-compute), but much more accurate.
    """

    _instance = None
    _model = None

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize cross-encoder model."""
        if self._model is None:
            self._load_model()

    def _load_model(self):
        """Load the cross-encoder model."""
        # Use a lightweight but effective cross-encoder
        # ms-marco-MiniLM-L-6-v2: Optimized for CPU, good accuracy
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

        logger.info(f"Loading reranker model: {model_name}")

        try:
            self._model = CrossEncoder(
                model_name,
                max_length=512,  # Max token length for query+document
                device='cpu'  # Force CPU (works on your 8-core laptop)
            )

            logger.info("Reranker model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load reranker model: {str(e)}")
            raise

    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Rerank search results using cross-encoder.

        Args:
            query: User query
            results: List of search results from hybrid search
            top_k: Number of top results to return after reranking

        Returns:
            Reranked results (sorted by cross-encoder score)
        """
        if not results:
            return []

        logger.info(f"Reranking {len(results)} results with cross-encoder")

        try:
            # Extract text from results
            # Handle both Qdrant format and legacy format
            texts = []
            for result in results:
                # Try different possible text locations
                doc = result.get("document", {})
                text = (
                    doc.get("text") or
                    result.get("payload", {}).get("text") or
                    ""
                )
                texts.append(text[:512])  # Truncate to max length

            # Create query-document pairs
            pairs = [[query, text] for text in texts]

            # Score with cross-encoder
            scores = self._model.predict(pairs)

            # Add rerank scores to results
            for result, score in zip(results, scores):
                result['rerank_score'] = float(score)
                # Keep original score for reference
                if 'score' in result:
                    result['original_score'] = result['score']

            # Sort by rerank score (descending)
            reranked = sorted(
                results,
                key=lambda x: x.get('rerank_score', 0),
                reverse=True
            )

            # Return top-k
            top_results = reranked[:top_k]

            logger.info(
                f"Reranking complete. Top score: {top_results[0].get('rerank_score', 0):.3f}"
            )

            return top_results

        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}, returning original results")
            # Fallback: return original results
            return results[:top_k]

    def batch_rerank(
        self,
        query: str,
        results_list: List[List[Dict]],
        top_k: int = 5
    ) -> List[List[Dict]]:
        """
        Rerank multiple result lists (useful for multi-query scenarios).

        Args:
            query: User query
            results_list: List of result lists
            top_k: Number of top results per list

        Returns:
            List of reranked result lists
        """
        return [self.rerank(query, results, top_k) for results in results_list]

    def get_model_info(self) -> Dict:
        """Get information about the loaded reranker model."""
        return {
            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "model_type": "cross-encoder",
            "device": "cpu",
            "max_length": 512,
            "purpose": "Reranking search results for improved relevance"
        }


# Global instance
reranker_service = RerankerService()


def rerank_results(query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Convenience function for reranking.

    Args:
        query: User query
        results: Search results
        top_k: Number of results to return

    Returns:
        Reranked results
    """
    return reranker_service.rerank(query, results, top_k)
