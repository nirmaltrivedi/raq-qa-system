"""
Context Manager - Intelligent context assembly and token budget management.
"""
from typing import List, Dict, Optional, Tuple
from app.core.config import settings
from app.core.logging import app_logger as logger


class ContextManager:

    def __init__(self, max_tokens: Optional[int] = None):
        self.max_tokens = max_tokens or settings.CONTEXT_WINDOW_TOKENS
        self.system_prompt_tokens = 300  # Reserved for system prompt
        self.response_buffer_tokens = 2048  # Reserved for response
        self.query_tokens_estimate = 100  # Estimate for query

        # Calculate available tokens for context
        self.available_tokens = (
            self.max_tokens
            - self.system_prompt_tokens
            - self.response_buffer_tokens
            - self.query_tokens_estimate
        )

        logger.info(f"ContextManager initialized with {self.available_tokens} tokens available for context")

    def estimate_tokens(self, text: str) -> int:
        # Simple estimation: 1 token ≈ 4 characters
        # More accurate with tiktoken, but this is faster
        return len(text) // 4

    def assemble_context(
        self,
        query: str,
        search_results: List[Dict],
        conversation_history: Optional[List[Dict]] = None,
        top_k: int = 5
    ) -> Tuple[List[str], Dict]:
        """
        Assemble context from search results and conversation history.

        Args:
            query: User query
            search_results: Retrieved document chunks
            conversation_history: Previous messages
            top_k: Number of top results to prioritize

        Returns:
            Tuple of (context_chunks, metadata)
        """
        logger.info(f"Assembling context for query with {len(search_results)} search results")

        # Track token usage
        token_usage = {
            "system_prompt": self.system_prompt_tokens,
            "query": self.estimate_tokens(query),
            "response_buffer": self.response_buffer_tokens,
            "conversation": 0,
            "search_results": 0,
            "total": 0
        }

        # Available tokens for dynamic content
        remaining_tokens = self.available_tokens

        # Priority 1: Recent conversation (if exists)
        conversation_tokens = 0
        if conversation_history:
            # Take last 3 messages max
            recent_messages = conversation_history[-3:]
            conversation_text = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in recent_messages
            ])
            conversation_tokens = self.estimate_tokens(conversation_text)

            # Limit conversation to 20% of available tokens
            max_conversation_tokens = int(self.available_tokens * 0.2)
            if conversation_tokens > max_conversation_tokens:
                # Truncate conversation
                logger.info(f"Truncating conversation from {conversation_tokens} to {max_conversation_tokens} tokens")
                conversation_tokens = max_conversation_tokens

            remaining_tokens -= conversation_tokens
            token_usage["conversation"] = conversation_tokens

        # Priority 2: Search results
        context_chunks = []
        search_tokens = 0

        for i, result in enumerate(search_results[:top_k * 2]):  # Consider more results initially
            # Handle nested document structure from SearchAgent
            if "document" in result:
                chunk_text = result["document"].get("text", "")
            else:
                chunk_text = result.get("text", "")

            chunk_tokens = self.estimate_tokens(chunk_text)

            # Check if we can fit this chunk
            if search_tokens + chunk_tokens <= remaining_tokens:
                context_chunks.append(chunk_text)
                search_tokens += chunk_tokens
                logger.debug(f"Added chunk {i+1} ({chunk_tokens} tokens)")
            else:
                # Check if we have minimum required chunks
                if len(context_chunks) >= 3:
                    logger.info(f"Reached token limit with {len(context_chunks)} chunks")
                    break
                else:
                    # Truncate chunk to fit
                    available = remaining_tokens - search_tokens
                    if available > 100:  # Only if we have meaningful space
                        truncated_text = chunk_text[:available * 4]  # Rough char estimate
                        context_chunks.append(truncated_text + "...")
                        search_tokens += available
                        logger.info(f"Truncated chunk {i+1} to fit")
                    break

        token_usage["search_results"] = search_tokens
        token_usage["total"] = (
            token_usage["system_prompt"]
            + token_usage["query"]
            + token_usage["conversation"]
            + token_usage["search_results"]
            + token_usage["response_buffer"]
        )

        logger.info(f"Context assembled: {len(context_chunks)} chunks, {token_usage['total']} total tokens")
        logger.info(f"Token breakdown: {token_usage}")

        return context_chunks, token_usage

    def format_search_results_for_context(
        self,
        search_results: List[Dict],
        include_metadata: bool = True
    ) -> List[Dict]:
        """
        Format search results for context assembly.

        Args:
            search_results: Raw search results from Typesense
            include_metadata: Include source metadata

        Returns:
            Formatted results
        """
        formatted = []

        for result in search_results:
            # Extract document from Typesense result structure
            doc = result.get("document", result)

            formatted_result = {
                "text": doc.get("text", ""),
                "document_id": doc.get("document_id", ""),
                "chunk_id": doc.get("id", ""),
                "filename": doc.get("filename", "unknown"),
                "relevance_score": result.get("text_match_score", 0) / 1000000  # Normalize score
            }

            if include_metadata and "page_number" in doc:
                formatted_result["page_number"] = doc["page_number"]

            formatted.append(formatted_result)

        return formatted

    def deduplicate_chunks(self, chunks: List[str], similarity_threshold: float = 0.9) -> List[str]:
        """
        Remove duplicate or highly similar chunks.

        Args:
            chunks: List of chunk texts
            similarity_threshold: Similarity threshold (0-1)

        Returns:
            Deduplicated chunks
        """
        if not chunks:
            return chunks

        unique_chunks = [chunks[0]]

        for chunk in chunks[1:]:
            is_duplicate = False

            for unique_chunk in unique_chunks:
                # Simple similarity: compare lengths and overlapping words
                words1 = set(chunk.lower().split())
                words2 = set(unique_chunk.lower().split())

                if len(words1) == 0 or len(words2) == 0:
                    continue

                overlap = len(words1 & words2)
                similarity = overlap / max(len(words1), len(words2))

                if similarity >= similarity_threshold:
                    is_duplicate = True
                    logger.debug(f"Removed duplicate chunk (similarity: {similarity:.2f})")
                    break

            if not is_duplicate:
                unique_chunks.append(chunk)

        logger.info(f"Deduplicated {len(chunks)} → {len(unique_chunks)} chunks")
        return unique_chunks

    def prioritize_chunks(
        self,
        chunks: List[Dict],
        query: str,
        relevance_weight: float = 0.7,
        recency_weight: float = 0.3
    ) -> List[Dict]:
        """
        Prioritize chunks based on relevance and recency.

        Args:
            chunks: List of chunk dictionaries with scores
            query: User query
            relevance_weight: Weight for relevance score
            recency_weight: Weight for recency

        Returns:
            Sorted chunks by priority
        """
        # For now, primarily use relevance (from Typesense hybrid search)
        # In future, can add recency boosting based on document timestamps

        sorted_chunks = sorted(
            chunks,
            key=lambda x: x.get("relevance_score", 0),
            reverse=True
        )

        logger.info("Chunks prioritized by relevance")
        return sorted_chunks


# Global instance
context_manager = ContextManager()


def assemble_qa_context(
    query: str,
    search_results: List[Dict],
    conversation_history: Optional[List[Dict]] = None
) -> Tuple[List[str], Dict]:
    """
    Convenience function to assemble Q&A context.

    Args:
        query: User query
        search_results: Search results from Typesense
        conversation_history: Previous conversation

    Returns:
        Tuple of (context_chunks, token_usage)
    """
    # Format search results
    formatted_results = context_manager.format_search_results_for_context(search_results)

    # Prioritize chunks
    prioritized = context_manager.prioritize_chunks(formatted_results, query)

    # Assemble context
    return context_manager.assemble_context(
        query=query,
        search_results=prioritized,
        conversation_history=conversation_history,
        top_k=settings.TOP_K_SEARCH_RESULTS
    )
