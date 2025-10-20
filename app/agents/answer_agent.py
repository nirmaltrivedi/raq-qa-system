from typing import Dict, Any, List, Optional
from app.agents.base_agent import BaseAgent
from app.services.llm_service import llm_service
from app.services.context_manager import context_manager


class AnswerAgent(BaseAgent):

    def __init__(self):
        super().__init__(name="AnswerAgent")

    async def execute(
        self,
        query: str,
        search_results: List[Dict],
        conversation_history: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        self.log_execution("Generating answer", f"query='{query[:50]}...'")

        try:
            # Filter low-quality chunks before processing
            filtered_results = self._filter_low_quality_chunks(search_results)

            self.log_execution(
                "Chunk filtering",
                f"Filtered {len(search_results)} → {len(filtered_results)} chunks"
            )

            # Assemble context
            context_chunks, token_usage = context_manager.assemble_context(
                query=query,
                search_results=filtered_results,
                conversation_history=conversation_history
            )

            if not context_chunks:
                self.log_execution("No context available", "Using query only")
                return {
                    "success": True,
                    "answer": "I don't have enough information in the available documents to answer this question.",
                    "sources": [],
                    "tokens_used": 0,
                    "confidence_score": 0.0
                }

            # Generate response using LLM
            system_prompt = self._build_system_prompt()

            result = llm_service.generate_response(
                system_prompt=system_prompt,
                user_query=query,
                conversation_history=conversation_history,
                context_chunks=context_chunks
            )

            # Extract sources from search results
            sources = self._extract_sources(search_results)

            self.log_execution(
                "Answer generated",
                f"tokens={result['tokens_used']}, sources={len(sources)}"
            )

            return {
                "success": True,
                "answer": result["response"],
                "sources": sources[:5],  # Top 5 sources
                "tokens_used": result["tokens_used"],
                "token_usage": token_usage,
                "confidence_score": self._calculate_confidence(search_results)
            }

        except Exception as e:
            self.log_error(f"Answer generation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "answer": "I encountered an error while generating the answer. Please try again."
            }

    def _build_system_prompt(self) -> str:
        """
        Build an improved system prompt that encourages more helpful answers
        while maintaining accuracy and grounding in context.
        """
        return """You are a helpful AI assistant that answers questions based on provided document context.

Instructions:
1. Answer the question using the information from the provided context chunks
2. If the context contains relevant information, synthesize it into a comprehensive answer
3. You may make reasonable inferences from the context, but clearly distinguish between:
   - Direct information from documents: Use "According to [document/source]..."
   - Your synthesis/inference: Use "Based on this information..." or "This suggests..."
4. ONLY say "I don't have enough information" if the context is completely unrelated to the question
5. If you find partial information:
   - Provide what you can answer
   - Clearly state what aspects you cannot address
   - Example: "While the documents explain X, they don't cover Y"
6. Include specific references when possible (e.g., "According to [filename] on page X...")
7. If asked a follow-up question, use conversation history to maintain context
8. If you find conflicting information, mention both perspectives and note the discrepancy
9. Be concise but comprehensive - prioritize clarity over brevity
10. Maintain a professional and helpful tone

Context Quality Guidelines:
- High confidence: Multiple relevant chunks with consistent information → Provide detailed answer
- Medium confidence: Some relevant information but incomplete → Answer what you can, note gaps
- Low confidence: Tangentially related information → Provide context-based insights with caveats
- No confidence: Completely unrelated context → State lack of information

Format your response clearly with proper structure (paragraphs, bullet points if helpful)."""

    def _extract_sources(self, search_results: List[Dict]) -> List[Dict]:

        sources = []

        for result in search_results:
            doc = result.get("document", {})

            source = {
                "document_id": doc.get("document_id", ""),
                "chunk_id": doc.get("id", ""),
                "filename": doc.get("filename", "unknown"),
                "relevance_score": result.get("text_match_score", 0) / 1000000,  # Normalize
                "snippet": doc.get("text", "")[:200] + "..."  # First 200 chars
            }

            # Add page number if available
            if "page_number" in doc:
                source["page_number"] = doc["page_number"]

            sources.append(source)

        return sources

    def _filter_low_quality_chunks(
        self,
        search_results: List[Dict],
        min_score: float = 0.3
    ) -> List[Dict]:
        """
        Filter out chunks with very low relevance scores.

        Args:
            search_results: List of search results with scores
            min_score: Minimum score threshold (0-1 scale)

        Returns:
            Filtered list of search results
        """
        if not search_results:
            return []

        # Try to get score from different possible fields
        filtered = []
        for result in search_results:
            score = result.get("score", 0)

            # If score is not in 0-1 range, try to normalize
            if score > 1:
                score = min(score / 1000000, 1.0)

            # Keep chunks above threshold
            if score >= min_score:
                filtered.append(result)

        # If filtering removed everything, keep at least top 3
        if not filtered and search_results:
            return search_results[:3]

        return filtered

    def _calculate_confidence(self, search_results: List[Dict]) -> float:
        """
        Calculate confidence score based on search results quality.

        Uses the score field from Qdrant hybrid search (0-1 range).
        """
        if not search_results:
            return 0.0

        # Get top score (Qdrant returns scores in 0-1 range for hybrid search)
        top_score = search_results[0].get("score", 0)

        # Normalize if needed (for backward compatibility)
        if top_score > 1:
            top_score = min(top_score / 1000000, 1.0)

        # Base confidence on top score
        confidence = min(top_score, 1.0)

        # Boost confidence if multiple high-quality results
        if len(search_results) >= 3:
            avg_score = sum(r.get("score", 0) for r in search_results[:3]) / 3

            if avg_score > 1:  # Normalize
                avg_score = min(avg_score / 1000000, 1.0)

            if avg_score > 0.5:  # High average score
                confidence = min(confidence + 0.1, 1.0)

        return round(confidence, 2)

    def format_answer_with_citations(
        self,
        answer: str,
        sources: List[Dict]
    ) -> str:

        formatted = answer

        # Add sources at the end
        if sources:
            formatted += "\n\n**Sources:**\n"
            for i, source in enumerate(sources[:3], 1):
                filename = source.get("filename", "Unknown")
                page = source.get("page_number")

                citation = f"{i}. {filename}"
                if page:
                    citation += f" (Page {page})"

                formatted += f"\n{citation}"

        return formatted
