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
            # Assemble context
            context_chunks, token_usage = context_manager.assemble_context(
                query=query,
                search_results=search_results,
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

        return """You are a helpful AI assistant that answers questions based on provided document context.

Instructions:
1. Answer the question using ONLY the information from the provided context chunks
2. If the context doesn't contain enough information, clearly state: "I don't have enough information in the provided documents to answer this question."
3. Include specific references when possible (e.g., "According to the document on page X...")
4. Be concise but comprehensive in your answers
5. If asked a follow-up question, consider the conversation history
6. Maintain a professional and helpful tone
7. If you find conflicting information in the context, mention both perspectives
8. Do not make up information or use knowledge outside the provided context

Format your response clearly and professionally."""

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

    def _calculate_confidence(self, search_results: List[Dict]) -> float:
        
        if not search_results:
            return 0.0

        top_score = search_results[0].get("text_match_score", 0) / 1000000

        confidence = min(top_score / 1000, 1.0)

        if len(search_results) >= 3:
            avg_score = sum(
                r.get("text_match_score", 0) for r in search_results[:3]
            ) / 3 / 1000000

            if avg_score > 500:  # High average score
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
