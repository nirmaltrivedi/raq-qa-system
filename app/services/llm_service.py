"""
LLM Service - Groq API integration using LangChain.
"""
from typing import List, Dict, Optional
import tiktoken
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from app.core.config import settings
from app.core.logging import app_logger as logger


class LLMService:
    """
    Service for interacting with Groq LLM via LangChain.
    """

    _instance = None
    _llm = None
    _tokenizer = None

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize Groq LLM client."""
        if self._llm is None:
            self._init_llm()
            self._init_tokenizer()

    def _init_llm(self):
        """Initialize Groq LLM client."""
        logger.info(f"Initializing Groq LLM: {settings.GROQ_MODEL}")

        try:
            self._llm = ChatGroq(
                groq_api_key=settings.GROQ_API_KEY,
                model_name=settings.GROQ_MODEL,
                temperature=settings.GROQ_TEMPERATURE,
                max_tokens=settings.GROQ_MAX_TOKENS
            )
            logger.info("Groq LLM initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM: {str(e)}")
            raise

    def _init_tokenizer(self):
        """Initialize tokenizer for counting tokens."""
        try:
            # Use cl100k_base encoding (compatible with most models)
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.info("Tokenizer initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize tokenizer: {str(e)}")
            self._tokenizer = None

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens in

        Returns:
            Number of tokens
        """
        if self._tokenizer is None:
            # Rough estimate: 1 token ≈ 4 characters
            return len(text) // 4

        try:
            return len(self._tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {str(e)}, using estimate")
            return len(text) // 4

    def generate_response(
        self,
        system_prompt: str,
        user_query: str,
        conversation_history: Optional[List[Dict]] = None,
        context_chunks: Optional[List[str]] = None
    ) -> Dict:
        """
        Generate response from LLM.

        Args:
            system_prompt: System instruction
            user_query: User question
            conversation_history: Previous messages (optional)
            context_chunks: Retrieved document chunks (optional)

        Returns:
            Dict with 'response' and 'tokens_used'
        """
        logger.info(f"Generating LLM response for query: '{user_query[:50]}...'")

        try:
            # Build messages
            messages = []

            # System message
            messages.append(SystemMessage(content=system_prompt))

            # Add conversation history if provided
            if conversation_history:
                for msg in conversation_history:
                    if msg['role'] == 'user':
                        messages.append(HumanMessage(content=msg['content']))
                    else:
                        messages.append(AIMessage(content=msg['content']))

            # Build user message with context
            user_message = user_query

            if context_chunks:
                # Prepend context to user query
                context_text = "\n\n".join([
                    f"[Context {i+1}]: {chunk}"
                    for i, chunk in enumerate(context_chunks)
                ])
                user_message = f"{context_text}\n\nBased on the context above, please answer:\n{user_query}"

            messages.append(HumanMessage(content=user_message))

            # Count total tokens (estimate)
            total_text = system_prompt + user_message
            if conversation_history:
                total_text += "".join([msg['content'] for msg in conversation_history])

            tokens_used = self.count_tokens(total_text)

            logger.info(f"Estimated tokens in prompt: {tokens_used}")

            # Generate response
            response = self._llm.invoke(messages)

            result = {
                "response": response.content,
                "tokens_used": tokens_used + self.count_tokens(response.content)
            }

            logger.info(f"Response generated successfully ({result['tokens_used']} tokens)")
            return result

        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            raise

    def generate_simple_response(self, prompt: str) -> str:
        """
        Generate a simple response without conversation history.

        Args:
            prompt: Single prompt string

        Returns:
            Response string
        """
        try:
            messages = [HumanMessage(content=prompt)]
            response = self._llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Simple generation failed: {str(e)}")
            raise


# Global instance
llm_service = LLMService()


def generate_answer(
    query: str,
    context_chunks: List[str],
    conversation_history: Optional[List[Dict]] = None
) -> Dict:
    """
    Convenience function to generate an answer.

    Args:
        query: User question
        context_chunks: Retrieved document chunks
        conversation_history: Previous conversation

    Returns:
        Dict with response and tokens_used
    """
    system_prompt = """You are a helpful AI assistant that answers questions based on provided document context.

Instructions:
1. Answer the question using ONLY the information from the provided context
2. If the context doesn't contain enough information, say so clearly
3. Include specific references to source documents when possible (e.g., "According to the document...")
4. Be concise but comprehensive
5. If asked a follow-up question, consider the conversation history
6. Always maintain a professional and helpful tone

If you cannot answer based on the context, respond with: "I don't have enough information in the provided documents to answer this question."
"""

    return llm_service.generate_response(
        system_prompt=system_prompt,
        user_query=query,
        conversation_history=conversation_history,
        context_chunks=context_chunks
    )
