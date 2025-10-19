from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    APP_NAME: str = "Smart Document Q&A System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    MIN_UPLOAD_SIZE: int = 1024  # 1KB
    ALLOWED_EXTENSIONS: str = ".pdf,.docx,.doc,.txt,.md"
    UPLOAD_DIR: str = "./uploads"
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./rag_system.db"
    
    # API
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"
    
    # Document Processing
    CHUNK_SIZE: int = 500  # tokens
    CHUNK_OVERLAP: int = 50  # tokens

    # Typesense Configuration
    TYPESENSE_HOST: str = "localhost"
    TYPESENSE_PORT: int = 8108
    TYPESENSE_PROTOCOL: str = "http"
    TYPESENSE_API_KEY: str = ""
    TYPESENSE_COLLECTION_NAME: str = "smart_qa_document_rag"

    # Embedding Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    USE_GPU: bool = False

    # Groq LLM Configuration
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    GROQ_TEMPERATURE: float = 0.7
    GROQ_MAX_TOKENS: int = 2048

    # Agent Configuration
    MAX_CONVERSATION_HISTORY: int = 5
    CONTEXT_WINDOW_TOKENS: int = 8192
    TOP_K_SEARCH_RESULTS: int = 5
    HYBRID_SEARCH_WEIGHT_VECTOR: float = 0.7
    HYBRID_SEARCH_WEIGHT_KEYWORD: float = 0.3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )
    
    @property
    def allowed_extensions_set(self) -> set:
        """Convert comma-separated extensions to set."""
        return set(ext.strip() for ext in self.ALLOWED_EXTENSIONS.split(","))
    
    @property
    def upload_paths(self) -> dict:
        """Get upload directory paths."""
        base = Path(self.UPLOAD_DIR)
        return {
            "base": base,
            "raw": base / "raw",
            "processed": base / "processed",
            "metadata": base / "metadata"
        }
    
    def ensure_directories(self):
        """Create required directories if they don't exist."""
        for path in self.upload_paths.values():
            path.mkdir(parents=True, exist_ok=True)

        # Logs directory
        Path(self.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

    @property
    def typesense_config(self) -> dict:
        """Get Typesense connection configuration."""
        return {
            "nodes": [{
                "host": self.TYPESENSE_HOST,
                "port": str(self.TYPESENSE_PORT),
                "protocol": self.TYPESENSE_PROTOCOL
            }],
            "api_key": self.TYPESENSE_API_KEY,
            "connection_timeout_seconds": 60  # Increased to support hybrid search with embeddings
        }


# Global settings instance
settings = Settings()

# Ensure directories exist on import
settings.ensure_directories()
