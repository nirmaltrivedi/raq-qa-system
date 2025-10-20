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
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  
    MIN_UPLOAD_SIZE: int = 1024  
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
    CHUNK_SIZE: int = 500  
    CHUNK_OVERLAP: int = 50  

    QDRANT_PATH: str = "./qdrant_storage"
    QDRANT_COLLECTION_NAME: str = "smart_qa_documents"
    QDRANT_DISTANCE_METRIC: str = "Cosine"
    QDRANT_USE_SPARSE_VECTORS: bool = True  

    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DIMENSION: int = 768
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
    def qdrant_config(self) -> dict:
        """Get Qdrant connection configuration."""
        return {
            "path": self.QDRANT_PATH,  
            "force_disable_check_same_thread": True 
        }


settings = Settings()

settings.ensure_directories()
