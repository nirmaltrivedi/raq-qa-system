from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    original_filename: str
    file_type: str
    file_size: int
    status: str
    message: str
    
    class Config:
        from_attributes = True


class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    original_filename: str
    file_type: str
    file_size: int
    status: str
    total_pages: Optional[int] = None
    total_words: Optional[int] = None
    language: Optional[str] = "en"
    has_tables: bool = False
    uploaded_at: Optional[datetime] = None
    parsed_at: Optional[datetime] = None
    cleaned_at: Optional[datetime] = None
    indexed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    description: Optional[str] = None
    
    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    total: int
    documents: List[DocumentMetadata]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    status_code: int


class HealthCheckResponse(BaseModel):
    status: str
    app_name: str
    version: str
    timestamp: datetime
    database: str
    upload_dir: str

class SourceDocument(BaseModel):
    document_id: str
    chunk_id: str
    filename: str
    page_number: Optional[int] = None
    relevance_score: float
    snippet: Optional[str] = None  # Text snippet from the chunk


class QARequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="User question")
    session_id: Optional[str] = Field(None, description="Conversation session ID")
    include_sources: bool = Field(True, description="Include source documents in response")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of search results")


class QAResponse(BaseModel):
    answer: str
    session_id: str
    sources: Optional[List[SourceDocument]] = None
    timestamp: datetime
    tokens_used: Optional[int] = None
    confidence_score: Optional[float] = None


class ConversationHistoryMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    sources: Optional[List[SourceDocument]] = None


class ConversationHistoryResponse(BaseModel):
    session_id: str
    messages: List[ConversationHistoryMessage]
    message_count: int
    created_at: datetime
    updated_at: datetime


class ConversationListResponse(BaseModel):
    conversations: List[dict]
    total: int
