from sqlalchemy import Column, String, Integer, DateTime, Text, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class Document(Base):
    
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, index=True)
    
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False) 
    file_size = Column(Integer, nullable=False)  
    file_path = Column(String, nullable=False)  
    
    status = Column(
        String, 
        nullable=False, 
        default="uploaded"
    )  
    
    total_pages = Column(Integer, nullable=True)
    total_words = Column(Integer, nullable=True)
    total_chars = Column(Integer, nullable=True)
    language = Column(String, default="en")
    has_tables = Column(Boolean, default=False)
    
    processed_text_path = Column(String, nullable=True) 
    error_message = Column(Text, nullable=True)
    
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    parsed_at = Column(DateTime(timezone=True), nullable=True)
    cleaned_at = Column(DateTime(timezone=True), nullable=True)
    indexed_at = Column(DateTime(timezone=True), nullable=True)
    
    tags = Column(Text, nullable=True)  
    description = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename}, status={self.status})>"
    
    def to_dict(self):
        return {
            "document_id": self.id,
            "filename": self.filename,
            "original_filename": self.original_filename,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "status": self.status,
            "total_pages": self.total_pages,
            "total_words": self.total_words,
            "language": self.language,
            "has_tables": self.has_tables,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None,
            "parsed_at": self.parsed_at.isoformat() if self.parsed_at else None,
            "cleaned_at": self.cleaned_at.isoformat() if self.cleaned_at else None,
            "indexed_at": self.indexed_at.isoformat() if self.indexed_at else None,
            "error_message": self.error_message,
            "description": self.description
        }


class ProcessingLog(Base):

    __tablename__ = "processing_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String, nullable=False, index=True)
    stage = Column(String, nullable=False)  
    status = Column(String, nullable=False)  
    message = Column(Text, nullable=True)
    processing_time = Column(Float, nullable=True)  
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<ProcessingLog(doc={self.document_id}, stage={self.stage}, status={self.status})>"


class Conversation(Base):

    __tablename__ = "conversations"

    id = Column(String, primary_key=True, index=True)  
    user_id = Column(String, nullable=True, index=True)  
    title = Column(String, nullable=True)  
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    message_count = Column(Integer, default=0)

    def __repr__(self):
        return f"<Conversation(id={self.id}, messages={self.message_count})>"

    def to_dict(self):
        return {
            "session_id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "message_count": self.message_count
        }


class ConversationMessage(Base):

    __tablename__ = "conversation_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)  

    role = Column(String, nullable=False)  
    content = Column(Text, nullable=False)

    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    tokens_used = Column(Integer, nullable=True)

    sources = Column(Text, nullable=True)  
    confidence_score = Column(Float, nullable=True)

    def __repr__(self):
        return f"<ConversationMessage(session={self.session_id}, role={self.role})>"

    def to_dict(self):
        import json
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "tokens_used": self.tokens_used,
            "sources": json.loads(self.sources) if self.sources else None,
            "confidence_score": self.confidence_score
        }
