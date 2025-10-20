"""
Document service - orchestrates document processing pipeline.
"""
from pathlib import Path
from typing import Dict
import json
import uuid
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.document import Document, ProcessingLog
from app.services.parser import DocumentParser
from app.services.cleaner import TextCleaner
from app.services.chunker import DocumentChunker
from app.services.embeddings import embedding_service
from app.services.qdrant_service import qdrant_service
from app.core.config import settings
from app.core.logging import app_logger as logger


class DocumentService:
    """Manages document processing workflow."""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.parser = DocumentParser()
        self.cleaner = TextCleaner()
    
    async def create_document_record(
        self,
        filename: str,
        original_filename: str,
        file_type: str,
        file_size: int,
        file_path: str
    ) -> Document:
        """Create initial document record in database."""
        
        document_id = str(uuid.uuid4())
        
        document = Document(
            id=document_id,
            filename=filename,
            original_filename=original_filename,
            file_type=file_type,
            file_size=file_size,
            file_path=file_path,
            status="uploaded"
        )
        
        self.db.add(document)
        await self.db.commit()
        await self.db.refresh(document)
        
        logger.info(f"Created document record: {document_id}")
        return document
    
    async def log_processing_step(
        self,
        document_id: str,
        stage: str,
        status: str,
        message: str = None,
        processing_time: float = None
    ):
        """Log processing step to database."""
        
        log_entry = ProcessingLog(
            document_id=document_id,
            stage=stage,
            status=status,
            message=message,
            processing_time=processing_time
        )
        
        self.db.add(log_entry)
        await self.db.commit()
    
    async def update_document_status(
        self,
        document_id: str,
        status: str,
        error_message: str = None,
        **kwargs
    ):
        """Update document status and metadata."""
        
        result = await self.db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()
        
        if document:
            document.status = status
            if error_message:
                document.error_message = error_message
            
            # Update additional fields
            for key, value in kwargs.items():
                if hasattr(document, key):
                    setattr(document, key, value)
            
            await self.db.commit()
            logger.info(f"Updated document {document_id} status to: {status}")
    
    async def process_document(self, document_id: str) -> Dict:
        """
        Complete document processing pipeline:
        1. Parse document
        2. Clean text
        3. Save processed text
        4. Update metadata
        
        Args:
            document_id: Document ID to process
            
        Returns:
            Dict with processing results
        """
        logger.info(f"Starting document processing: {document_id}")
        
        try:
            # Get document from database
            result = await self.db.execute(
                select(Document).where(Document.id == document_id)
            )
            document = result.scalar_one_or_none()
            
            if not document:
                raise ValueError(f"Document not found: {document_id}")
            
            # Update status to parsing
            await self.update_document_status(document_id, "parsing")
            
            # Step 1: Parse document
            logger.info(f"Parsing document: {document.filename}")
            file_path = Path(document.file_path)
            
            try:
                parsed_data = self.parser.parse_document(file_path, document.file_type)
                
                # Log parsing success
                await self.log_processing_step(
                    document_id,
                    "parsing",
                    "success",
                    f"Parsed {parsed_data['metadata'].get('total_pages', 'N/A')} pages",
                    parsed_data['metadata'].get('processing_time')
                )
                
                # Update document with parsing metadata
                await self.update_document_status(
                    document_id,
                    "parsed",
                    total_pages=parsed_data['metadata'].get('total_pages'),
                    total_words=parsed_data['metadata'].get('total_words'),
                    has_tables=parsed_data['metadata'].get('has_tables', False),
                    parsed_at=datetime.utcnow()
                )
                
            except Exception as e:
                error_msg = f"Parsing failed: {str(e)}"
                logger.error(error_msg)
                await self.log_processing_step(document_id, "parsing", "error", error_msg)
                await self.update_document_status(document_id, "failed", error_msg)
                raise
            
            # Step 2: Clean text
            logger.info(f"Cleaning text for document: {document.filename}")
            await self.update_document_status(document_id, "cleaning")
            
            try:
                cleaned_data = self.cleaner.clean_text(parsed_data['raw_text'])
                
                # Log cleaning success
                await self.log_processing_step(
                    document_id,
                    "cleaning",
                    "success",
                    f"Cleaned text: {cleaned_data['metadata']['chars_removed']} chars removed",
                    cleaned_data['metadata']['processing_time']
                )
                
            except Exception as e:
                error_msg = f"Cleaning failed: {str(e)}"
                logger.error(error_msg)
                await self.log_processing_step(document_id, "cleaning", "error", error_msg)
                await self.update_document_status(document_id, "failed", error_msg)
                raise
            
            # Step 3: Save processed text
            logger.info(f"Saving processed text for document: {document.filename}")
            
            processed_dir = settings.upload_paths["processed"]
            processed_file = processed_dir / f"{document_id}.json"
            
            # Combine parsed and cleaned data
            processed_data = {
                "document_id": document_id,
                "filename": document.filename,
                "file_type": document.file_type,
                "parsed_data": parsed_data,
                "cleaned_data": cleaned_data,
                "processed_at": datetime.utcnow().isoformat()
            }
            
            with open(processed_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
            # Update document with final status
            await self.update_document_status(
                document_id,
                "cleaned",
                processed_text_path=str(processed_file),
                total_chars=cleaned_data['metadata']['cleaned_char_count'],
                cleaned_at=datetime.utcnow()
            )
            
            # Step 4: Chunk the cleaned text
            logger.info(f"Chunking document: {document.filename}")
            await self.update_document_status(document_id, "chunking")
            
            try:
                chunker = DocumentChunker(strategy="sentence-aware")
                
                chunk_metadata = {
                    "document_id": document_id,
                    "filename": document.original_filename,
                    "file_type": document.file_type
                }
                
                chunks = chunker.chunk_text(cleaned_data['cleaned_text'], chunk_metadata)
                chunk_stats = chunker.get_chunk_stats(chunks)
                
                logger.info(f"Created {len(chunks)} chunks")
                
                # Log chunking success
                await self.log_processing_step(
                    document_id,
                    "chunking",
                    "success",
                    f"Created {len(chunks)} chunks",
                    None
                )
                
                # Add chunks to processed data
                processed_data['chunks'] = chunks
                processed_data['chunk_stats'] = chunk_stats
                
                # Save updated data with chunks
                with open(processed_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, indent=2, ensure_ascii=False)
                
            except Exception as e:
                error_msg = f"Chunking failed: {str(e)}"
                logger.error(error_msg)
                await self.log_processing_step(document_id, "chunking", "error", error_msg)
                # Don't fail the whole process, just continue
                chunks = []
            
            # Step 5: Generate embeddings
            if chunks:
                logger.info(f"Generating embeddings for {len(chunks)} chunks")
                await self.update_document_status(document_id, "embedding")
                
                try:
                    chunks_with_embeddings = embedding_service.embed_chunks(chunks)
                    
                    logger.info(f"Generated embeddings for all chunks")
                    
                    # Log embedding success
                    await self.log_processing_step(
                        document_id,
                        "embedding",
                        "success",
                        f"Generated embeddings for {len(chunks)} chunks",
                        None
                    )
                    
                    # Update processed data with embeddings
                    processed_data['chunks'] = chunks_with_embeddings
                    
                    # Save with embeddings
                    with open(processed_file, 'w', encoding='utf-8') as f:
                        json.dump(processed_data, f, indent=2, ensure_ascii=False)
                    
                except Exception as e:
                    error_msg = f"Embedding generation failed: {str(e)}"
                    logger.error(error_msg)
                    await self.log_processing_step(document_id, "embedding", "error", error_msg)
                    chunks_with_embeddings = chunks  # Continue without embeddings
            else:
                chunks_with_embeddings = []
            
            # Step 6: Index in Qdrant
            if chunks_with_embeddings:
                logger.info(f"Indexing {len(chunks_with_embeddings)} chunks in Qdrant")
                await self.update_document_status(document_id, "indexing")

                try:
                    index_result = qdrant_service.index_chunks(
                        chunks_with_embeddings,
                        document_id
                    )

                    logger.info(f"Indexed {index_result['successful']} chunks successfully")

                    # Log indexing success
                    await self.log_processing_step(
                        document_id,
                        "indexing",
                        "success",
                        f"Indexed {index_result['successful']} chunks",
                        None
                    )

                    # Final status: indexed
                    await self.update_document_status(
                        document_id,
                        "indexed",
                        indexed_at=datetime.utcnow()
                    )

                except Exception as e:
                    error_msg = f"Qdrant indexing failed: {str(e)}"
                    logger.error(error_msg)
                    await self.log_processing_step(document_id, "indexing", "error", error_msg)
                    # Mark as cleaned but not indexed
                    await self.update_document_status(document_id, "cleaned")
            
            logger.info(f"Document processing completed: {document_id}")
            
            return {
                "status": "success",
                "document_id": document_id,
                "parsed_metadata": parsed_data['metadata'],
                "cleaned_metadata": cleaned_data['metadata'],
                "chunk_stats": chunk_stats if chunks else None,
                "indexed": len(chunks_with_embeddings) if chunks_with_embeddings else 0,
                "processed_file": str(processed_file)
            }
            
        except Exception as e:
            logger.error(f"Document processing failed for {document_id}: {str(e)}")
            raise
    
    async def get_document(self, document_id: str) -> Document:
        """Get document by ID."""
        result = await self.db.execute(
            select(Document).where(Document.id == document_id)
        )
        return result.scalar_one_or_none()
    
    async def list_documents(self, skip: int = 0, limit: int = 100):
        """List all documents with pagination."""
        result = await self.db.execute(
            select(Document)
            .order_by(Document.uploaded_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_processed_text(self, document_id: str) -> Dict:
        """Get processed text for a document."""
        document = await self.get_document(document_id)
        
        if not document or not document.processed_text_path:
            return None
        
        processed_file = Path(document.processed_text_path)
        if not processed_file.exists():
            return None
        
        with open(processed_file, 'r', encoding='utf-8') as f:
            return json.load(f)
