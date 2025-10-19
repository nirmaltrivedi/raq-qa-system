from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pathlib import Path
import uuid
import aiofiles
from datetime import datetime

from app.core.database import get_db
from app.core.config import settings
from app.core.logging import app_logger as logger
from app.services.document_service import DocumentService
from app.models.schemas import (
    DocumentUploadResponse,
    DocumentMetadata,
    DocumentListResponse,
    ErrorResponse
)


router = APIRouter(prefix="/documents", tags=["documents"])


def validate_file(file: UploadFile) -> tuple:
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Check extension
    file_path = Path(file.filename)
    file_ext = file_path.suffix.lower()
    
    if file_ext not in settings.allowed_extensions_set:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(settings.allowed_extensions_set)}"
        )
    
    return file_ext


def sanitize_filename(filename: str) -> str:
    # Remove path components
    filename = Path(filename).name
    
    # Replace spaces and special chars with underscores
    safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in filename)
    
    # Remove multiple consecutive underscores
    safe_name = "_".join(filter(None, safe_name.split("_")))
    
    return safe_name


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db)
):
    
    logger.info(f"Received upload request: {file.filename}")
    
    try:
        # Validate file
        file_ext = validate_file(file)
        
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        # Validate size
        if file_size < settings.MIN_UPLOAD_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too small. Minimum size: {settings.MIN_UPLOAD_SIZE} bytes"
            )
        
        if file_size > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE / 1024 / 1024:.0f}MB"
            )
        
        # Generate unique filename
        document_id = str(uuid.uuid4())
        sanitized_name = sanitize_filename(file.filename)
        safe_filename = f"{document_id}_{sanitized_name}"
        
        # Save to raw directory
        raw_dir = settings.upload_paths["raw"]
        file_path = raw_dir / safe_filename
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        logger.info(f"File saved: {file_path} ({file_size} bytes)")
        
        # Create document record in database
        doc_service = DocumentService(db)
        document = await doc_service.create_document_record(
            filename=safe_filename,
            original_filename=file.filename,
            file_type=file_ext,
            file_size=file_size,
            file_path=str(file_path)
        )
        
        # Process document in background
        if background_tasks:
            background_tasks.add_task(doc_service.process_document, document.id)
            logger.info(f"Added background task for document processing: {document.id}")
        
        return DocumentUploadResponse(
            document_id=document.id,
            filename=document.filename,
            original_filename=document.original_filename,
            file_type=document.file_type,
            file_size=document.file_size,
            status=document.status,
            message="Document uploaded successfully. Processing in background."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/{document_id}", response_model=DocumentMetadata)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    
    doc_service = DocumentService(db)
    document = await doc_service.get_document(document_id)
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return document.to_dict()


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    
    doc_service = DocumentService(db)
    documents = await doc_service.list_documents(skip=skip, limit=limit)
    
    return DocumentListResponse(
        total=len(documents),
        documents=[doc.to_dict() for doc in documents]
    )


@router.post("/{document_id}/process")
async def process_document(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    
    doc_service = DocumentService(db)
    document = await doc_service.get_document(document_id)
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document.status not in ["uploaded", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Document cannot be processed. Current status: {document.status}"
        )
    
    try:
        result = await doc_service.process_document(document_id)
        return {
            "status": "success",
            "message": "Document processed successfully",
            "result": result
        }
    except Exception as e:
        logger.error(f"Processing failed for {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("/{document_id}/content")
async def get_document_content(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    
    doc_service = DocumentService(db)
    document = await doc_service.get_document(document_id)
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document.status != "cleaned":
        raise HTTPException(
            status_code=400,
            detail=f"Document not yet processed. Current status: {document.status}"
        )
    
    processed_data = await doc_service.get_processed_text(document_id)
    
    if not processed_data:
        raise HTTPException(status_code=404, detail="Processed text not found")
    
    return processed_data
