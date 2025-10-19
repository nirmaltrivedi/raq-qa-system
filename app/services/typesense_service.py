"""
Typesense indexing service for document chunks.
Supports hybrid search (vector + keyword).
"""
from typing import List, Dict, Optional
import typesense
from app.core.logging import app_logger as logger
from app.core.config import settings


class TypesenseService:
    """
    Service for managing Typesense collections and indexing.
    """
    
    _instance = None
    _client = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize Typesense client."""
        if self._client is None:
            self._init_client()
    
    def _init_client(self):
        """Initialize Typesense client connection."""
        logger.info("Initializing Typesense client...")
        
        try:
            self._client = typesense.Client(settings.typesense_config)
            
            # Test connection by listing collections (or creating one if none exist)
            try:
                collections = self._client.collections.retrieve()
                logger.info(f"Typesense connected successfully. Found {len(collections)} collections.")
            except Exception as e:
                # If we can't retrieve collections, at least the client was created
                logger.info(f"Typesense client initialized. Will verify connection on first use.")
            
        except Exception as e:
            logger.error(f"Failed to initialize Typesense client: {str(e)}")
            raise
    
    @property
    def client(self):
        """Get Typesense client."""
        return self._client
    
    def create_collection(self, collection_name: str = None, force: bool = False) -> Dict:
        """
        Create Typesense collection for document chunks.
        
        Args:
            collection_name: Name of collection (default from settings)
            force: If True, delete existing collection first
            
        Returns:
            Collection creation response
        """
        collection_name = collection_name or settings.TYPESENSE_COLLECTION_NAME
        
        logger.info(f"Creating Typesense collection: {collection_name}")
        
        # Define schema
        schema = {
            "name": collection_name,
            "fields": [
                # IDs
                {"name": "id", "type": "string"},
                {"name": "document_id", "type": "string", "facet": True},
                {"name": "chunk_index", "type": "int32"},
                
                # Content
                {"name": "text", "type": "string"},
                
                # Metadata
                {"name": "filename", "type": "string", "facet": True},
                {"name": "file_type", "type": "string", "facet": True, "optional": True},
                {"name": "char_count", "type": "int32"},
                {"name": "word_count", "type": "int32"},
                
                # Vector embedding for semantic search
                {
                    "name": "embedding",
                    "type": "float[]",
                    "num_dim": settings.EMBEDDING_DIMENSION
                },
                
                # Optional fields
                {"name": "page_number", "type": "int32", "optional": True},
                {"name": "created_at", "type": "int64"},
            ],
            "default_sorting_field": "created_at"
        }
        
        try:
            # Delete if exists and force=True
            if force:
                try:
                    self._client.collections[collection_name].delete()
                    logger.info(f"Deleted existing collection: {collection_name}")
                except Exception:
                    pass
            
            # Create collection
            response = self._client.collections.create(schema)
            logger.info(f"Collection created successfully: {collection_name}")
            return response
            
        except Exception as e:
            if "already exists" in str(e):
                logger.warning(f"Collection already exists: {collection_name}")
                return {"status": "exists"}
            else:
                logger.error(f"Failed to create collection: {str(e)}")
                raise
    
    def collection_exists(self, collection_name: str = None) -> bool:
        """Check if collection exists."""
        collection_name = collection_name or settings.TYPESENSE_COLLECTION_NAME
        
        try:
            self._client.collections[collection_name].retrieve()
            return True
        except Exception:
            return False
    
    def index_chunks(
        self,
        chunks: List[Dict],
        document_id: str,
        collection_name: str = None
    ) -> Dict:
        """
        Index document chunks into Typesense.
        
        Args:
            chunks: List of chunk dictionaries with text and embeddings
            document_id: Document ID
            collection_name: Collection name
            
        Returns:
            Indexing statistics
        """
        collection_name = collection_name or settings.TYPESENSE_COLLECTION_NAME
        
        logger.info(f"Indexing {len(chunks)} chunks for document {document_id}")
        
        # Ensure collection exists
        if not self.collection_exists(collection_name):
            logger.info("Collection doesn't exist, creating...")
            self.create_collection(collection_name)
        
        # Prepare documents for indexing
        import time
        current_timestamp = int(time.time())
        
        documents = []
        for chunk in chunks:
            doc = {
                "id": f"{document_id}_{chunk['chunk_index']}",
                "document_id": document_id,
                "chunk_index": chunk['chunk_index'],
                "text": chunk['text'],
                "char_count": chunk['char_count'],
                "word_count": chunk['word_count'],
                "created_at": current_timestamp
            }
            
            # Add optional fields if present
            if 'filename' in chunk:
                doc['filename'] = chunk['filename']
            if 'file_type' in chunk:
                doc['file_type'] = chunk['file_type']
            if 'page_number' in chunk:
                doc['page_number'] = chunk['page_number']
            if 'embedding' in chunk:
                doc['embedding'] = chunk['embedding']
            
            documents.append(doc)
        
        try:
            # Batch import
            results = self._client.collections[collection_name].documents.import_(
                documents,
                {"action": "upsert"}  # Update if exists, insert if not
            )
            
            # Count successes and failures
            success_count = sum(1 for r in results if r.get('success'))
            failure_count = len(results) - success_count
            
            logger.info(f"Indexed {success_count} chunks successfully, {failure_count} failures")
            
            return {
                "total_chunks": len(chunks),
                "successful": success_count,
                "failed": failure_count,
                "collection": collection_name
            }
            
        except Exception as e:
            logger.error(f"Indexing failed: {str(e)}")
            raise
    
    def search(
        self,
        query: str,
        collection_name: str = None,
        limit: int = 10,
        filter_by: str = None,
        hybrid_search: bool = True
    ) -> Dict:
        """
        Search documents using hybrid search (vector + keyword).
        
        Args:
            query: Search query
            collection_name: Collection to search
            limit: Number of results
            filter_by: Filter expression (e.g., "document_id:=abc123")
            hybrid_search: Use both vector and keyword search
            
        Returns:
            Search results
        """
        collection_name = collection_name or settings.TYPESENSE_COLLECTION_NAME
        
        logger.info(f"Searching: '{query}' in {collection_name}")
        
        search_params = {
            "q": query,
            "query_by": "text",
            "limit": limit,
        }
        
        # Add vector search if hybrid
        if hybrid_search:
            # Generate embedding for query
            from app.services.embeddings import embedding_service
            query_embedding = embedding_service.encode_single(query)
            search_params["vector_query"] = f"embedding:([{','.join(map(str, query_embedding))}], k:{limit})"
        
        # Add filters if provided
        if filter_by:
            search_params["filter_by"] = filter_by
        
        try:
            results = self._client.collections[collection_name].documents.search(search_params)
            
            logger.info(f"Found {results.get('found', 0)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise
    
    def delete_document_chunks(
        self,
        document_id: str,
        collection_name: str = None
    ) -> Dict:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document ID
            collection_name: Collection name
            
        Returns:
            Deletion statistics
        """
        collection_name = collection_name or settings.TYPESENSE_COLLECTION_NAME
        
        logger.info(f"Deleting chunks for document: {document_id}")
        
        try:
            # Delete by filter
            result = self._client.collections[collection_name].documents.delete({
                "filter_by": f"document_id:={document_id}"
            })
            
            logger.info(f"Deleted {result.get('num_deleted', 0)} chunks")
            return result
            
        except Exception as e:
            logger.error(f"Deletion failed: {str(e)}")
            raise
    
    def get_collection_stats(self, collection_name: str = None) -> Dict:
        """Get collection statistics."""
        collection_name = collection_name or settings.TYPESENSE_COLLECTION_NAME
        
        try:
            collection = self._client.collections[collection_name].retrieve()
            return {
                "name": collection.get("name"),
                "num_documents": collection.get("num_documents"),
                "num_fields": len(collection.get("fields", [])),
                "created_at": collection.get("created_at")
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {}


# Global instance
typesense_service = TypesenseService()


def index_document_chunks(chunks: List[Dict], document_id: str) -> Dict:
    """
    Convenience function to index document chunks.
    
    Args:
        chunks: List of chunks with embeddings
        document_id: Document ID
        
    Returns:
        Indexing statistics
    """
    return typesense_service.index_chunks(chunks, document_id)