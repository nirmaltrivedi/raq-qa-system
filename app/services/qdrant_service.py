"""
Qdrant vector database service for hybrid search (BM25 + Vector).
Embedded mode - no Docker required.
"""
import uuid
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    ScoredPoint,
    SearchRequest,
    Prefetch,
    QueryRequest,
    SparseVector,
    SparseIndexParams,
    SparseVectorParams,
)
from app.core.logging import app_logger as logger
from app.core.config import settings


class QdrantService:
    """
    Service for managing Qdrant collections and hybrid search.
    Uses embedded mode (no Docker needed) for local development.
    """

    _instance = None
    _client = None

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize Qdrant client."""
        if self._client is None:
            self._init_client()

    def _init_client(self):
        """Initialize Qdrant client in embedded mode."""
        logger.info("Initializing Qdrant client (embedded mode)...")

        try:
            # Create embedded Qdrant client (local storage, no Docker)
            self._client = QdrantClient(path=settings.QDRANT_PATH)

            logger.info(f"Qdrant client initialized at: {settings.QDRANT_PATH}")

            # List collections to verify connection
            collections = self._client.get_collections()
            logger.info(f"Qdrant connected successfully. Found {len(collections.collections)} collections.")

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            raise

    @property
    def client(self):
        """Get Qdrant client."""
        return self._client

    def create_collection(self, collection_name: str = None, force: bool = False) -> Dict:
        """
        Create Qdrant collection for document chunks with hybrid search support.

        Features:
        - Dense vectors: 384-dim embeddings for semantic search
        - Sparse vectors: BM25 tokenization for keyword search
        - HNSW index: Fast approximate nearest neighbor search
        - Payload indexing: Fast filtering on metadata

        Args:
            collection_name: Name of collection (default from settings)
            force: If True, delete existing collection first

        Returns:
            Collection creation response
        """
        collection_name = collection_name or settings.QDRANT_COLLECTION_NAME

        logger.info(f"Creating Qdrant collection: {collection_name}")

        try:
            # Delete if exists and force=True
            if force:
                try:
                    self._client.delete_collection(collection_name=collection_name)
                    logger.info(f"Deleted existing collection: {collection_name}")
                except Exception:
                    pass

            # Check if collection already exists
            if self.collection_exists(collection_name):
                logger.warning(f"Collection already exists: {collection_name}")
                return {"status": "exists"}

            # Map distance metric
            distance_map = {
                "Cosine": Distance.COSINE,
                "Euclidean": Distance.EUCLID,
                "Dot": Distance.DOT
            }
            distance = distance_map.get(settings.QDRANT_DISTANCE_METRIC, Distance.COSINE)

            # Create collection with dense vectors (embeddings)
            vectors_config = VectorParams(
                size=settings.EMBEDDING_DIMENSION,  # 384 for all-MiniLM-L6-v2
                distance=distance,
                on_disk=False  # Keep in memory for faster search (POC)
            )

            # Configure sparse vectors for BM25 keyword search
            sparse_vectors_config = None
            if settings.QDRANT_USE_SPARSE_VECTORS:
                sparse_vectors_config = {
                    "text": SparseVectorParams(
                        index=SparseIndexParams(
                            on_disk=False
                        )
                    )
                }

            # Create collection
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config
            )

            # Create payload indexes for fast filtering
            # Index document_id for filtering
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="document_id",
                field_schema="keyword"
            )

            # Index filename for faceted search
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="filename",
                field_schema="keyword"
            )

            # Index file_type for filtering
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="file_type",
                field_schema="keyword"
            )

            logger.info(f"Collection created successfully: {collection_name}")
            return {"status": "created", "collection": collection_name}

        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            raise

    def collection_exists(self, collection_name: str = None) -> bool:
        """Check if collection exists."""
        collection_name = collection_name or settings.QDRANT_COLLECTION_NAME

        try:
            collections = self._client.get_collections()
            return any(c.name == collection_name for c in collections.collections)
        except Exception:
            return False

    def _generate_sparse_vector(self, text: str) -> SparseVector:
        """
        Generate sparse vector for BM25 keyword search with improved tokenization.

        Improvements over simple version:
        - Proper tokenization (handles punctuation)
        - Stopword removal
        - Stemming for better matching
        - More robust term frequency calculation
        """
        try:
            import re
            from collections import Counter

            # Tokenize: lowercase, remove punctuation, split
            # Remove special characters but keep alphanumeric and spaces
            text_clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
            tokens = text_clean.split()

            # Remove very short tokens (likely noise)
            tokens = [t for t in tokens if len(t) > 2]

            # Simple stopword removal (most common English stopwords)
            stopwords = {
                'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
                'in', 'with', 'to', 'for', 'of', 'as', 'by', 'that', 'this',
                'it', 'from', 'are', 'was', 'were', 'been', 'be', 'have', 'has',
                'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could',
                'may', 'might', 'must', 'can', 'am', 'he', 'she', 'they', 'we',
                'you', 'me', 'him', 'her', 'them', 'us', 'his', 'its', 'our',
                'their', 'what', 'when', 'where', 'who', 'why', 'how'
            }
            tokens = [t for t in tokens if t not in stopwords]

            # Simple stemming (remove common suffixes)
            def simple_stem(word):
                """Very basic stemming - removes common suffixes."""
                if word.endswith('ing'):
                    return word[:-3]
                elif word.endswith('ed'):
                    return word[:-2]
                elif word.endswith('es'):
                    return word[:-2]
                elif word.endswith('s'):
                    return word[:-1]
                return word

            tokens = [simple_stem(t) for t in tokens]

            # Count term frequencies
            term_freq = Counter(tokens)

            # Convert to sparse vector format
            indices = []
            values = []

            for token, freq in term_freq.items():
                # Use consistent hashing for token to index mapping
                token_hash = hash(token) % 100000  # 100k vocabulary size
                indices.append(token_hash)
                # Use TF (term frequency) as value
                values.append(float(freq))

            return SparseVector(
                indices=indices,
                values=values
            )

        except Exception as e:
            logger.warning(f"Sparse vector generation failed: {e}, using fallback")
            # Fallback to simple tokenization
            tokens = text.lower().split()
            term_freq = Counter(tokens)
            indices = [hash(t) % 100000 for t in term_freq.keys()]
            values = [float(f) for f in term_freq.values()]
            return SparseVector(indices=indices, values=values)

    def index_chunks(
        self,
        chunks: List[Dict],
        document_id: str,
        collection_name: str = None
    ) -> Dict:
        """
        Index document chunks into Qdrant with hybrid search support.

        Each chunk is indexed with:
        - Dense vector: Embedding for semantic search
        - Sparse vector: BM25 tokens for keyword search (if enabled)
        - Payload: Metadata (text, document_id, filename, etc.)

        Args:
            chunks: List of chunk dictionaries with text and embeddings
            document_id: Document ID
            collection_name: Collection name

        Returns:
            Indexing statistics
        """
        collection_name = collection_name or settings.QDRANT_COLLECTION_NAME

        logger.info(f"Indexing {len(chunks)} chunks for document {document_id}")

        # Ensure collection exists
        if not self.collection_exists(collection_name):
            logger.info("Collection doesn't exist, creating...")
            self.create_collection(collection_name)

        # Prepare points for indexing
        points = []

        for chunk in chunks:
            # Generate unique UUID for this chunk
            # Qdrant only accepts: (1) valid UUIDs or (2) 64-bit unsigned integers
            point_id = str(uuid.uuid4())

            # Prepare payload (metadata)
            payload = {
                "text": chunk['text'],
                "document_id": document_id,
                "chunk_index": chunk['chunk_index'],
                "char_count": chunk['char_count'],
                "word_count": chunk['word_count'],
                # Store original chunk reference for tracking
                "original_chunk_id": f"{document_id}_{chunk['chunk_index']}"
            }

            # Add optional fields if present
            if 'filename' in chunk:
                payload['filename'] = chunk['filename']
            if 'file_type' in chunk:
                payload['file_type'] = chunk['file_type']
            if 'page_number' in chunk:
                payload['page_number'] = chunk['page_number']

            # Get dense embedding vector
            if 'embedding' not in chunk:
                logger.warning(f"Chunk {point_id} missing embedding, skipping")
                continue

            dense_vector = chunk['embedding']

            # Generate sparse vector for BM25 keyword search
            sparse_vector = None
            if settings.QDRANT_USE_SPARSE_VECTORS:
                sparse_vector = self._generate_sparse_vector(chunk['text'])

            # Create point
            point = PointStruct(
                id=point_id,
                vector={
                    "": dense_vector,  # Default dense vector
                    **({"text": sparse_vector} if sparse_vector else {})
                },
                payload=payload
            )

            points.append(point)

        try:
            # Batch upsert (insert or update)
            self._client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True  # Wait for indexing to complete
            )

            logger.info(f"Successfully indexed {len(points)} chunks")

            return {
                "total_chunks": len(chunks),
                "successful": len(points),
                "failed": len(chunks) - len(points),
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
        Search documents using hybrid search (vector + BM25 keyword).

        Hybrid search strategy:
        1. Vector search: Semantic similarity using embeddings
        2. BM25 search: Keyword matching using sparse vectors
        3. Fusion: Combine results using RRF (Reciprocal Rank Fusion)

        Args:
            query: Search query
            collection_name: Collection to search
            limit: Number of results
            filter_by: Filter expression (e.g., "document_id:=abc123")
            hybrid_search: Use both vector and keyword search

        Returns:
            Search results with scores
        """
        collection_name = collection_name or settings.QDRANT_COLLECTION_NAME

        logger.info(f"Searching: '{query}' in {collection_name}")

        # Build filter if provided
        query_filter = None
        if filter_by:
            # Parse filter string (format: "field:=value")
            if ":=" in filter_by:
                field, value = filter_by.split(":=", 1)
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key=field,
                            match=MatchValue(value=value)
                        )
                    ]
                )

        try:
            # Generate query embedding for vector search
            from app.services.embeddings import embedding_service
            query_embedding = embedding_service.encode_single(query)

            if hybrid_search and settings.QDRANT_USE_SPARSE_VECTORS:
                # Hybrid search: Vector + BM25 with RRF fusion

                # Generate sparse vector for keyword search
                query_sparse = self._generate_sparse_vector(query)

                # Use query API for hybrid search with fusion
                search_result = self._client.query_points(
                    collection_name=collection_name,
                    query=query_embedding,  # Dense vector query
                    using="",  # Default dense vector
                    limit=limit,
                    query_filter=query_filter,
                    with_payload=True,
                    prefetch=[
                        Prefetch(
                            query=query_sparse,
                            using="text",  # Sparse vector name
                            limit=limit * 2  # Fetch more for fusion
                        )
                    ]
                )

                # Convert to standard format
                results = search_result.points if hasattr(search_result, 'points') else search_result

            else:
                # Vector-only search
                results = self._client.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    limit=limit,
                    query_filter=query_filter,
                    with_payload=True
                )

            # Format results
            hits = []
            for result in results:
                hit = {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                }
                hits.append(hit)

            logger.info(f"Found {len(hits)} results")

            return {
                "found": len(hits),
                "hits": hits,
                "query": query,
                "search_time_ms": 0  # Qdrant doesn't provide timing in response
            }

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
        collection_name = collection_name or settings.QDRANT_COLLECTION_NAME

        logger.info(f"Deleting chunks for document: {document_id}")

        try:
            # Delete by filter
            result = self._client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                ),
                wait=True
            )

            logger.info(f"Deleted chunks for document: {document_id}")
            return {
                "status": "deleted",
                "document_id": document_id
            }

        except Exception as e:
            logger.error(f"Deletion failed: {str(e)}")
            raise

    def get_collection_stats(self, collection_name: str = None) -> Dict:
        """Get collection statistics."""
        collection_name = collection_name or settings.QDRANT_COLLECTION_NAME

        try:
            collection_info = self._client.get_collection(collection_name=collection_name)

            return {
                "name": collection_name,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status,
                "config": {
                    "distance": str(collection_info.config.params.vectors.distance),
                    "vector_size": collection_info.config.params.vectors.size
                }
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {}


# Global instance
qdrant_service = QdrantService()


def index_document_chunks(chunks: List[Dict], document_id: str) -> Dict:
    """
    Convenience function to index document chunks.

    Args:
        chunks: List of chunks with embeddings
        document_id: Document ID

    Returns:
        Indexing statistics
    """
    return qdrant_service.index_chunks(chunks, document_id)
