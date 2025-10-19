import re
from typing import List, Dict
from app.core.logging import app_logger as logger
from app.core.config import settings


class DocumentChunker:
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        strategy: str = "sentence-aware"
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.strategy = strategy
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        logger.info(f"Chunking text using '{self.strategy}' strategy")
        logger.info(f"Text length: {len(text)} chars, Target chunk size: {self.chunk_size}")
        
        if self.strategy == "fixed":
            chunks = self._fixed_size_chunks(text)
        elif self.strategy == "sentence-aware":
            chunks = self._sentence_aware_chunks(text)
        elif self.strategy == "paragraph-aware":
            chunks = self._paragraph_aware_chunks(text)
        else:
            logger.warning(f"Unknown strategy '{self.strategy}', using sentence-aware")
            chunks = self._sentence_aware_chunks(text)
        
        # Add metadata to each chunk
        enriched_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk_dict = {
                "chunk_index": i,
                "text": chunk_text,
                "char_count": len(chunk_text),
                "word_count": len(chunk_text.split()),
            }
            
            # Add provided metadata
            if metadata:
                chunk_dict.update(metadata)
            
            enriched_chunks.append(chunk_dict)
        
        logger.info(f"Created {len(enriched_chunks)} chunks")
        return enriched_chunks
    
    def _fixed_size_chunks(self, text: str) -> List[str]:
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            if chunk.strip():
                chunks.append(chunk)
            
            # Move forward by chunk_size - overlap
            start += (self.chunk_size - self.chunk_overlap)
        
        return chunks
    
    def _sentence_aware_chunks(self, text: str) -> List[str]:
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _paragraph_aware_chunks(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            para_length = len(paragraph)
            
            if para_length > self.chunk_size * 1.5:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                para_chunks = self._sentence_aware_chunks(paragraph)
                chunks.extend(para_chunks)
                continue
            
            if current_length + para_length > self.chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                
                if len(current_chunk[-1]) <= self.chunk_overlap:
                    current_chunk = [current_chunk[-1]]
                    current_length = len(current_chunk[-1])
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(paragraph)
            current_length += para_length
        
        # Add final chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def get_chunk_stats(self, chunks: List[Dict]) -> Dict:
        if not chunks:
            return {}
        
        char_counts = [c['char_count'] for c in chunks]
        word_counts = [c['word_count'] for c in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chars_per_chunk": sum(char_counts) / len(char_counts),
            "min_chars": min(char_counts),
            "max_chars": max(char_counts),
            "avg_words_per_chunk": sum(word_counts) / len(word_counts),
            "total_chars": sum(char_counts),
            "total_words": sum(word_counts)
        }


def chunk_document(text: str, document_id: str, filename: str, strategy: str = "sentence-aware") -> Dict:
    chunker = DocumentChunker(strategy=strategy)
    
    metadata = {
        "document_id": document_id,
        "filename": filename
    }
    
    chunks = chunker.chunk_text(text, metadata)
    stats = chunker.get_chunk_stats(chunks)
    
    return {
        "chunks": chunks,
        "stats": stats,
        "strategy": strategy
    }