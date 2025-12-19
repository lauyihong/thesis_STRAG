"""
Vector RAG Implementation

Standard vector-based retrieval-augmented generation baseline.
Uses embedding similarity for document retrieval.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Handle both package and direct imports
try:
    from .base import BaseRAGSystem, RAGConfig, RAGResponse, MockLLM, OpenAILLM
except ImportError:
    from base import BaseRAGSystem, RAGConfig, RAGResponse, MockLLM, OpenAILLM


@dataclass
class VectorRAGConfig(RAGConfig):
    """Configuration for Vector RAG."""
    name: str = "vector_rag"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    chunk_size: int = 500  # Characters
    chunk_overlap: int = 100
    top_k: int = 5
    use_mock_embeddings: bool = False  # For testing without API


class VectorRAG(BaseRAGSystem):
    """
    Vector RAG baseline implementation.
    
    Pipeline:
    1. Split documents into chunks
    2. Embed chunks using OpenAI embeddings (or mock)
    3. For queries: embed query, find top-k similar chunks
    4. Generate answer using LLM with retrieved context
    """
    
    def __init__(self, config: Optional[VectorRAGConfig] = None):
        self.config = config or VectorRAGConfig()
        super().__init__(self.config)
        
        # Storage
        self.chunks: List[Dict[str, Any]] = []  # {text, doc_id, chunk_idx, embedding}
        self.embeddings: Optional[np.ndarray] = None
        
        # LLM for answer generation
        if self.config.use_llm_answer:
            self.llm = OpenAILLM(self.config.llm_model)
        else:
            self.llm = MockLLM()
        
        # Embedding client
        self._embedding_client = None
    
    @property
    def embedding_client(self):
        if self._embedding_client is None and not self.config.use_mock_embeddings:
            from openai import OpenAI
            self._embedding_client = OpenAI()
        return self._embedding_client
    
    def _chunk_text(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = start + self.config.chunk_size
            chunk_text = text[start:end]
            
            if chunk_text.strip():  # Only add non-empty chunks
                chunks.append({
                    "text": chunk_text,
                    "doc_id": doc_id,
                    "chunk_idx": chunk_idx,
                    "start_char": start,
                    "end_char": end
                })
                chunk_idx += 1
            
            start = end - self.config.chunk_overlap
            if start >= len(text) - self.config.chunk_overlap:
                break
        
        return chunks
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        if self.config.use_mock_embeddings:
            # Return random embedding for testing
            np.random.seed(hash(text) % 2**32)
            return np.random.randn(self.config.embedding_dim).astype(np.float32)
        
        response = self.embedding_client.embeddings.create(
            model=self.config.embedding_model,
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    def _get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Get embeddings for multiple texts with batching."""
        if self.config.use_mock_embeddings:
            embeddings = []
            for text in texts:
                np.random.seed(hash(text) % 2**32)
                embeddings.append(np.random.randn(self.config.embedding_dim))
            return np.array(embeddings, dtype=np.float32)
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.embedding_client.embeddings.create(
                model=self.config.embedding_model,
                input=batch
            )
            batch_embeddings = [np.array(d.embedding) for d in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings, dtype=np.float32)
    
    def index(self, data: Any) -> None:
        """
        Index documents into the vector store.
        
        Args:
            data: Either:
                - Dict with 'deeds' key (structured data - will convert to text)
                - Dict mapping doc_id to text
                - List of text strings
        """
        start_time = time.time()
        self.chunks = []
        
        # Handle different input formats
        if isinstance(data, dict):
            if 'deeds' in data:
                # Structured data - convert to text
                try:
                    from ..data.text_converter import TextConverter
                except ImportError:
                    import sys
                    from pathlib import Path
                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    from data.text_converter import TextConverter
                converter = TextConverter()
                texts = converter.convert_all(data)
            else:
                # Already doc_id -> text mapping
                texts = data
        elif isinstance(data, list):
            # List of texts
            texts = {f"doc_{i}": text for i, text in enumerate(data)}
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Chunk all documents
        for doc_id, text in texts.items():
            doc_chunks = self._chunk_text(text, doc_id)
            self.chunks.extend(doc_chunks)
        
        # Get embeddings for all chunks
        chunk_texts = [c["text"] for c in self.chunks]
        self.embeddings = self._get_embeddings_batch(chunk_texts)
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-10)
        
        self.is_indexed = True
        self.index_time = datetime.now()
        
        index_duration = time.time() - start_time
        print(f"[VectorRAG] Indexed {len(texts)} documents, {len(self.chunks)} chunks in {index_duration:.2f}s")
    
    def _retrieve(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Dict, float]]:
        """
        Retrieve top-k most similar chunks for a query.
        
        Returns:
            List of (chunk_dict, similarity_score) tuples
        """
        if not self.is_indexed:
            raise RuntimeError("Must call index() before query()")
        
        k = top_k or self.config.top_k
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        
        # Compute cosine similarities
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(similarities[idx])))
        
        return results
    
    def query(self, question: str) -> RAGResponse:
        """
        Query the vector RAG system.
        
        Args:
            question: Natural language question
        
        Returns:
            RAGResponse with answer and metadata
        """
        start_time = time.time()
        
        # Retrieve relevant chunks
        retrieved = self._retrieve(question)
        
        # Build context from retrieved chunks
        context_parts = []
        retrieved_ids = []
        for chunk, score in retrieved:
            context_parts.append(f"[{chunk['doc_id']}]: {chunk['text']}")
            retrieved_ids.append(chunk['doc_id'])
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        answer = self.llm.complete(question, context)
        
        latency = (time.time() - start_time) * 1000  # Convert to ms
        
        return RAGResponse(
            answer=answer,
            retrieved_context=[c['text'] for c, _ in retrieved],
            retrieved_ids=list(set(retrieved_ids)),  # Unique doc IDs
            metadata={
                "num_chunks_retrieved": len(retrieved),
                "similarity_scores": [s for _, s in retrieved]
            },
            latency_ms=latency
        )
    
    def reset(self) -> None:
        """Reset the vector store."""
        self.chunks = []
        self.embeddings = None
        self.is_indexed = False
        self.index_time = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = super().get_stats()
        stats.update({
            "num_chunks": len(self.chunks),
            "embedding_dim": self.config.embedding_dim,
            "chunk_size": self.config.chunk_size,
            "use_mock": self.config.use_mock_embeddings
        })
        return stats


if __name__ == "__main__":
    # Quick test
    config = VectorRAGConfig(use_mock_embeddings=True, use_llm_answer=False)
    rag = VectorRAG(config)
    
    # Test with simple documents
    docs = {
        "deed_0001": "Deed Record: deed_0001. Date: 1924-03-15. John Smith to Mary Johnson. Property: Oak Street, Pine Valley.",
        "deed_0002": "Deed Record: deed_0002. Date: 1926-07-20. Robert Brown to Sarah Davis. Property: Maple Avenue, Oak Heights.",
        "deed_0003": "Deed Record: deed_0003. Date: 1924-11-05. William Taylor to Helen Wilson. Property: Oak Street, Pine Valley."
    }
    
    rag.index(docs)
    print(f"\nStats: {rag.get_stats()}")
    
    response = rag.query("Find deeds on Oak Street")
    print(f"\nQuery: Find deeds on Oak Street")
    print(f"Answer: {response.answer}")
    print(f"Retrieved IDs: {response.retrieved_ids}")
    print(f"Latency: {response.latency_ms:.2f}ms")
