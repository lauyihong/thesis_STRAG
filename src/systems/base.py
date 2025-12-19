"""
Base RAG System Interface

Defines the abstract interface that all RAG systems must implement
for consistent evaluation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class RAGResponse:
    """Standardized response from a RAG system."""
    answer: str
    retrieved_context: List[str] = field(default_factory=list)
    retrieved_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass
class RAGConfig:
    """Base configuration for RAG systems."""
    name: str = "base_rag"
    use_llm_answer: bool = True
    llm_model: str = "gpt-4o-mini"
    top_k: int = 5


class BaseRAGSystem(ABC):
    """
    Abstract base class for RAG systems.
    
    All RAG implementations (Vector RAG, LightRAG, Custom Graph RAG)
    must inherit from this class to ensure consistent interface.
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.is_indexed = False
        self.index_time: Optional[datetime] = None
    
    @property
    def name(self) -> str:
        """Return the name of this RAG system."""
        return self.config.name
    
    @abstractmethod
    def index(self, data: Any) -> None:
        """
        Index/ingest data into the RAG system.
        
        Args:
            data: Data to index. Format depends on implementation:
                  - Vector RAG: List of text documents
                  - Graph RAG: Structured data dict or graph
                  - LightRAG: Text documents
        """
        pass
    
    @abstractmethod
    def query(self, question: str) -> RAGResponse:
        """
        Query the RAG system with a natural language question.
        
        Args:
            question: Natural language question
        
        Returns:
            RAGResponse with answer and metadata
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset/clear the RAG system's index."""
        pass
    
    def batch_query(self, questions: List[str]) -> List[RAGResponse]:
        """
        Query the system with multiple questions.
        
        Default implementation queries sequentially.
        Subclasses may override for batch optimization.
        """
        return [self.query(q) for q in questions]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "name": self.name,
            "is_indexed": self.is_indexed,
            "index_time": self.index_time.isoformat() if self.index_time else None
        }


class MockLLM:
    """
    Mock LLM for testing without API calls.
    
    Returns simple pattern-based responses.
    """
    
    def __init__(self):
        pass
    
    def complete(self, prompt: str, context: str = "") -> str:
        """Generate a mock response based on context."""
        # Extract any deed IDs from context
        import re
        deed_ids = re.findall(r'deed_\d+', context)
        
        if deed_ids:
            return f"Based on the provided context, the relevant deeds are: {', '.join(set(deed_ids))}."
        else:
            return "I could not find relevant information to answer this question."


class OpenAILLM:
    """
    OpenAI LLM wrapper for answer generation.
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client
    
    def complete(self, prompt: str, context: str = "", system_prompt: str = None) -> str:
        """Generate a response using OpenAI."""
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant analyzing historical deed documents. "
                "Answer questions based only on the provided context. "
                "If the answer cannot be determined from the context, say so."
            )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=500
        )
        
        return response.choices[0].message.content
