"""
LightRAG Wrapper

Wrapper for the LightRAG framework to provide consistent interface
for comparison with other RAG systems.
"""

import os
import time
import shutil
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Handle both package and direct imports
try:
    from .base import BaseRAGSystem, RAGConfig, RAGResponse
except ImportError:
    from base import BaseRAGSystem, RAGConfig, RAGResponse


@dataclass
class LightRAGConfig(RAGConfig):
    """Configuration for LightRAG."""
    name: str = "lightrag"
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    working_dir: str = "./data/lightrag_cache"
    mode: str = "hybrid"  # naive, local, global, hybrid


class LightRAGWrapper(BaseRAGSystem):
    """
    Wrapper for LightRAG framework.
    
    LightRAG provides:
    - Automatic entity/relationship extraction via LLM
    - Dual-level retrieval (low-level entities + high-level concepts)
    - Multiple retrieval modes (naive, local, global, hybrid)
    
    This wrapper provides a consistent interface for benchmarking.
    """
    
    def __init__(self, config: Optional[LightRAGConfig] = None):
        self.config = config or LightRAGConfig()
        super().__init__(self.config)
        
        self.rag = None
        self.working_dir = Path(self.config.working_dir)
        self._documents_indexed = 0
    
    def _init_lightrag(self):
        """Initialize the LightRAG instance."""
        try:
            from lightrag import LightRAG, QueryParam
            from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
        except ImportError:
            raise ImportError(
                "LightRAG not installed. Install with: pip install lightrag-hku"
            )
        
        # Create working directory
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LightRAG
        self.rag = LightRAG(
            working_dir=str(self.working_dir),
            llm_model_func=gpt_4o_mini_complete,
            embedding_func=openai_embed
        )
        
        return self.rag
    
    def index(self, data: Any) -> None:
        """
        Index documents into LightRAG.
        
        Args:
            data: Either:
                - Dict with 'deeds' key (structured data - will convert to text)
                - Dict mapping doc_id to text
                - List of text strings
                - Single string (all documents concatenated)
        """
        start_time = time.time()
        
        # Initialize LightRAG if needed
        if self.rag is None:
            self._init_lightrag()
        
        # Handle different input formats
        if isinstance(data, str):
            # Single text blob
            texts = [data]
        elif isinstance(data, dict):
            if 'deeds' in data:
                # Structured data - convert to text
                from ..data.text_converter import TextConverter
                converter = TextConverter(style="mixed")
                text_dict = converter.convert_all(data)
                texts = list(text_dict.values())
            else:
                # Already doc_id -> text mapping
                texts = list(data.values())
        elif isinstance(data, list):
            texts = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Insert documents
        self._documents_indexed = len(texts)
        
        # LightRAG insert accepts a single string, so we join with separators
        combined_text = "\n\n---\n\n".join(texts)
        
        print(f"[LightRAG] Inserting {len(texts)} documents...")
        self.rag.insert(combined_text)
        
        self.is_indexed = True
        self.index_time = datetime.now()
        
        duration = time.time() - start_time
        print(f"[LightRAG] Indexed {len(texts)} documents in {duration:.2f}s")
    
    def query(self, question: str, mode: Optional[str] = None) -> RAGResponse:
        """
        Query LightRAG.
        
        Args:
            question: Natural language question
            mode: Override retrieval mode (naive/local/global/hybrid)
        
        Returns:
            RAGResponse with answer and metadata
        """
        if not self.is_indexed:
            raise RuntimeError("Must call index() before query()")
        
        start_time = time.time()
        use_mode = mode or self.config.mode
        
        try:
            from lightrag import QueryParam
            
            # Set up query parameters
            param = QueryParam(mode=use_mode)
            
            # Execute query
            answer = self.rag.query(question, param=param)
            
        except Exception as e:
            answer = f"Error during query: {str(e)}"
        
        latency = (time.time() - start_time) * 1000
        
        return RAGResponse(
            answer=answer,
            retrieved_context=[],  # LightRAG doesn't expose retrieved chunks directly
            retrieved_ids=[],      # Would need to parse from answer
            metadata={
                "mode": use_mode,
                "framework": "lightrag"
            },
            latency_ms=latency
        )
    
    def query_all_modes(self, question: str) -> Dict[str, RAGResponse]:
        """
        Query using all retrieval modes for comparison.
        
        Returns dict mapping mode name to response.
        """
        modes = ["naive", "local", "global", "hybrid"]
        results = {}
        
        for mode in modes:
            results[mode] = self.query(question, mode=mode)
        
        return results
    
    def reset(self) -> None:
        """Reset LightRAG (clear cache)."""
        if self.working_dir.exists():
            shutil.rmtree(self.working_dir)
        
        self.rag = None
        self.is_indexed = False
        self.index_time = None
        self._documents_indexed = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = super().get_stats()
        stats.update({
            "documents_indexed": self._documents_indexed,
            "mode": self.config.mode,
            "working_dir": str(self.working_dir)
        })
        
        # Try to get graph stats if available
        if self.rag is not None:
            try:
                # LightRAG stores graph in working_dir
                graph_file = self.working_dir / "graph_chunk_entity_relation.graphml"
                if graph_file.exists():
                    import networkx as nx
                    graph = nx.read_graphml(graph_file)
                    stats["lightrag_nodes"] = graph.number_of_nodes()
                    stats["lightrag_edges"] = graph.number_of_edges()
            except Exception:
                pass
        
        return stats
    
    def inspect_graph(self) -> Optional[Dict[str, Any]]:
        """
        Inspect the graph built by LightRAG.
        
        Returns dict with graph information if available.
        """
        if not self.is_indexed:
            return None
        
        try:
            import networkx as nx
            
            graph_file = self.working_dir / "graph_chunk_entity_relation.graphml"
            if not graph_file.exists():
                return {"error": "Graph file not found"}
            
            graph = nx.read_graphml(graph_file)
            
            # Collect node types
            node_types = {}
            for node, data in graph.nodes(data=True):
                node_type = data.get('type', 'unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            # Collect edge types
            edge_types = {}
            for _, _, data in graph.edges(data=True):
                edge_type = data.get('type', 'unknown')
                edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
            
            # Sample entities
            sample_entities = list(graph.nodes())[:10]
            
            return {
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "node_types": node_types,
                "edge_types": edge_types,
                "sample_entities": sample_entities
            }
            
        except Exception as e:
            return {"error": str(e)}


# Convenience classes for different modes
class LightRAGNaive(LightRAGWrapper):
    """LightRAG with naive (vector-only) retrieval."""
    def __init__(self):
        config = LightRAGConfig(name="lightrag_naive", mode="naive")
        super().__init__(config)


class LightRAGHybrid(LightRAGWrapper):
    """LightRAG with hybrid retrieval (recommended)."""
    def __init__(self):
        config = LightRAGConfig(name="lightrag_hybrid", mode="hybrid")
        super().__init__(config)


if __name__ == "__main__":
    # Quick test (requires OPENAI_API_KEY)
    import sys
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this test")
        sys.exit(1)
    
    # Create test documents
    test_docs = [
        "Deed deed_0001: On January 15, 1924, John Smith conveyed property at 123 Oak Street in Pine Valley subdivision to Mary Johnson. The deed contains a racial restrictive covenant.",
        "Deed deed_0002: On March 20, 1926, Robert Brown sold property at 456 Maple Avenue in Oak Heights to Sarah Davis. No restrictive covenants present.",
        "Deed deed_0003: On November 5, 1924, William Taylor transferred property at 123 Oak Street in Pine Valley to Helen Wilson. Contains covenant restricting sale to certain races."
    ]
    
    # Test LightRAG
    print("Testing LightRAG...")
    rag = LightRAGHybrid()
    rag.index(test_docs)
    
    print(f"\nStats: {rag.get_stats()}")
    
    # Test query
    response = rag.query("Which properties in Pine Valley have racial covenants?")
    print(f"\nQuery: Which properties in Pine Valley have racial covenants?")
    print(f"Answer: {response.answer}")
    print(f"Latency: {response.latency_ms:.2f}ms")
    
    # Inspect graph
    print(f"\nGraph inspection: {rag.inspect_graph()}")
    
    # Cleanup
    rag.reset()
