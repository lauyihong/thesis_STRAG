"""
LightRAG Wrapper

Wrapper for the LightRAG framework to provide consistent interface
for comparison with other RAG systems.

Supports both real LLM calls (requires OPENAI_API_KEY) and mock mode
for testing without API access.
"""

import os
import re
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
    use_mock: bool = False  # Use mock mode (no API calls)


class MockLightRAG:
    """
    Mock LightRAG for testing without API calls.

    Stores documents and returns simple pattern-based responses.
    """

    def __init__(self, working_dir: str):
        self.working_dir = Path(working_dir)
        self.documents: List[str] = []
        self.deed_ids: List[str] = []

    def insert(self, text: str):
        """Store documents for later retrieval."""
        self.documents.append(text)
        # Extract deed IDs from text
        deed_ids = re.findall(r'deed_\d+', text.lower())
        self.deed_ids.extend(deed_ids)
        self.deed_ids = list(set(self.deed_ids))

    def query(self, question: str, param: Any = None) -> str:
        """Generate mock response based on stored documents."""
        question_lower = question.lower()

        # Simple pattern matching for mock responses
        matching_deeds = []

        # Extract year from question
        year_match = re.search(r'\b(19\d{2})\b', question)
        decade_match = re.search(r'(19\d0)s', question_lower)

        # Search documents for matching content
        for doc in self.documents:
            doc_lower = doc.lower()
            deed_ids_in_doc = re.findall(r'deed_\d+', doc_lower)

            # Check year match
            if year_match:
                year = year_match.group(1)
                if year in doc:
                    matching_deeds.extend(deed_ids_in_doc)

            # Check decade match
            elif decade_match:
                decade = int(decade_match.group(1))
                for year in range(decade, decade + 10):
                    if str(year) in doc:
                        matching_deeds.extend(deed_ids_in_doc)
                        break

            # Check for subdivision names
            subdivision_patterns = ['pine valley', 'oak heights', 'maple grove', 'cedar hills']
            for sub in subdivision_patterns:
                if sub in question_lower and sub in doc_lower:
                    matching_deeds.extend(deed_ids_in_doc)

            # Check for conflict queries
            if 'conflict' in question_lower or 'inconsisten' in question_lower:
                if 'conflict' in doc_lower:
                    matching_deeds.extend(deed_ids_in_doc)

        matching_deeds = list(set(matching_deeds))

        if matching_deeds:
            return f"Based on the documents, the relevant deeds are: {', '.join(matching_deeds)}."
        else:
            return "I could not find relevant information to answer this question."


class LightRAGWrapper(BaseRAGSystem):
    """
    Wrapper for LightRAG framework.

    LightRAG provides:
    - Automatic entity/relationship extraction via LLM
    - Dual-level retrieval (low-level entities + high-level concepts)
    - Multiple retrieval modes (naive, local, global, hybrid)

    This wrapper provides a consistent interface for benchmarking.
    Supports mock mode for testing without API access.
    """

    def __init__(self, config: Optional[LightRAGConfig] = None):
        self.config = config or LightRAGConfig()
        super().__init__(self.config)

        self.rag = None
        self.working_dir = Path(self.config.working_dir)
        self._documents_indexed = 0
        self._indexed_data = None  # Store data for deed ID extraction

    def _init_lightrag(self):
        """Initialize the LightRAG instance."""
        if self.config.use_mock:
            self.rag = MockLightRAG(str(self.working_dir))
            return self.rag

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

        # Store data for later deed ID extraction
        self._indexed_data = data

        # Handle different input formats
        if isinstance(data, str):
            # Single text blob
            texts = [data]
        elif isinstance(data, dict):
            if 'deeds' in data:
                # Structured data - convert to text
                try:
                    from ..data.text_converter import TextConverter
                except ImportError:
                    import sys
                    from pathlib import Path
                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    from data.text_converter import TextConverter
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

        mode_str = "(mock)" if self.config.use_mock else ""
        print(f"[LightRAG {mode_str}] Inserting {len(texts)} documents...")
        self.rag.insert(combined_text)

        self.is_indexed = True
        self.index_time = datetime.now()

        duration = time.time() - start_time
        print(f"[LightRAG {mode_str}] Indexed {len(texts)} documents in {duration:.2f}s")
    
    def _extract_deed_ids(self, answer: str) -> List[str]:
        """Extract deed IDs from answer text for schema alignment."""
        deed_ids = re.findall(r'deed_\d+', answer.lower())
        return list(set(deed_ids))

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
            if self.config.use_mock:
                # Mock mode - no need for QueryParam
                answer = self.rag.query(question, param=None)
            else:
                from lightrag import QueryParam

                # Set up query parameters
                param = QueryParam(mode=use_mode)

                # Execute query
                answer = self.rag.query(question, param=param)

        except Exception as e:
            answer = f"Error during query: {str(e)}"

        latency = (time.time() - start_time) * 1000

        # Extract deed IDs from answer for consistent output schema
        retrieved_ids = self._extract_deed_ids(answer)

        return RAGResponse(
            answer=answer,
            retrieved_context=[],  # LightRAG doesn't expose retrieved chunks directly
            retrieved_ids=retrieved_ids,
            metadata={
                "mode": use_mode,
                "framework": "lightrag",
                "mock": self.config.use_mock
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
    def __init__(self, use_mock: bool = False):
        config = LightRAGConfig(name="lightrag_naive", mode="naive", use_mock=use_mock)
        super().__init__(config)


class LightRAGHybrid(LightRAGWrapper):
    """LightRAG with hybrid retrieval (recommended)."""
    def __init__(self, use_mock: bool = False):
        config = LightRAGConfig(name="lightrag_hybrid", mode="hybrid", use_mock=use_mock)
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
