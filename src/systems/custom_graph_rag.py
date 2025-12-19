"""
Custom Graph RAG Implementation

Graph-based retrieval-augmented generation with explicit spatio-temporal schema.
Includes V1 (basic) and V2 (enhanced query parsing) versions.
"""

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from enum import Enum

# Handle both package and direct imports
try:
    from .base import BaseRAGSystem, RAGConfig, RAGResponse, MockLLM, OpenAILLM
except ImportError:
    from base import BaseRAGSystem, RAGConfig, RAGResponse, MockLLM, OpenAILLM


class QueryType(Enum):
    """Types of queries the system can handle."""
    TEMPORAL = "temporal"           # Questions about time
    SPATIAL = "spatial"             # Questions about location
    SPATIOTEMPORAL = "spatiotemporal"  # Combined time + location
    ENTITY = "entity"               # Direct entity lookup
    CONFLICT = "conflict"           # Data conflict detection
    UNKNOWN = "unknown"


@dataclass
class ParsedQuery:
    """Result of query parsing."""
    query_type: QueryType
    temporal_constraints: Dict[str, Any]  # {start_year, end_year, specific_year, decade}
    spatial_constraints: Dict[str, Any]   # {subdivision, street, town}
    entity_references: List[str]          # Specific entity IDs mentioned
    is_count_query: bool = False          # "How many..."
    is_list_query: bool = True            # "List all...", "Find..."


@dataclass 
class CustomGraphRAGConfig(RAGConfig):
    """Configuration for Custom Graph RAG."""
    name: str = "custom_graph_rag"
    version: str = "v2"  # v1 or v2
    use_llm_answer: bool = True


class QueryParser:
    """
    Parses natural language queries to extract constraints.
    
    V1: Basic regex patterns
    V2: Enhanced parsing with better temporal range handling
    """
    
    # Temporal patterns
    YEAR_PATTERN = r'\b(1[89]\d{2}|20[0-2]\d)\b'
    DECADE_PATTERN = r'\b(1[89]\d0|19[0-4]0)s\b'
    RANGE_PATTERN = r'between\s+(\d{4})\s+and\s+(\d{4})'
    BEFORE_PATTERN = r'before\s+(\d{4})'
    AFTER_PATTERN = r'after\s+(\d{4})'
    DURING_PATTERN = r'during\s+the\s+(\d{4})s'
    
    # Spatial patterns  
    SUBDIVISION_KEYWORDS = ["subdivision", "in", "within"]
    STREET_KEYWORDS = ["street", "avenue", "road", "lane", "on"]
    
    # Query type indicators
    COUNT_PATTERNS = [r'how many', r'count', r'number of']
    CONFLICT_PATTERNS = [r'conflict', r'inconsisten', r'error', r'invalid', r'incorrect']
    
    def __init__(self, version: str = "v2", data: Optional[Dict] = None):
        self.version = version
        self.data = data or {}
        
        # Build lookup tables from data
        self.subdivision_names = set()
        self.street_names = set()
        
        if data:
            for sub in data.get('subdivisions', {}).values():
                self.subdivision_names.add(sub['name'].lower())
            for street in data.get('streets', {}).values():
                self.street_names.add(street['name'].lower())
    
    def parse(self, query: str) -> ParsedQuery:
        """Parse a natural language query."""
        query_lower = query.lower()
        
        # Determine query type
        query_type = self._detect_query_type(query_lower)
        
        # Extract constraints
        temporal = self._extract_temporal_constraints(query_lower)
        spatial = self._extract_spatial_constraints(query_lower)
        entities = self._extract_entity_references(query)
        
        # Check for count query
        is_count = any(re.search(p, query_lower) for p in self.COUNT_PATTERNS)
        
        return ParsedQuery(
            query_type=query_type,
            temporal_constraints=temporal,
            spatial_constraints=spatial,
            entity_references=entities,
            is_count_query=is_count,
            is_list_query=not is_count
        )
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect the type of query."""
        # Check for conflict queries
        if any(re.search(p, query) for p in self.CONFLICT_PATTERNS):
            return QueryType.CONFLICT
        
        has_temporal = bool(re.search(self.YEAR_PATTERN, query) or 
                          re.search(self.DECADE_PATTERN, query) or
                          re.search(self.RANGE_PATTERN, query))
        
        has_spatial = any(kw in query for kw in self.SUBDIVISION_KEYWORDS + self.STREET_KEYWORDS)
        has_spatial = has_spatial or any(name in query for name in self.subdivision_names)
        
        if has_temporal and has_spatial:
            return QueryType.SPATIOTEMPORAL
        elif has_temporal:
            return QueryType.TEMPORAL
        elif has_spatial:
            return QueryType.SPATIAL
        else:
            return QueryType.ENTITY
    
    def _extract_temporal_constraints(self, query: str) -> Dict[str, Any]:
        """Extract temporal constraints from query."""
        constraints = {}
        
        # Check for year range
        range_match = re.search(self.RANGE_PATTERN, query)
        if range_match:
            constraints['start_year'] = int(range_match.group(1))
            constraints['end_year'] = int(range_match.group(2))
            return constraints
        
        # Check for decade (V2 enhanced)
        if self.version == "v2":
            decade_match = re.search(self.DECADE_PATTERN, query)
            if decade_match:
                decade = int(decade_match.group(1).replace('s', ''))
                constraints['start_year'] = decade
                constraints['end_year'] = decade + 9
                constraints['decade'] = decade
                return constraints
            
            # Check "during the 1920s" pattern
            during_match = re.search(self.DURING_PATTERN, query)
            if during_match:
                decade = int(during_match.group(1))
                constraints['start_year'] = decade
                constraints['end_year'] = decade + 9
                constraints['decade'] = decade
                return constraints
        
        # Check before/after
        before_match = re.search(self.BEFORE_PATTERN, query)
        if before_match:
            constraints['end_year'] = int(before_match.group(1)) - 1
        
        after_match = re.search(self.AFTER_PATTERN, query)
        if after_match:
            constraints['start_year'] = int(after_match.group(1)) + 1
        
        # Check for specific year
        year_matches = re.findall(self.YEAR_PATTERN, query)
        if year_matches and 'start_year' not in constraints and 'end_year' not in constraints:
            constraints['specific_year'] = int(year_matches[0])
        
        return constraints
    
    def _extract_spatial_constraints(self, query: str) -> Dict[str, Any]:
        """Extract spatial constraints from query."""
        constraints = {}
        query_lower = query.lower()
        
        # Check for subdivision names
        for name in self.subdivision_names:
            if name in query_lower:
                constraints['subdivision_name'] = name
                # Find subdivision ID
                for sub_id, sub in self.data.get('subdivisions', {}).items():
                    if sub['name'].lower() == name:
                        constraints['subdivision_id'] = sub_id
                        break
                break
        
        # Check for street names
        for name in self.street_names:
            if name in query_lower:
                constraints['street_name'] = name
                break
        
        return constraints
    
    def _extract_entity_references(self, query: str) -> List[str]:
        """Extract specific entity IDs (e.g., deed_0001) from query."""
        # Match deed IDs
        deed_matches = re.findall(r'deed_\d+', query, re.IGNORECASE)
        return [d.lower() for d in deed_matches]


class CustomGraphRAG(BaseRAGSystem):
    """
    Custom Graph RAG with explicit spatio-temporal schema.
    
    Key differences from generic Graph RAG:
    1. Domain-specific schema (Deed, Street, Subdivision, TimePoint)
    2. Explicit temporal relationships (PRECEDES, SIGNED_ON)
    3. Explicit spatial relationships (MENTIONS_STREET, IN_SUBDIVISION)
    4. Query parser that understands spatio-temporal constraints
    """
    
    def __init__(self, config: Optional[CustomGraphRAGConfig] = None):
        self.config = config or CustomGraphRAGConfig()
        super().__init__(self.config)
        
        self.graph = None
        self.data = None
        self.query_engine = None
        self.query_parser = None
        
        # LLM for answer generation
        if self.config.use_llm_answer:
            self.llm = OpenAILLM(self.config.llm_model)
        else:
            self.llm = MockLLM()
    
    def index(self, data: Dict[str, Any]) -> None:
        """
        Build knowledge graph from structured data.
        
        Args:
            data: Output from SyntheticDeedGenerator.generate()
        """
        start_time = time.time()
        
        try:
            from ..knowledge_graph.builder import KnowledgeGraphBuilder, GraphQueryEngine
        except ImportError:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from knowledge_graph.builder import KnowledgeGraphBuilder, GraphQueryEngine
        
        self.data = data
        
        # Build graph
        builder = KnowledgeGraphBuilder()
        self.graph = builder.build(data)
        
        # Initialize query engine
        self.query_engine = GraphQueryEngine(self.graph)
        
        # Initialize query parser
        self.query_parser = QueryParser(version=self.config.version, data=data)
        
        self.is_indexed = True
        self.index_time = datetime.now()
        
        stats = builder.get_stats()
        duration = time.time() - start_time
        print(f"[CustomGraphRAG-{self.config.version}] Built graph with {stats['total_nodes']} nodes, {stats['total_edges']} edges in {duration:.2f}s")
    
    def _execute_graph_query(self, parsed: ParsedQuery) -> List[str]:
        """
        Execute query on the knowledge graph based on parsed constraints.
        
        Returns list of matching deed IDs.
        """
        if not self.query_engine:
            return []
        
        results = set()
        
        # Handle different query types
        if parsed.query_type == QueryType.CONFLICT:
            results = set(self.query_engine.get_conflict_deeds("all"))
        
        elif parsed.query_type == QueryType.TEMPORAL:
            temporal = parsed.temporal_constraints
            if 'specific_year' in temporal:
                results = set(self.query_engine.get_deeds_by_year(temporal['specific_year']))
            elif 'start_year' in temporal or 'end_year' in temporal:
                start = temporal.get('start_year', 1900)
                end = temporal.get('end_year', 2000)
                results = set(self.query_engine.get_deeds_in_year_range(start, end))
        
        elif parsed.query_type == QueryType.SPATIAL:
            spatial = parsed.spatial_constraints
            if 'subdivision_id' in spatial:
                results = set(self.query_engine.get_deeds_by_subdivision(spatial['subdivision_id']))
            elif parsed.entity_references:
                # Get neighbors of specified deed
                for deed_id in parsed.entity_references:
                    neighbors = self.query_engine.get_street_neighbors(deed_id)
                    results.update(neighbors)
        
        elif parsed.query_type == QueryType.SPATIOTEMPORAL:
            temporal = parsed.temporal_constraints
            spatial = parsed.spatial_constraints
            
            # Get subdivision + decade (most common L4 pattern)
            if 'subdivision_id' in spatial and ('decade' in temporal or 'start_year' in temporal):
                sub_id = spatial['subdivision_id']
                decade = temporal.get('decade', temporal.get('start_year', 1920))
                
                # Special handling for covenant queries
                results = set(self.query_engine.get_covenants_in_subdivision_during_decade(sub_id, decade))
                
                # If no covenant-specific results, get all deeds matching constraints
                if not results:
                    all_in_sub = self.query_engine.get_deeds_by_subdivision(sub_id)
                    start = temporal.get('start_year', decade)
                    end = temporal.get('end_year', decade + 9)
                    
                    for deed_id in all_in_sub:
                        deed_data = self.graph.nodes.get(deed_id, {})
                        year = deed_data.get('signed_year', 0)
                        if start <= year <= end:
                            results.add(deed_id)
        
        elif parsed.query_type == QueryType.ENTITY:
            # Direct entity lookup
            if parsed.entity_references:
                results = set(parsed.entity_references)
        
        return list(results)
    
    def _format_context(self, deed_ids: List[str]) -> str:
        """Format deed information as context for LLM."""
        if not deed_ids:
            return "No matching deeds found."
        
        context_parts = []
        for deed_id in deed_ids[:20]:  # Limit context size
            deed = self.data.get('deeds', {}).get(deed_id, {})
            if deed:
                parts = [
                    f"Deed: {deed_id}",
                    f"Date: {deed.get('signed_date', 'Unknown')}",
                    f"Location: {deed.get('street_name', 'Unknown')}, {deed.get('subdivision_name', 'Unknown')}",
                    f"Parties: {deed.get('grantor_name', 'Unknown')} → {deed.get('grantee_name', 'Unknown')}"
                ]
                if deed.get('has_covenant'):
                    parts.append(f"Covenant: Yes")
                context_parts.append(" | ".join(parts))
        
        return "\n".join(context_parts)
    
    def query(self, question: str) -> RAGResponse:
        """
        Query the graph RAG system.
        
        Args:
            question: Natural language question
        
        Returns:
            RAGResponse with answer and metadata
        """
        start_time = time.time()
        
        if not self.is_indexed:
            raise RuntimeError("Must call index() before query()")
        
        # Parse query
        parsed = self.query_parser.parse(question)
        
        # Execute graph query
        matching_deeds = self._execute_graph_query(parsed)
        
        # Format context
        context = self._format_context(matching_deeds)
        
        # Generate answer
        if parsed.is_count_query:
            answer = str(len(matching_deeds))
        elif self.config.use_llm_answer and matching_deeds:
            answer = self.llm.complete(question, context)
        else:
            if matching_deeds:
                answer = f"Found {len(matching_deeds)} matching deeds: {', '.join(matching_deeds[:10])}"
                if len(matching_deeds) > 10:
                    answer += f" (and {len(matching_deeds) - 10} more)"
            else:
                answer = "No matching deeds found for the given constraints."
        
        latency = (time.time() - start_time) * 1000
        
        return RAGResponse(
            answer=answer,
            retrieved_context=[context],
            retrieved_ids=matching_deeds,
            metadata={
                "query_type": parsed.query_type.value,
                "temporal_constraints": parsed.temporal_constraints,
                "spatial_constraints": parsed.spatial_constraints,
                "num_results": len(matching_deeds),
                "version": self.config.version
            },
            latency_ms=latency
        )
    
    def reset(self) -> None:
        """Reset the graph."""
        self.graph = None
        self.data = None
        self.query_engine = None
        self.query_parser = None
        self.is_indexed = False
        self.index_time = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = super().get_stats()
        if self.graph:
            stats.update({
                "num_nodes": self.graph.number_of_nodes(),
                "num_edges": self.graph.number_of_edges(),
                "version": self.config.version
            })
        return stats


# Convenience classes for V1 and V2
class CustomGraphRAGV1(CustomGraphRAG):
    """Graph RAG V1 with basic query parsing."""
    def __init__(self, use_llm_answer: bool = True):
        config = CustomGraphRAGConfig(name="custom_graph_rag_v1", version="v1", use_llm_answer=use_llm_answer)
        super().__init__(config)


class CustomGraphRAGV2(CustomGraphRAG):
    """Graph RAG V2 with enhanced query parsing for spatio-temporal queries."""
    def __init__(self, use_llm_answer: bool = True):
        config = CustomGraphRAGConfig(name="custom_graph_rag_v2", version="v2", use_llm_answer=use_llm_answer)
        super().__init__(config)


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, '..')
    from data.synthetic_generator import SyntheticDeedGenerator, GeneratorConfig
    
    # Generate test data
    config = GeneratorConfig(num_deeds=50, seed=42)
    generator = SyntheticDeedGenerator(config)
    data = generator.generate()
    
    # Test V2
    rag = CustomGraphRAGV2(use_llm_answer=False)
    rag.index(data)
    
    print(f"\nStats: {rag.get_stats()}")
    
    # Test queries
    test_questions = [
        "Find all deeds recorded in 1924",
        "List deeds signed between 1920 and 1930",
        "How many covenants in Pine Valley during the 1920s?",
        "Identify deeds with date conflicts"
    ]
    
    for q in test_questions:
        response = rag.query(q)
        print(f"\nQ: {q}")
        print(f"A: {response.answer}")
        print(f"Type: {response.metadata['query_type']}")
        print(f"Found: {len(response.retrieved_ids)} deeds")
