"""
Knowledge Graph Builder

Builds a NetworkX graph from structured deed data following the defined schema.
"""

import networkx as nx
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict

# Handle both package and direct imports
try:
    from .schema import NodeType, EdgeType, DEED_KG_SCHEMA
except ImportError:
    from schema import NodeType, EdgeType, DEED_KG_SCHEMA


class KnowledgeGraphBuilder:
    """
    Builds a knowledge graph from structured deed data.
    
    The graph encodes:
    - Entity nodes (deeds, streets, subdivisions, persons, time points)
    - Spatial relationships (deed->street->subdivision->town->county)
    - Temporal relationships (deed->time_point, time_point->time_point precedence)
    - Party relationships (person->deed as grantor/grantee)
    - Derived relationships (deed<->deed via shared streets)
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_counts = defaultdict(int)
        self.edge_counts = defaultdict(int)
    
    def build(self, data: Dict[str, Any]) -> nx.DiGraph:
        """
        Build complete knowledge graph from deed data.
        
        Args:
            data: Output from SyntheticDeedGenerator.generate()
        
        Returns:
            NetworkX directed graph
        """
        self.graph = nx.DiGraph()
        self.node_counts = defaultdict(int)
        self.edge_counts = defaultdict(int)
        
        # Build nodes and edges in order
        self._add_county_nodes(data)
        self._add_town_nodes(data)
        self._add_subdivision_nodes(data)
        self._add_street_nodes(data)
        self._add_person_nodes(data)
        self._add_time_nodes(data)
        self._add_deed_nodes(data)
        
        # Add relationships
        self._add_spatial_edges(data)
        self._add_temporal_edges(data)
        self._add_party_edges(data)
        self._add_shared_street_edges(data)
        
        return self.graph
    
    def _add_county_nodes(self, data: Dict[str, Any]):
        """Add county nodes."""
        counties = set()
        for sub in data.get('subdivisions', {}).values():
            counties.add(sub.get('county', 'Unknown'))
        
        for county in counties:
            node_id = f"county_{county.lower().replace(' ', '_')}"
            self.graph.add_node(
                node_id,
                node_type=NodeType.COUNTY.value,
                name=county
            )
            self.node_counts[NodeType.COUNTY] += 1
    
    def _add_town_nodes(self, data: Dict[str, Any]):
        """Add town nodes and connect to counties."""
        towns = {}  # name -> county
        for sub in data.get('subdivisions', {}).values():
            town = sub.get('town')
            county = sub.get('county', 'Unknown')
            if town:
                towns[town] = county
        
        for town, county in towns.items():
            node_id = f"town_{town.lower().replace(' ', '_')}"
            self.graph.add_node(
                node_id,
                node_type=NodeType.TOWN.value,
                name=town
            )
            self.node_counts[NodeType.TOWN] += 1
            
            # Connect to county
            county_id = f"county_{county.lower().replace(' ', '_')}"
            if self.graph.has_node(county_id):
                self.graph.add_edge(
                    node_id, county_id,
                    edge_type=EdgeType.IN_COUNTY.value
                )
                self.edge_counts[EdgeType.IN_COUNTY] += 1
    
    def _add_subdivision_nodes(self, data: Dict[str, Any]):
        """Add subdivision nodes and connect to towns."""
        for sub_id, sub in data.get('subdivisions', {}).items():
            self.graph.add_node(
                sub_id,
                node_type=NodeType.SUBDIVISION.value,
                name=sub['name'],
                established_year=sub.get('established_year')
            )
            self.node_counts[NodeType.SUBDIVISION] += 1
            
            # Connect to town
            town = sub.get('town')
            if town:
                town_id = f"town_{town.lower().replace(' ', '_')}"
                if self.graph.has_node(town_id):
                    self.graph.add_edge(
                        sub_id, town_id,
                        edge_type=EdgeType.IN_TOWN.value
                    )
                    self.edge_counts[EdgeType.IN_TOWN] += 1
    
    def _add_street_nodes(self, data: Dict[str, Any]):
        """Add street nodes and connect to subdivisions."""
        for street_id, street in data.get('streets', {}).items():
            self.graph.add_node(
                street_id,
                node_type=NodeType.STREET.value,
                name=street['name'],
                subdivision_id=street.get('subdivision_id')
            )
            self.node_counts[NodeType.STREET] += 1
            
            # Connect to subdivision
            sub_id = street.get('subdivision_id')
            if sub_id and self.graph.has_node(sub_id):
                self.graph.add_edge(
                    street_id, sub_id,
                    edge_type=EdgeType.IN_SUBDIVISION.value
                )
                self.edge_counts[EdgeType.IN_SUBDIVISION] += 1
    
    def _add_person_nodes(self, data: Dict[str, Any]):
        """Add person nodes."""
        for person_id, person in data.get('persons', {}).items():
            self.graph.add_node(
                person_id,
                node_type=NodeType.PERSON.value,
                name=person['name'],
                first_name=person.get('first_name'),
                last_name=person.get('last_name')
            )
            self.node_counts[NodeType.PERSON] += 1
    
    def _add_time_nodes(self, data: Dict[str, Any]):
        """Add time point nodes for all dates in the data."""
        dates = set()
        
        for deed in data.get('deeds', {}).values():
            if deed.get('signed_date'):
                dates.add(deed['signed_date'])
            if deed.get('recorded_date'):
                dates.add(deed['recorded_date'])
        
        for date_str in dates:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                node_id = f"time_{date_str}"
                
                self.graph.add_node(
                    node_id,
                    node_type=NodeType.TIME_POINT.value,
                    date=date_str,
                    year=dt.year,
                    month=dt.month,
                    day=dt.day,
                    decade=(dt.year // 10) * 10
                )
                self.node_counts[NodeType.TIME_POINT] += 1
            except ValueError:
                continue
    
    def _add_deed_nodes(self, data: Dict[str, Any]):
        """Add deed nodes."""
        for deed_id, deed in data.get('deeds', {}).items():
            self.graph.add_node(
                deed_id,
                node_type=NodeType.DEED.value,
                signed_date=deed.get('signed_date'),
                signed_year=deed.get('signed_year'),
                recorded_date=deed.get('recorded_date'),
                has_covenant=deed.get('has_covenant', False),
                covenant_text=deed.get('covenant_text'),
                plan_book=deed.get('plan_book'),
                plan_page=deed.get('plan_page'),
                review_status=deed.get('review_status'),
                has_date_conflict=deed.get('has_date_conflict', False),
                has_review_conflict=deed.get('has_review_conflict', False)
            )
            self.node_counts[NodeType.DEED] += 1
    
    def _add_spatial_edges(self, data: Dict[str, Any]):
        """Add deed->street edges."""
        for deed_id, deed in data.get('deeds', {}).items():
            street_id = deed.get('street_id')
            if street_id and self.graph.has_node(street_id):
                self.graph.add_edge(
                    deed_id, street_id,
                    edge_type=EdgeType.MENTIONS_STREET.value
                )
                self.edge_counts[EdgeType.MENTIONS_STREET] += 1
    
    def _add_temporal_edges(self, data: Dict[str, Any]):
        """Add deed->time_point edges and time precedence edges."""
        # Deed -> TimePoint edges
        for deed_id, deed in data.get('deeds', {}).items():
            signed_date = deed.get('signed_date')
            if signed_date:
                time_id = f"time_{signed_date}"
                if self.graph.has_node(time_id):
                    self.graph.add_edge(
                        deed_id, time_id,
                        edge_type=EdgeType.SIGNED_ON.value
                    )
                    self.edge_counts[EdgeType.SIGNED_ON] += 1
            
            recorded_date = deed.get('recorded_date')
            if recorded_date:
                time_id = f"time_{recorded_date}"
                if self.graph.has_node(time_id):
                    self.graph.add_edge(
                        deed_id, time_id,
                        edge_type=EdgeType.RECORDED_ON.value
                    )
                    self.edge_counts[EdgeType.RECORDED_ON] += 1
        
        # Time precedence edges (only for same-year dates to avoid explosion)
        time_nodes = [n for n, d in self.graph.nodes(data=True) 
                     if d.get('node_type') == NodeType.TIME_POINT.value]
        
        # Group by year for efficiency
        by_year = defaultdict(list)
        for node_id in time_nodes:
            year = self.graph.nodes[node_id].get('year')
            if year:
                by_year[year].append(node_id)
        
        # Add PRECEDES edges within each year
        for year, nodes in by_year.items():
            sorted_nodes = sorted(nodes, key=lambda n: self.graph.nodes[n]['date'])
            for i in range(len(sorted_nodes) - 1):
                self.graph.add_edge(
                    sorted_nodes[i], sorted_nodes[i + 1],
                    edge_type=EdgeType.PRECEDES.value
                )
                self.edge_counts[EdgeType.PRECEDES] += 1
    
    def _add_party_edges(self, data: Dict[str, Any]):
        """Add person->deed edges for grantors and grantees."""
        for deed_id, deed in data.get('deeds', {}).items():
            grantor_id = deed.get('grantor_id')
            if grantor_id and self.graph.has_node(grantor_id):
                self.graph.add_edge(
                    grantor_id, deed_id,
                    edge_type=EdgeType.GRANTOR_OF.value
                )
                self.edge_counts[EdgeType.GRANTOR_OF] += 1
            
            grantee_id = deed.get('grantee_id')
            if grantee_id and self.graph.has_node(grantee_id):
                self.graph.add_edge(
                    grantee_id, deed_id,
                    edge_type=EdgeType.GRANTEE_OF.value
                )
                self.edge_counts[EdgeType.GRANTEE_OF] += 1
    
    def _add_shared_street_edges(self, data: Dict[str, Any]):
        """Add deed<->deed edges for deeds sharing streets."""
        # Group deeds by street
        deeds_by_street = defaultdict(list)
        for deed_id, deed in data.get('deeds', {}).items():
            street_id = deed.get('street_id')
            if street_id:
                deeds_by_street[street_id].append(deed_id)
        
        # Add edges between deeds sharing streets
        for street_id, deed_ids in deeds_by_street.items():
            for i, deed1 in enumerate(deed_ids):
                for deed2 in deed_ids[i + 1:]:
                    # Add bidirectional edges
                    self.graph.add_edge(
                        deed1, deed2,
                        edge_type=EdgeType.SHARES_STREET.value,
                        street_id=street_id
                    )
                    self.graph.add_edge(
                        deed2, deed1,
                        edge_type=EdgeType.SHARES_STREET.value,
                        street_id=street_id
                    )
                    self.edge_counts[EdgeType.SHARES_STREET] += 2
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_counts": {k.value: v for k, v in self.node_counts.items()},
            "edge_counts": {k.value: v for k, v in self.edge_counts.items()}
        }
    
    def save_graphml(self, path: str):
        """Save graph to GraphML format."""
        nx.write_graphml(self.graph, path)
    
    @classmethod
    def load_graphml(cls, path: str) -> nx.DiGraph:
        """Load graph from GraphML format."""
        return nx.read_graphml(path)


class GraphQueryEngine:
    """
    Query engine for the deed knowledge graph.
    
    Provides methods for common query patterns used by Graph RAG.
    """
    
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
    
    def get_deeds_by_year(self, year: int) -> List[str]:
        """Get all deeds signed in a specific year."""
        return [
            n for n, d in self.graph.nodes(data=True)
            if d.get('node_type') == NodeType.DEED.value
            and d.get('signed_year') == year
        ]
    
    def get_deeds_in_year_range(self, start_year: int, end_year: int) -> List[str]:
        """Get all deeds signed within a year range."""
        return [
            n for n, d in self.graph.nodes(data=True)
            if d.get('node_type') == NodeType.DEED.value
            and start_year <= d.get('signed_year', 0) <= end_year
        ]
    
    def get_deeds_by_subdivision(self, subdivision_id: str) -> List[str]:
        """Get all deeds in a subdivision (via street relationships)."""
        # Find streets in subdivision
        streets = [
            n for n, d in self.graph.nodes(data=True)
            if d.get('node_type') == NodeType.STREET.value
            and d.get('subdivision_id') == subdivision_id
        ]
        
        # Find deeds mentioning these streets
        deeds = set()
        for street_id in streets:
            for pred in self.graph.predecessors(street_id):
                if self.graph.nodes[pred].get('node_type') == NodeType.DEED.value:
                    deeds.add(pred)
        
        return list(deeds)
    
    def get_street_neighbors(self, deed_id: str) -> List[str]:
        """Get deeds that share a street with the given deed."""
        neighbors = set()
        
        for _, target, data in self.graph.edges(deed_id, data=True):
            if data.get('edge_type') == EdgeType.SHARES_STREET.value:
                neighbors.add(target)
        
        return list(neighbors)
    
    def get_covenants_in_subdivision_during_decade(
        self, subdivision_id: str, decade: int
    ) -> List[str]:
        """Get deeds with covenants in a subdivision during a decade."""
        # Get all deeds in subdivision
        deeds_in_sub = self.get_deeds_by_subdivision(subdivision_id)
        
        # Filter by decade and covenant
        matching = []
        for deed_id in deeds_in_sub:
            deed_data = self.graph.nodes[deed_id]
            year = deed_data.get('signed_year', 0)
            has_covenant = deed_data.get('has_covenant', False)
            
            if has_covenant and decade <= year < decade + 10:
                matching.append(deed_id)
        
        return matching
    
    def get_conflict_deeds(self, conflict_type: str = "all") -> List[str]:
        """Get deeds with conflicts."""
        conflicts = []
        
        for n, d in self.graph.nodes(data=True):
            if d.get('node_type') != NodeType.DEED.value:
                continue
            
            if conflict_type == "date" and d.get('has_date_conflict'):
                conflicts.append(n)
            elif conflict_type == "review" and d.get('has_review_conflict'):
                conflicts.append(n)
            elif conflict_type == "all":
                if d.get('has_date_conflict') or d.get('has_review_conflict'):
                    conflicts.append(n)
        
        return conflicts


if __name__ == "__main__":
    # Test with synthetic data
    import sys
    sys.path.insert(0, '..')
    from data.synthetic_generator import SyntheticDeedGenerator, GeneratorConfig
    
    config = GeneratorConfig(num_deeds=20)
    generator = SyntheticDeedGenerator(config)
    data = generator.generate()
    
    builder = KnowledgeGraphBuilder()
    graph = builder.build(data)
    
    stats = builder.get_stats()
    print("=== Graph Statistics ===")
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Total edges: {stats['total_edges']}")
    print("\nNode counts:")
    for node_type, count in stats['node_counts'].items():
        print(f"  {node_type}: {count}")
    print("\nEdge counts:")
    for edge_type, count in stats['edge_counts'].items():
        print(f"  {edge_type}: {count}")
    
    # Test queries
    engine = GraphQueryEngine(graph)
    deeds_1924 = engine.get_deeds_by_year(1924)
    print(f"\nDeeds in 1924: {len(deeds_1924)}")
