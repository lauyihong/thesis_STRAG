"""
Knowledge Graph Schema Definition

Defines the node types, edge types, and their properties for the 
spatio-temporal deed knowledge graph.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class NodeType(Enum):
    """Node types in the knowledge graph."""
    DEED = "deed"
    STREET = "street"
    SUBDIVISION = "subdivision"
    PERSON = "person"
    TIME_POINT = "time_point"
    TOWN = "town"
    COUNTY = "county"


class EdgeType(Enum):
    """Edge types (relationships) in the knowledge graph."""
    # Spatial relationships
    MENTIONS_STREET = "mentions_street"       # Deed -> Street
    IN_SUBDIVISION = "in_subdivision"         # Street -> Subdivision
    IN_TOWN = "in_town"                       # Subdivision -> Town
    IN_COUNTY = "in_county"                   # Town -> County
    SHARES_STREET = "shares_street"           # Deed <-> Deed (computed)
    
    # Temporal relationships
    SIGNED_ON = "signed_on"                   # Deed -> TimePoint
    RECORDED_ON = "recorded_on"               # Deed -> TimePoint
    PRECEDES = "precedes"                     # TimePoint -> TimePoint
    
    # Party relationships
    GRANTOR_OF = "grantor_of"                 # Person -> Deed
    GRANTEE_OF = "grantee_of"                 # Person -> Deed


@dataclass
class NodeSchema:
    """Schema definition for a node type."""
    node_type: NodeType
    required_properties: List[str]
    optional_properties: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class EdgeSchema:
    """Schema definition for an edge type."""
    edge_type: EdgeType
    source_type: NodeType
    target_type: NodeType
    properties: List[str] = field(default_factory=list)
    description: str = ""


# Define complete schema
DEED_KG_SCHEMA = {
    "nodes": {
        NodeType.DEED: NodeSchema(
            node_type=NodeType.DEED,
            required_properties=["deed_id", "signed_date", "signed_year"],
            optional_properties=["recorded_date", "has_covenant", "covenant_text", 
                               "plan_book", "plan_page", "review_status"],
            description="A property deed document"
        ),
        NodeType.STREET: NodeSchema(
            node_type=NodeType.STREET,
            required_properties=["street_id", "name"],
            optional_properties=["subdivision_id"],
            description="A street where property is located"
        ),
        NodeType.SUBDIVISION: NodeSchema(
            node_type=NodeType.SUBDIVISION,
            required_properties=["subdivision_id", "name"],
            optional_properties=["established_year"],
            description="A residential subdivision"
        ),
        NodeType.PERSON: NodeSchema(
            node_type=NodeType.PERSON,
            required_properties=["person_id", "name"],
            optional_properties=["first_name", "last_name"],
            description="A party to a deed transaction"
        ),
        NodeType.TIME_POINT: NodeSchema(
            node_type=NodeType.TIME_POINT,
            required_properties=["date", "year"],
            optional_properties=["month", "day", "decade"],
            description="A point in time"
        ),
        NodeType.TOWN: NodeSchema(
            node_type=NodeType.TOWN,
            required_properties=["name"],
            optional_properties=[],
            description="A town or city"
        ),
        NodeType.COUNTY: NodeSchema(
            node_type=NodeType.COUNTY,
            required_properties=["name"],
            optional_properties=[],
            description="A county"
        )
    },
    "edges": {
        EdgeType.MENTIONS_STREET: EdgeSchema(
            edge_type=EdgeType.MENTIONS_STREET,
            source_type=NodeType.DEED,
            target_type=NodeType.STREET,
            description="Deed mentions/references a street"
        ),
        EdgeType.IN_SUBDIVISION: EdgeSchema(
            edge_type=EdgeType.IN_SUBDIVISION,
            source_type=NodeType.STREET,
            target_type=NodeType.SUBDIVISION,
            description="Street is located in subdivision"
        ),
        EdgeType.IN_TOWN: EdgeSchema(
            edge_type=EdgeType.IN_TOWN,
            source_type=NodeType.SUBDIVISION,
            target_type=NodeType.TOWN,
            description="Subdivision is in town"
        ),
        EdgeType.IN_COUNTY: EdgeSchema(
            edge_type=EdgeType.IN_COUNTY,
            source_type=NodeType.TOWN,
            target_type=NodeType.COUNTY,
            description="Town is in county"
        ),
        EdgeType.SHARES_STREET: EdgeSchema(
            edge_type=EdgeType.SHARES_STREET,
            source_type=NodeType.DEED,
            target_type=NodeType.DEED,
            properties=["street_id"],
            description="Two deeds share the same street"
        ),
        EdgeType.SIGNED_ON: EdgeSchema(
            edge_type=EdgeType.SIGNED_ON,
            source_type=NodeType.DEED,
            target_type=NodeType.TIME_POINT,
            description="Deed was signed on date"
        ),
        EdgeType.RECORDED_ON: EdgeSchema(
            edge_type=EdgeType.RECORDED_ON,
            source_type=NodeType.DEED,
            target_type=NodeType.TIME_POINT,
            description="Deed was recorded on date"
        ),
        EdgeType.PRECEDES: EdgeSchema(
            edge_type=EdgeType.PRECEDES,
            source_type=NodeType.TIME_POINT,
            target_type=NodeType.TIME_POINT,
            description="Time point A comes before time point B"
        ),
        EdgeType.GRANTOR_OF: EdgeSchema(
            edge_type=EdgeType.GRANTOR_OF,
            source_type=NodeType.PERSON,
            target_type=NodeType.DEED,
            description="Person is the grantor (seller) of deed"
        ),
        EdgeType.GRANTEE_OF: EdgeSchema(
            edge_type=EdgeType.GRANTEE_OF,
            source_type=NodeType.PERSON,
            target_type=NodeType.DEED,
            description="Person is the grantee (buyer) of deed"
        )
    }
}


def get_schema_summary() -> str:
    """Get a human-readable summary of the schema."""
    lines = ["=== Deed Knowledge Graph Schema ===\n"]
    
    lines.append("NODE TYPES:")
    for node_type, schema in DEED_KG_SCHEMA["nodes"].items():
        lines.append(f"  {node_type.value}: {schema.description}")
        lines.append(f"    Required: {', '.join(schema.required_properties)}")
        if schema.optional_properties:
            lines.append(f"    Optional: {', '.join(schema.optional_properties)}")
    
    lines.append("\nEDGE TYPES:")
    for edge_type, schema in DEED_KG_SCHEMA["edges"].items():
        lines.append(
            f"  {schema.source_type.value} --[{edge_type.value}]--> {schema.target_type.value}"
        )
        lines.append(f"    {schema.description}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    print(get_schema_summary())
