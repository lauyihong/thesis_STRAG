# Paper Notes: STRAG Thesis

This document summarizes the method, experimental setup, metrics, and limitations for the thesis on Spatio-Temporal Graph RAG.

## 1. Research Motivation

Historical legal documents (deeds, contracts, property records) contain rich spatio-temporal relationships that generic RAG systems struggle to capture. This thesis investigates whether explicit spatio-temporal graph modeling improves question-answering performance on such documents.

### Key Insight

Deed documents exhibit:
- **Temporal relationships**: Deeds are signed, recorded, and reference prior transactions
- **Spatial relationships**: Properties share streets, subdivisions, and neighborhoods
- **Entity relationships**: Grantors/grantees appear across multiple transactions

Standard vector retrieval treats documents as independent, losing these structural relationships.

## 2. Method Overview

### 2.1 Systems Compared

| System | Retrieval Method | Graph Type |
|--------|-----------------|------------|
| **Vector RAG** | Embedding similarity | None |
| **LightRAG** | Dual-level (entity + concept) | Auto-extracted |
| **Custom Graph RAG** | Constraint-based traversal | Domain-specific |

### 2.2 Custom Graph RAG Architecture

```
Query → Parser → Constraint Extraction → Graph Traversal → Context → LLM → Answer
```

**Key components:**

1. **Domain-Specific Schema**
   - 7 node types: DEED, STREET, SUBDIVISION, PERSON, TIME_POINT, TOWN, COUNTY
   - 10 edge types: MENTIONS_STREET, IN_SUBDIVISION, PRECEDES, SIGNED_ON, etc.

2. **Query Parser**
   - Extracts temporal constraints (years, decades, ranges)
   - Extracts spatial constraints (subdivisions, streets)
   - Detects query type (temporal, spatial, spatio-temporal, conflict)

3. **Constraint-Based Retrieval**
   - Routes queries to appropriate graph traversal methods
   - Applies temporal/spatial filters before retrieval
   - Supports multi-hop traversal for street-neighbor queries

### 2.3 V1 vs V2 Differences

| Feature | V1 | V2 |
|---------|----|----|
| Decade patterns | Basic (`1920s`) | Enhanced (`during the 1920s`) |
| Temporal ranges | Year only | Year + decade inference |
| Spatial multi-hop | Basic | Improved subdivision matching |

## 3. Experimental Setup

### 3.1 Dataset

- **Synthetic deed corpus**: 100 deeds (configurable)
- **5 subdivisions** with 15 streets each
- **Temporal range**: 1910-1950
- **Injected conflicts**: ~7% date conflicts, ~10% review conflicts
- **Covenant rate**: ~40% of deeds

### 3.2 Benchmark Questions

5 difficulty levels, 10 questions each:

| Level | Description | Example | Ground Truth |
|-------|-------------|---------|--------------|
| L1 | Single-hop | "Find deeds from 1924" | List of deed IDs |
| L2 | Temporal | "Deeds between 1926-1939" | List of deed IDs |
| L3 | Spatial | "Deeds sharing street with deed_0001" | List of deed IDs |
| L4 | Spatio-temporal | "Covenants in Pine Valley 1920s" | List or count |
| L5 | Conflict | "Deeds with date conflicts" | List of deed IDs |

### 3.3 Evaluation Protocol

1. Index all deeds into each system
2. Query with all 50 benchmark questions
3. Compare predicted deed IDs vs ground truth
4. Aggregate metrics by level and overall

## 4. Metrics

### 4.1 Primary Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **F1 Score** | 2 × (P × R) / (P + R) | Balance of precision and recall |
| **Exact Match** | 1.0 if pred == truth else 0.0 | Strict correctness |

Where:
- Precision = |pred ∩ truth| / |pred|
- Recall = |pred ∩ truth| / |truth|

### 4.2 Secondary Metrics

| Metric | Description |
|--------|-------------|
| **Temporal Accuracy** | % of returned deeds satisfying temporal constraints |
| **Spatial Accuracy** | % of returned deeds in correct subdivision/street |
| **Hop Coverage** | Recall for multi-hop questions specifically |

### 4.3 Answer Normalization

Predictions are normalized before comparison:
- Extract deed IDs from text (`deed_\d+` pattern)
- Convert to lowercase set
- Handle count vs list questions separately

## 5. Results

### 5.1 Overall Performance (Toy Dataset)

| System | F1 | Exact Match |
|--------|-----|------------|
| Vector RAG | 0.252 | 0.000 |
| Graph RAG V1 | 0.700 | 0.667 |
| Graph RAG V2 | **0.767** | **0.733** |
| LightRAG Naive | 0.098 | 0.000 |
| LightRAG Hybrid | 0.098 | 0.000 |

### 5.2 Per-Level Breakdown

| Level | Vector | Graph V2 | Improvement |
|-------|--------|----------|-------------|
| L1 | 0.378 | 0.667 | +76% |
| L2 | 0.398 | 1.000 | +151% |
| L3 | 0.095 | 1.000 | +953% |
| L4 | 0.111 | 0.333 | +200% |
| L5 | 0.278 | 0.833 | +200% |

### 5.3 Key Findings

1. **Explicit schema matters**: V2 outperforms V1 on L4 (spatio-temporal) due to better decade handling
2. **Multi-hop is hardest for vectors**: L3 shows largest gap (0.095 vs 1.000)
3. **Conflict detection benefits from flags**: L5 performs well with explicit `has_date_conflict` fields
4. **LightRAG struggles without domain guidance**: Generic graph extraction misses deed-specific patterns

## 6. Limitations

### 6.1 Dataset Limitations

- Synthetic data may not capture real-world complexity
- Limited to single county/town hierarchy
- Covenant text is templated (5 variations)
- Street overlap pattern is controlled (configurable rate)

### 6.2 System Limitations

- **Query Parser**: Regex-based, may miss complex phrasings
- **LLM Dependency**: Final answers still require LLM generation
- **No Semantic Matching**: Exact string matching for subdivision names
- **Fixed Schema**: Requires redesign for new document types

### 6.3 Evaluation Limitations

- F1 on deed ID retrieval only (not answer quality)
- No semantic equivalence checking
- Mock mode results may differ from real LLM behavior
- Small question set per level (10)

## 7. Future Work

1. **Schema Learning**: Automatically infer schema from documents
2. **Hybrid Retrieval**: Combine graph constraints with vector similarity
3. **Temporal Reasoning**: Add duration, recurrence, and event sequences
4. **Real Data Evaluation**: Test on actual historical deed archives
5. **Cross-Document Reasoning**: Chain of transactions over time

## 8. Reproducibility Checklist

- [x] All code in version control
- [x] Fixed random seeds (default: 42)
- [x] Configuration via YAML file
- [x] Mock mode for testing without API
- [x] Deterministic data generation
- [x] Results saved with timestamps
- [x] LaTeX tables auto-generated

## 9. Key Files for Paper

| File | Use |
|------|-----|
| `outputs/results/experiment_results_*.json` | Raw results data |
| `outputs/tables/main_results_table.tex` | Table 1: Main results |
| `outputs/tables/improvement_table.tex` | Table 2: Improvement vs baseline |
| `outputs/figures/overall_comparison.pdf` | Figure 1: Bar chart |
| `outputs/figures/level_heatmap.pdf` | Figure 2: Heatmap |
| `outputs/figures/level_lines.pdf` | Figure 3: Level progression |

## 10. Thesis Outline Mapping

| Thesis Section | Code Location |
|----------------|---------------|
| 3.1 Data Model | `src/knowledge_graph/schema.py` |
| 3.2 Graph Construction | `src/knowledge_graph/builder.py` |
| 3.3 Query Processing | `src/systems/custom_graph_rag.py:QueryParser` |
| 4.1 Benchmark Design | `src/data/benchmark_questions.py` |
| 4.2 Evaluation Metrics | `src/evaluation/metrics.py` |
| 5.1 Results | `outputs/results/` |
| 5.2 Analysis | `scripts/03_analyze_results.py` |
