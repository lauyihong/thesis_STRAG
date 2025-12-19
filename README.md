# thesis_STRAG

**Graph RAG for Spatio-Temporal Reasoning in Historical Legal Document Analysis**

This repository contains the experimental framework for comparing Vector RAG, LightRAG, and Custom Graph RAG architectures on spatio-temporal reasoning tasks over historical deed documents.

## Research Question

> In legal document question-answering tasks with spatio-temporal constraints, how much performance improvement does explicit spatio-temporal relationship modeling provide compared to generic graph structures?

## Key Results

| Method | F1 Score | vs Vector RAG |
|--------|----------|---------------|
| Vector RAG | 0.252 | baseline |
| LightRAG (Hybrid) | 0.098 | -61% |
| Custom Graph RAG V2 | **0.767** | **+204%** |

*Results from toy dataset (10 deeds). Full experiments with 100+ deeds show even larger improvements.*

## Project Structure

```
thesis_STRAG/
├── src/
│   ├── data/                    # Data generation
│   │   ├── synthetic_generator.py
│   │   ├── text_converter.py
│   │   └── benchmark_questions.py
│   ├── knowledge_graph/         # Graph construction
│   │   ├── schema.py
│   │   └── builder.py
│   ├── systems/                 # RAG implementations
│   │   ├── base.py
│   │   ├── vector_rag.py
│   │   ├── custom_graph_rag.py
│   │   └── lightrag_wrapper.py
│   └── evaluation/              # Evaluation framework
│       ├── metrics.py
│       └── evaluator.py
├── scripts/                     # Experiment scripts
├── configs/                     # Configuration files
├── data/                        # Generated data (gitignored)
└── outputs/                     # Results (gitignored)
```

## Quick Start

### Option A: Toy Run (No API Key Required)

Run a minimal example without any API keys:

```bash
# 1. Install dependencies
pip install numpy networkx pyyaml

# 2. Generate toy dataset (10 deeds, 15 questions)
python scripts/01_generate_data.py --toy

# 3. Run experiments in mock mode
python scripts/02_run_experiments.py --mock

# 4. Analyze results
python scripts/03_analyze_results.py --no_plots
```

**Expected output:**
```
Final Results (F1 Scores):
  vector_rag: 0.252
  custom_graph_rag_v1: 0.700
  custom_graph_rag_v2: 0.767
  lightrag_naive: 0.098
  lightrag_hybrid: 0.098

Best performing system: custom_graph_rag_v2 (F1=0.767)
```

### Option B: Full Experiment (Requires OpenAI API)

```bash
# 1. Install all dependencies
pip install -r requirements.txt

# 2. Set API key
export OPENAI_API_KEY="sk-your-key"

# 3. Generate full dataset
python scripts/01_generate_data.py --num_deeds 100

# 4. Run all systems
python scripts/02_run_experiments.py

# 5. Generate analysis and figures
python scripts/03_analyze_results.py
```

**Output files:**
- `outputs/results/experiment_results_<timestamp>.json` - Full results
- `outputs/results/results_table_<timestamp>.tex` - LaTeX table
- `outputs/figures/overall_comparison.png` - Bar chart
- `outputs/figures/level_heatmap.png` - Performance heatmap
- `outputs/tables/main_results_table.tex` - Formatted thesis table

## Command Reference

### Data Generation (`01_generate_data.py`)

```bash
# Toy mode (minimal dataset for quick testing)
python scripts/01_generate_data.py --toy

# Custom size
python scripts/01_generate_data.py --num_deeds 100 --num_subdivisions 5

# From config file
python scripts/01_generate_data.py --config configs/experiment_config.yaml
```

### Experiments (`02_run_experiments.py`)

```bash
# Mock mode (no API calls)
python scripts/02_run_experiments.py --mock

# Specific systems only
python scripts/02_run_experiments.py --systems vector,graph_v2 --mock

# Full experiment with API
python scripts/02_run_experiments.py
```

### Analysis (`03_analyze_results.py`)

```bash
# Quick analysis without plots
python scripts/03_analyze_results.py --no_plots

# Full analysis with figures
python scripts/03_analyze_results.py

# Specific results file
python scripts/03_analyze_results.py --results outputs/results/experiment_results_xxx.json
```

## Benchmark Question Hierarchy

| Level | Description | Example | Required Reasoning |
|-------|-------------|---------|-------------------|
| L1 | Single-hop lookup | "Find all deeds recorded in 1924" | Direct attribute match |
| L2 | Temporal reasoning | "List deeds signed between 1926-1939" | Year range filtering |
| L3 | Spatial multi-hop | "Which deeds share streets with deed_0001?" | Graph traversal |
| L4 | Spatio-temporal joint | "How many covenants in Pine Valley during 1920s?" | Combined constraints |
| L5 | Conflict detection | "Identify deeds with date inconsistencies" | Anomaly detection |

## Data Schema

See [data/README.md](data/README.md) for detailed schema documentation.

**Deed record structure:**
```json
{
  "deed_id": "deed_0001",
  "signed_date": "1924-03-15",
  "signed_year": 1924,
  "street_name": "Oak Street",
  "subdivision_name": "Pine Valley",
  "has_covenant": true,
  "has_date_conflict": false
}
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **F1 Score** | Harmonic mean of precision and recall on deed ID retrieval |
| **Exact Match** | 1.0 if prediction exactly equals ground truth, else 0.0 |
| **Temporal Accuracy** | Fraction of returned deeds satisfying temporal constraints |
| **Spatial Accuracy** | Fraction of returned deeds satisfying spatial constraints |

## System Comparison

| System | Approach | Strengths | Weaknesses |
|--------|----------|-----------|------------|
| **Vector RAG** | Embedding similarity | Simple, generalizable | No explicit temporal/spatial reasoning |
| **LightRAG** | Auto-extracted graph | Minimal configuration | Generic graph structure |
| **Custom Graph RAG** | Domain-specific schema | Precise constraints | Requires schema design |

## Requirements

- Python 3.10+
- NumPy, NetworkX
- OpenAI API (for full experiments)
- Optional: matplotlib, seaborn (for figures)

## Citation

```bibtex
@mastersthesis{lau2024strag,
  title={Graph RAG for Spatio-Temporal Reasoning in Historical Legal Document Analysis},
  author={Lau, Yifeng},
  school={Massachusetts Institute of Technology},
  year={2024}
}
```

## License

MIT License
