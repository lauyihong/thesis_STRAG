# thesis_STRAG

**Graph RAG for Spatio-Temporal Reasoning in Historical Legal Document Analysis**

This repository contains the experimental framework for comparing Vector RAG, LightRAG, and Custom Graph RAG architectures on spatio-temporal reasoning tasks over historical deed documents.

## Research Question

> In legal document question-answering tasks with spatio-temporal constraints, how much performance improvement does explicit spatio-temporal relationship modeling provide compared to generic graph structures?

## Key Results

| Method | F1 Score | vs Vector RAG |
|--------|----------|---------------|
| Vector RAG | 0.007 | baseline |
| LightRAG (Hybrid) | TBD | TBD |
| Custom Graph RAG V2 | 0.598 | **+8,352%** |

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

### 1. Environment Setup

```bash
conda create -n strag python=3.11 -y
conda activate strag
pip install -r requirements.txt
```

### 2. Set API Key

```bash
export OPENAI_API_KEY="sk-your-key"
```

### 3. Run Experiments

```bash
# Generate synthetic data
python scripts/01_generate_data.py --num_deeds 100

# Run all RAG systems
python scripts/02_run_experiments.py

# Analyze results
python scripts/03_analyze_results.py
```

## Benchmark Question Hierarchy

| Level | Description | Example |
|-------|-------------|---------|
| L1 | Single-hop lookup | "Find all deeds recorded in 1924" |
| L2 | Temporal reasoning | "List deeds signed between 1926-1939" |
| L3 | Spatial multi-hop | "Which deeds share streets with deed_0001?" |
| L4 | Spatio-temporal joint | "How many covenants in Pine Valley during 1910s?" |
| L5 | Conflict detection | "Identify deeds with inconsistent date annotations" |

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
