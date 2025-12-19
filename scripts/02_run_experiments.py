#!/usr/bin/env python3
"""
Step 2: Run Experiments

Runs all RAG systems on the benchmark questions and saves results.

Usage:
    python scripts/02_run_experiments.py
    python scripts/02_run_experiments.py --systems vector,graph_v2
    python scripts/02_run_experiments.py --mock  # Use mock embeddings (no API)
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.synthetic_generator import SyntheticDeedGenerator
from data.benchmark_questions import BenchmarkQuestionGenerator
from systems.vector_rag import VectorRAG, VectorRAGConfig
from systems.custom_graph_rag import CustomGraphRAGV1, CustomGraphRAGV2
from systems.lightrag_wrapper import LightRAGHybrid, LightRAGNaive
from evaluation.evaluator import Evaluator


def load_data(data_dir: Path):
    """Load generated data and questions."""
    # Load structured data
    data_path = data_dir / "synthetic_deeds.json"
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Load questions
    questions_path = data_dir / "benchmark_questions.json"
    questions = BenchmarkQuestionGenerator.load(str(questions_path))
    
    return data, questions


def create_systems(args, data):
    """Create RAG systems based on arguments."""
    systems = []
    
    requested = args.systems.split(",") if args.systems else ["all"]
    
    # Vector RAG
    if "all" in requested or "vector" in requested:
        config = VectorRAGConfig(
            use_mock_embeddings=args.mock,
            use_llm_answer=not args.mock
        )
        systems.append(VectorRAG(config))
    
    # Custom Graph RAG V1
    if "all" in requested or "graph_v1" in requested:
        systems.append(CustomGraphRAGV1(use_llm_answer=not args.mock))
    
    # Custom Graph RAG V2
    if "all" in requested or "graph_v2" in requested:
        systems.append(CustomGraphRAGV2(use_llm_answer=not args.mock))
    
    # LightRAG (requires API)
    if not args.mock:
        if "all" in requested or "lightrag_naive" in requested:
            systems.append(LightRAGNaive())
        
        if "all" in requested or "lightrag_hybrid" in requested:
            systems.append(LightRAGHybrid())
    
    return systems


def main():
    parser = argparse.ArgumentParser(description="Run RAG experiments")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./outputs/results", help="Output directory")
    parser.add_argument("--systems", type=str, default=None, 
                       help="Comma-separated systems to run (vector,graph_v1,graph_v2,lightrag_naive,lightrag_hybrid)")
    parser.add_argument("--mock", action="store_true", help="Use mock embeddings (no API calls)")
    parser.add_argument("--run_id", type=str, default=None, help="Run identifier")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check API key if not mock
    if not args.mock and not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Use --mock for testing without API.")
        print("Set API key: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)
    
    print("=" * 60)
    print("Step 2: Running RAG Experiments")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    data, questions = load_data(data_dir)
    print(f"  Loaded {len(data['deeds'])} deeds")
    print(f"  Loaded {sum(len(q) for q in questions.values())} questions")
    
    # Create systems
    print("\nInitializing RAG systems...")
    systems = create_systems(args, data)
    print(f"  Systems to evaluate: {[s.name for s in systems]}")
    
    # Index data into each system
    print("\nIndexing data into each system...")
    for system in systems:
        print(f"  Indexing {system.name}...")
        system.index(data)
    
    # Run evaluation
    print("\nRunning evaluation...")
    evaluator = Evaluator(data)
    results = evaluator.compare_systems(systems, questions, verbose=True)
    
    # Save results
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"experiment_results_{run_id}.json"
    evaluator.save_results(results, str(results_path))
    
    # Generate LaTeX table
    latex_table = evaluator.generate_latex_table(results)
    latex_path = output_dir / f"results_table_{run_id}.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {latex_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)
    
    print("\nFinal Results (F1 Scores):")
    for system_name, result in results.items():
        print(f"  {system_name}: {result.overall_f1:.3f}")
    
    # Best system
    best_system = max(results.items(), key=lambda x: x[1].overall_f1)
    print(f"\nBest performing system: {best_system[0]} (F1={best_system[1].overall_f1:.3f})")


if __name__ == "__main__":
    main()
