#!/usr/bin/env python3
"""
Step 1: Generate Synthetic Data

Generates synthetic deed documents and benchmark questions for RAG evaluation.

Usage:
    python scripts/01_generate_data.py --toy              # Quick toy dataset (10 deeds)
    python scripts/01_generate_data.py --num_deeds 100    # Custom size
    python scripts/01_generate_data.py --config configs/experiment_config.yaml
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.synthetic_generator import SyntheticDeedGenerator, GeneratorConfig, load_config_from_yaml
from data.benchmark_questions import BenchmarkQuestionGenerator
from data.text_converter import TextConverter


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic deed data")
    parser.add_argument("--toy", action="store_true", help="Generate minimal toy dataset (10 deeds, 3 questions/level)")
    parser.add_argument("--num_deeds", type=int, default=100, help="Number of deeds to generate")
    parser.add_argument("--num_subdivisions", type=int, default=5, help="Number of subdivisions")
    parser.add_argument("--questions_per_level", type=int, default=10, help="Questions per difficulty level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--text_style", type=str, default="mixed", choices=["concise", "narrative", "mixed"])

    args = parser.parse_args()

    # Apply toy mode settings
    if args.toy:
        args.num_deeds = 10
        args.num_subdivisions = 2
        args.questions_per_level = 3
        print("==> TOY MODE: Using minimal dataset for quick testing")

    # Load config
    if args.config:
        config = load_config_from_yaml(args.config)
        print(f"Loaded config from: {args.config}")
    else:
        config = GeneratorConfig(
            num_deeds=args.num_deeds,
            num_subdivisions=args.num_subdivisions,
            seed=args.seed
        )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Step 1: Generating Synthetic Deed Data")
    print("=" * 60)
    print(f"  Deeds: {config.num_deeds}")
    print(f"  Subdivisions: {config.num_subdivisions}")
    print(f"  Seed: {config.seed}")
    print()
    
    # Generate structured data
    print("Generating structured data...")
    generator = SyntheticDeedGenerator(config)
    data = generator.generate()
    
    # Save structured data
    structured_path = output_dir / "synthetic_deeds.json"
    with open(structured_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {structured_path}")
    
    # Generate ground truth index
    ground_truth = generator.get_ground_truth()
    gt_path = output_dir / "ground_truth.json"
    with open(gt_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    print(f"  Saved: {gt_path}")
    
    # Generate benchmark questions
    print("\nGenerating benchmark questions...")
    question_gen = BenchmarkQuestionGenerator(data, seed=config.seed)
    questions = question_gen.generate_all(questions_per_level=args.questions_per_level)
    
    questions_path = output_dir / "benchmark_questions.json"
    question_gen.save(questions, str(questions_path))
    print(f"  Saved: {questions_path}")
    
    # Print question counts
    total_questions = sum(len(q) for q in questions.values())
    print(f"\n  Total questions: {total_questions}")
    for level, q_list in questions.items():
        print(f"    {level}: {len(q_list)}")
    
    # Convert to text documents for LightRAG
    print("\nConverting to text documents...")
    converter = TextConverter(style=args.text_style)
    
    # Save as individual files
    text_dir = output_dir / "deeds_text"
    files = converter.save_as_documents(data, str(text_dir))
    print(f"  Saved {len(files)} text files to: {text_dir}")
    
    # Save as single combined file
    combined_path = output_dir / "deeds_combined.txt"
    converter.save_as_single_file(data, str(combined_path))
    print(f"  Saved combined text: {combined_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Data Generation Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"  - synthetic_deeds.json    ({len(data['deeds'])} deeds)")
    print(f"  - ground_truth.json       (precomputed answers)")
    print(f"  - benchmark_questions.json ({total_questions} questions)")
    print(f"  - deeds_text/             ({len(files)} text files)")
    print(f"  - deeds_combined.txt      (single file for LightRAG)")
    
    # Print sample question
    print("\nSample questions:")
    for level, q_list in list(questions.items())[:3]:
        if q_list:
            print(f"  [{level}] {q_list[0].question}")


if __name__ == "__main__":
    main()
