#!/usr/bin/env python3
"""
Step 3: Analyze Results

Analyzes experiment results and generates visualizations for thesis.

Usage:
    python scripts/03_analyze_results.py
    python scripts/03_analyze_results.py --results outputs/results/experiment_results_xxx.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_results(results_path: Path) -> Dict[str, Any]:
    """Load experiment results from JSON."""
    with open(results_path, 'r') as f:
        return json.load(f)


def print_summary(results: Dict[str, Any]):
    """Print summary statistics."""
    print("=" * 70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    
    systems = list(results.keys())
    
    # Overall metrics
    print("\n### Overall Performance ###")
    print(f"{'System':<25} {'F1':>10} {'EM':>10} {'Latency(ms)':>12}")
    print("-" * 60)
    
    for system in systems:
        r = results[system]
        print(f"{system:<25} {r['overall_f1']:>10.3f} {r['overall_exact_match']:>10.3f} {r['avg_latency_ms']:>12.1f}")
    
    # Per-level breakdown
    print("\n### Per-Level F1 Scores ###")
    
    levels = list(results[systems[0]]['level_results'].keys())
    
    header = f"{'Level':<25}" + "".join(f"{s:>15}" for s in systems)
    print(header)
    print("-" * len(header))
    
    for level in levels:
        row = f"{level:<25}"
        for system in systems:
            f1 = results[system]['level_results'][level]['avg_f1']
            row += f"{f1:>15.3f}"
        print(row)
    
    # Improvement analysis
    if len(systems) >= 2:
        print("\n### Improvement Analysis ###")
        
        # Compare to Vector RAG baseline
        baseline = None
        for s in systems:
            if 'vector' in s.lower():
                baseline = s
                break
        
        if baseline:
            baseline_f1 = results[baseline]['overall_f1']
            print(f"\nBaseline: {baseline} (F1={baseline_f1:.3f})")
            print()
            
            for system in systems:
                if system != baseline:
                    f1 = results[system]['overall_f1']
                    if baseline_f1 > 0:
                        improvement = ((f1 - baseline_f1) / baseline_f1) * 100
                        print(f"  {system}: {f1:.3f} ({improvement:+.1f}% vs baseline)")
                    else:
                        print(f"  {system}: {f1:.3f}")


def generate_plots(results: Dict[str, Any], output_dir: Path):
    """Generate visualization plots."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Skipping plot generation.")
        return
    
    systems = list(results.keys())
    levels = list(results[systems[0]]['level_results'].keys())
    
    # Plot 1: Overall F1 comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    f1_scores = [results[s]['overall_f1'] for s in systems]
    bars = ax.bar(systems, f1_scores, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12'][:len(systems)])
    
    ax.set_ylabel('F1 Score')
    ax.set_title('Overall RAG System Performance')
    ax.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_comparison.png', dpi=150)
    plt.savefig(output_dir / 'overall_comparison.pdf')
    print(f"Saved: {output_dir / 'overall_comparison.png'}")
    plt.close()
    
    # Plot 2: Per-level performance heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data = []
    for system in systems:
        row = [results[system]['level_results'][level]['avg_f1'] for level in levels]
        data.append(row)
    
    data = np.array(data)
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([l.replace('_', '\n') for l in levels], rotation=0)
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels(systems)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('F1 Score')
    
    # Add text annotations
    for i in range(len(systems)):
        for j in range(len(levels)):
            text = ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', 
                          color='white' if data[i, j] < 0.5 else 'black')
    
    ax.set_title('F1 Scores by System and Difficulty Level')
    plt.tight_layout()
    plt.savefig(output_dir / 'level_heatmap.png', dpi=150)
    plt.savefig(output_dir / 'level_heatmap.pdf')
    print(f"Saved: {output_dir / 'level_heatmap.png'}")
    plt.close()
    
    # Plot 3: Line chart showing performance across levels
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, system in enumerate(systems):
        f1_by_level = [results[system]['level_results'][level]['avg_f1'] for level in levels]
        ax.plot(range(len(levels)), f1_by_level, 
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                label=system, linewidth=2, markersize=8)
    
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(['L1', 'L2', 'L3', 'L4', 'L5'])
    ax.set_xlabel('Difficulty Level')
    ax.set_ylabel('F1 Score')
    ax.set_title('Performance Across Difficulty Levels')
    ax.legend(loc='best')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'level_lines.png', dpi=150)
    plt.savefig(output_dir / 'level_lines.pdf')
    print(f"Saved: {output_dir / 'level_lines.png'}")
    plt.close()
    
    print(f"\nAll plots saved to: {output_dir}")


def generate_latex_tables(results: Dict[str, Any], output_dir: Path):
    """Generate LaTeX tables for thesis."""
    systems = list(results.keys())
    levels = list(results[systems[0]]['level_results'].keys())
    
    # Table 1: Main results table
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{RAG System Comparison: F1 Scores by Difficulty Level}",
        r"\label{tab:main-results}",
        r"\begin{tabular}{l" + "c" * len(systems) + "}",
        r"\toprule",
        r"Difficulty Level & " + " & ".join([s.replace("_", r"\_") for s in systems]) + r" \\",
        r"\midrule"
    ]
    
    for level in levels:
        level_display = level.replace("_", r"\_")
        row = f"{level_display} & "
        scores = []
        for s in systems:
            f1 = results[s]['level_results'][level]['avg_f1']
            scores.append(f1)
        
        # Bold the best score
        max_score = max(scores)
        formatted = []
        for score in scores:
            if score == max_score:
                formatted.append(r"\textbf{" + f"{score:.3f}" + "}")
            else:
                formatted.append(f"{score:.3f}")
        
        row += " & ".join(formatted) + r" \\"
        lines.append(row)
    
    lines.extend([
        r"\midrule",
        r"Overall & " + " & ".join([
            (r"\textbf{" + f"{results[s]['overall_f1']:.3f}" + "}") 
            if results[s]['overall_f1'] == max(results[x]['overall_f1'] for x in systems)
            else f"{results[s]['overall_f1']:.3f}"
            for s in systems
        ]) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])
    
    table1_path = output_dir / "main_results_table.tex"
    with open(table1_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Saved: {table1_path}")
    
    # Table 2: Improvement over baseline
    baseline_system = None
    for s in systems:
        if 'vector' in s.lower():
            baseline_system = s
            break
    
    if baseline_system:
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Performance Improvement over Vector RAG Baseline}",
            r"\label{tab:improvement}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"System & F1 Score & Improvement & Latency (ms) & Speedup \\",
            r"\midrule"
        ]
        
        baseline_f1 = results[baseline_system]['overall_f1']
        baseline_latency = results[baseline_system]['avg_latency_ms']
        
        for system in systems:
            f1 = results[system]['overall_f1']
            latency = results[system]['avg_latency_ms']
            
            if baseline_f1 > 0:
                improvement = ((f1 - baseline_f1) / baseline_f1) * 100
                improvement_str = f"{improvement:+.1f}\\%"
            else:
                improvement_str = "N/A"
            
            if baseline_latency > 0:
                speedup = baseline_latency / latency
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "N/A"
            
            system_display = system.replace("_", r"\_")
            lines.append(f"{system_display} & {f1:.3f} & {improvement_str} & {latency:.1f} & {speedup_str} \\\\")
        
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}"
        ])
        
        table2_path = output_dir / "improvement_table.tex"
        with open(table2_path, 'w') as f:
            f.write("\n".join(lines))
        print(f"Saved: {table2_path}")


def analyze_failures(results: Dict[str, Any], output_dir: Path):
    """Analyze failure cases for insights."""
    print("\n### Failure Analysis ###\n")
    
    systems = list(results.keys())
    
    for system in systems:
        print(f"\n{system}:")
        
        failed_questions = []
        for level, level_data in results[system]['level_results'].items():
            for q in level_data.get('questions', []):
                if not q.get('is_correct', True):
                    failed_questions.append({
                        'level': level,
                        'question': q['question'],
                        'ground_truth': q['ground_truth'],
                        'prediction': q['prediction'],
                        'f1': q['metrics'].get('f1', 0)
                    })
        
        if not failed_questions:
            print("  No failures recorded (or detailed results not available)")
            continue
        
        # Group by level
        by_level = {}
        for fq in failed_questions:
            level = fq['level']
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(fq)
        
        for level, failures in by_level.items():
            print(f"  {level}: {len(failures)} failures")
            # Show worst failure
            if failures:
                worst = min(failures, key=lambda x: x['f1'])
                print(f"    Worst: Q=\"{worst['question'][:50]}...\"")
                print(f"           Expected: {str(worst['ground_truth'])[:50]}")
                print(f"           Got: {str(worst['prediction'])[:50]}")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--results", type=str, default=None, help="Path to results JSON")
    parser.add_argument("--output_dir", type=str, default="./outputs/figures", help="Output directory for figures")
    parser.add_argument("--no_plots", action="store_true", help="Skip plot generation")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find results file
    if args.results:
        results_path = Path(args.results)
    else:
        # Find most recent results file
        results_dir = Path("./outputs/results")
        if not results_dir.exists():
            print("No results directory found. Run experiments first.")
            sys.exit(1)
        
        results_files = list(results_dir.glob("experiment_results_*.json"))
        if not results_files:
            print("No results files found. Run experiments first.")
            sys.exit(1)
        
        results_path = max(results_files, key=lambda p: p.stat().st_mtime)
        print(f"Using most recent results: {results_path}")
    
    # Load results
    results = load_results(results_path)
    
    # Print summary
    print_summary(results)
    
    # Analyze failures
    analyze_failures(results, output_dir)
    
    # Generate plots
    if not args.no_plots:
        print("\n### Generating Plots ###")
        generate_plots(results, output_dir)
    
    # Generate LaTeX tables
    print("\n### Generating LaTeX Tables ###")
    tables_dir = Path("./outputs/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    generate_latex_tables(results, tables_dir)
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
