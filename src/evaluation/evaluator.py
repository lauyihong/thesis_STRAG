"""
Unified Evaluator

Provides a single entry point for evaluating RAG systems on benchmark questions.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Type
from datetime import datetime
from pathlib import Path

from .metrics import MetricsCalculator, f1_score, exact_match

# Handle both package and direct imports
try:
    from ..systems.base import BaseRAGSystem, RAGResponse
    from ..data.benchmark_questions import BenchmarkQuestion
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from systems.base import BaseRAGSystem, RAGResponse
    from data.benchmark_questions import BenchmarkQuestion


@dataclass
class QuestionResult:
    """Result for a single question."""
    question_id: str
    level: str
    question: str
    ground_truth: Any
    prediction: Any
    response: RAGResponse
    metrics: Dict[str, float]
    is_correct: bool  # Based on F1 > 0.5 threshold


@dataclass
class LevelResult:
    """Aggregated results for a difficulty level."""
    level: str
    num_questions: int
    avg_f1: float
    avg_precision: float
    avg_recall: float
    avg_exact_match: float
    avg_temporal_accuracy: Optional[float]
    avg_spatial_accuracy: Optional[float]
    avg_latency_ms: float
    question_results: List[QuestionResult]


@dataclass
class SystemResult:
    """Complete results for a RAG system."""
    system_name: str
    timestamp: str
    total_questions: int
    overall_f1: float
    overall_exact_match: float
    avg_latency_ms: float
    level_results: Dict[str, LevelResult]
    system_stats: Dict[str, Any]


class Evaluator:
    """
    Unified evaluator for RAG systems.
    
    Handles:
    - Running systems on benchmark questions
    - Computing metrics per question and per level
    - Aggregating results
    - Saving results to JSON
    """
    
    def __init__(self, data: Dict[str, Any]):
        """
        Initialize evaluator with dataset.
        
        Args:
            data: Full dataset from SyntheticDeedGenerator
        """
        self.data = data
        self.metrics_calc = MetricsCalculator(data)
    
    def evaluate_system(
        self,
        system: BaseRAGSystem,
        questions: Dict[str, List[BenchmarkQuestion]],
        verbose: bool = True
    ) -> SystemResult:
        """
        Evaluate a RAG system on all benchmark questions.
        
        Args:
            system: RAG system to evaluate
            questions: Dict mapping level names to question lists
            verbose: Print progress
        
        Returns:
            SystemResult with all metrics
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating: {system.name}")
            print(f"{'='*60}")
        
        level_results = {}
        all_question_results = []
        total_latency = 0
        
        for level_name, level_questions in questions.items():
            if verbose:
                print(f"\n  Level: {level_name} ({len(level_questions)} questions)")
            
            level_question_results = []
            
            for q in level_questions:
                # Query the system
                start = time.time()
                response = system.query(q.question)
                latency = (time.time() - start) * 1000
                total_latency += latency
                
                # Extract prediction from response
                prediction = self._extract_prediction(response, q)
                
                # Compute metrics
                metrics = self.metrics_calc.compute_all(
                    prediction, 
                    q.ground_truth,
                    q.metadata
                )
                
                # Determine correctness
                is_correct = metrics.get('f1', 0) > 0.5
                
                result = QuestionResult(
                    question_id=q.question_id,
                    level=q.level,
                    question=q.question,
                    ground_truth=q.ground_truth,
                    prediction=prediction,
                    response=response,
                    metrics=metrics,
                    is_correct=is_correct
                )
                
                level_question_results.append(result)
                all_question_results.append(result)
            
            # Aggregate level results
            level_result = self._aggregate_level_results(level_name, level_question_results)
            level_results[level_name] = level_result
            
            if verbose:
                print(f"    F1: {level_result.avg_f1:.3f} | EM: {level_result.avg_exact_match:.3f} | Latency: {level_result.avg_latency_ms:.1f}ms")
        
        # Compute overall metrics
        all_f1 = [r.metrics.get('f1', 0) for r in all_question_results]
        all_em = [r.metrics.get('exact_match', 0) for r in all_question_results]
        
        overall_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0
        overall_em = sum(all_em) / len(all_em) if all_em else 0
        avg_latency = total_latency / len(all_question_results) if all_question_results else 0
        
        result = SystemResult(
            system_name=system.name,
            timestamp=datetime.now().isoformat(),
            total_questions=len(all_question_results),
            overall_f1=overall_f1,
            overall_exact_match=overall_em,
            avg_latency_ms=avg_latency,
            level_results=level_results,
            system_stats=system.get_stats()
        )
        
        if verbose:
            print(f"\n  Overall: F1={overall_f1:.3f}, EM={overall_em:.3f}, Latency={avg_latency:.1f}ms")
        
        return result
    
    def _extract_prediction(self, response: RAGResponse, question: BenchmarkQuestion) -> Any:
        """
        Extract prediction from RAG response based on question type.
        
        For count questions, tries to extract a number.
        For list questions, uses retrieved IDs or extracts from answer.
        """
        # If we have retrieved IDs, prefer those
        if response.retrieved_ids:
            # For count questions, return the count
            if question.metadata.get('type', '').endswith('_count') or 'how many' in question.question.lower():
                return len(response.retrieved_ids)
            return response.retrieved_ids
        
        # Otherwise, try to extract from answer text
        answer = response.answer
        
        # Try to extract deed IDs
        import re
        deed_ids = re.findall(r'deed_\d+', answer.lower())
        if deed_ids:
            return deed_ids
        
        # Try to extract number for count questions
        if 'how many' in question.question.lower():
            numbers = re.findall(r'\b(\d+)\b', answer)
            if numbers:
                return int(numbers[0])
        
        # Return raw answer
        return answer
    
    def _aggregate_level_results(
        self, 
        level_name: str, 
        results: List[QuestionResult]
    ) -> LevelResult:
        """Aggregate results for a single level."""
        if not results:
            return LevelResult(
                level=level_name,
                num_questions=0,
                avg_f1=0, avg_precision=0, avg_recall=0,
                avg_exact_match=0,
                avg_temporal_accuracy=None,
                avg_spatial_accuracy=None,
                avg_latency_ms=0,
                question_results=[]
            )
        
        # Collect metrics
        f1_scores = [r.metrics.get('f1', 0) for r in results]
        precision_scores = [r.metrics.get('precision', 0) for r in results]
        recall_scores = [r.metrics.get('recall', 0) for r in results]
        em_scores = [r.metrics.get('exact_match', 0) for r in results]
        latencies = [r.response.latency_ms for r in results]
        
        # Temporal/spatial accuracy (only if present)
        temporal_scores = [r.metrics['temporal_accuracy'] for r in results if 'temporal_accuracy' in r.metrics]
        spatial_scores = [r.metrics['spatial_accuracy'] for r in results if 'spatial_accuracy' in r.metrics]
        
        return LevelResult(
            level=level_name,
            num_questions=len(results),
            avg_f1=sum(f1_scores) / len(f1_scores),
            avg_precision=sum(precision_scores) / len(precision_scores),
            avg_recall=sum(recall_scores) / len(recall_scores),
            avg_exact_match=sum(em_scores) / len(em_scores),
            avg_temporal_accuracy=sum(temporal_scores) / len(temporal_scores) if temporal_scores else None,
            avg_spatial_accuracy=sum(spatial_scores) / len(spatial_scores) if spatial_scores else None,
            avg_latency_ms=sum(latencies) / len(latencies),
            question_results=results
        )
    
    def compare_systems(
        self,
        systems: List[BaseRAGSystem],
        questions: Dict[str, List[BenchmarkQuestion]],
        verbose: bool = True
    ) -> Dict[str, SystemResult]:
        """
        Compare multiple RAG systems on the same benchmark.
        
        Args:
            systems: List of RAG systems to compare
            questions: Benchmark questions
            verbose: Print progress
        
        Returns:
            Dict mapping system name to results
        """
        results = {}
        
        for system in systems:
            result = self.evaluate_system(system, questions, verbose)
            results[system.name] = result
        
        if verbose:
            self._print_comparison_table(results)
        
        return results
    
    def _print_comparison_table(self, results: Dict[str, SystemResult]):
        """Print a comparison table of results."""
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        # Header
        systems = list(results.keys())
        header = f"{'Level':<25}" + "".join(f"{s:<15}" for s in systems)
        print(header)
        print("-" * len(header))
        
        # Get all levels
        levels = list(list(results.values())[0].level_results.keys())
        
        # Per-level F1 scores
        for level in levels:
            row = f"{level:<25}"
            for system in systems:
                f1 = results[system].level_results[level].avg_f1
                row += f"{f1:<15.3f}"
            print(row)
        
        # Overall
        print("-" * len(header))
        row = f"{'OVERALL F1':<25}"
        for system in systems:
            row += f"{results[system].overall_f1:<15.3f}"
        print(row)
        
        # Latency
        row = f"{'Avg Latency (ms)':<25}"
        for system in systems:
            row += f"{results[system].avg_latency_ms:<15.1f}"
        print(row)
    
    def save_results(self, results: Dict[str, SystemResult], output_path: str):
        """Save results to JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        serializable = {}
        for system_name, result in results.items():
            level_data = {}
            for level_name, level_result in result.level_results.items():
                level_data[level_name] = {
                    "num_questions": level_result.num_questions,
                    "avg_f1": level_result.avg_f1,
                    "avg_precision": level_result.avg_precision,
                    "avg_recall": level_result.avg_recall,
                    "avg_exact_match": level_result.avg_exact_match,
                    "avg_temporal_accuracy": level_result.avg_temporal_accuracy,
                    "avg_spatial_accuracy": level_result.avg_spatial_accuracy,
                    "avg_latency_ms": level_result.avg_latency_ms,
                    "questions": [
                        {
                            "question_id": qr.question_id,
                            "question": qr.question,
                            "ground_truth": qr.ground_truth,
                            "prediction": qr.prediction if not isinstance(qr.prediction, (list, set)) else list(qr.prediction),
                            "metrics": qr.metrics,
                            "is_correct": qr.is_correct
                        }
                        for qr in level_result.question_results
                    ]
                }
            
            serializable[system_name] = {
                "timestamp": result.timestamp,
                "total_questions": result.total_questions,
                "overall_f1": result.overall_f1,
                "overall_exact_match": result.overall_exact_match,
                "avg_latency_ms": result.avg_latency_ms,
                "level_results": level_data,
                "system_stats": result.system_stats
            }
        
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)
        
        print(f"\nResults saved to: {path}")
    
    def generate_latex_table(self, results: Dict[str, SystemResult]) -> str:
        """Generate LaTeX table for thesis."""
        systems = list(results.keys())
        levels = list(list(results.values())[0].level_results.keys())
        
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{RAG System Comparison Results}",
            r"\label{tab:rag-comparison}",
            r"\begin{tabular}{l" + "c" * len(systems) + "}",
            r"\toprule",
            r"Level & " + " & ".join(systems) + r" \\",
            r"\midrule"
        ]
        
        for level in levels:
            row = level.replace("_", r"\_") + " & "
            row += " & ".join(f"{results[s].level_results[level].avg_f1:.3f}" for s in systems)
            row += r" \\"
            lines.append(row)
        
        lines.extend([
            r"\midrule",
            r"Overall F1 & " + " & ".join(f"{results[s].overall_f1:.3f}" for s in systems) + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}"
        ])
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Quick test with mock data
    print("Evaluator module loaded successfully")
