"""
Evaluation Metrics

Implements evaluation metrics for RAG system comparison:
- Exact Match (EM)
- F1 Score
- Temporal Accuracy
- Spatial Accuracy
- Hop Coverage
"""

import re
from typing import List, Set, Any, Union, Dict
from collections import Counter


def normalize_answer(answer: Any) -> Set[str]:
    """
    Normalize an answer to a set of strings for comparison.
    
    Handles:
    - Single strings
    - Lists of strings
    - Integers (converted to string)
    - Sets
    """
    if answer is None:
        return set()
    
    if isinstance(answer, bool):
        return {str(answer).lower()}
    
    if isinstance(answer, (int, float)):
        return {str(int(answer))}
    
    if isinstance(answer, str):
        # Try to extract deed IDs or other structured items
        deed_ids = set(re.findall(r'deed_\d+', answer.lower()))
        if deed_ids:
            return deed_ids
        
        # Check if it's a number
        number_match = re.search(r'\b(\d+)\b', answer)
        if number_match:
            return {number_match.group(1)}
        
        # Return normalized string
        return {answer.lower().strip()}
    
    if isinstance(answer, (list, tuple)):
        result = set()
        for item in answer:
            result.update(normalize_answer(item))
        return result
    
    if isinstance(answer, set):
        result = set()
        for item in answer:
            result.update(normalize_answer(item))
        return result
    
    return {str(answer).lower()}


def extract_deed_ids(text: str) -> Set[str]:
    """Extract deed IDs from a text string."""
    return set(re.findall(r'deed_\d+', text.lower()))


def exact_match(prediction: Any, ground_truth: Any) -> float:
    """
    Compute Exact Match score.
    
    Returns 1.0 if prediction exactly matches ground truth, 0.0 otherwise.
    """
    pred_set = normalize_answer(prediction)
    truth_set = normalize_answer(ground_truth)
    
    return 1.0 if pred_set == truth_set else 0.0


def f1_score(prediction: Any, ground_truth: Any) -> Dict[str, float]:
    """
    Compute F1 score with precision and recall.
    
    Returns dict with precision, recall, and f1 scores.
    """
    pred_set = normalize_answer(prediction)
    truth_set = normalize_answer(ground_truth)
    
    if not pred_set and not truth_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    
    if not pred_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    if not truth_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    true_positives = len(pred_set & truth_set)
    
    precision = true_positives / len(pred_set)
    recall = true_positives / len(truth_set)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {"precision": precision, "recall": recall, "f1": f1}


def temporal_accuracy(
    prediction: Any,
    ground_truth: Any,
    temporal_constraint: Dict[str, int],
    data: Dict[str, Any]
) -> float:
    """
    Compute temporal accuracy - how well predictions satisfy temporal constraints.
    
    Args:
        prediction: Predicted deed IDs
        ground_truth: Ground truth deed IDs
        temporal_constraint: Dict with start_year, end_year, or specific_year
        data: Full dataset for looking up deed dates
    
    Returns:
        Fraction of predictions that satisfy temporal constraints
    """
    pred_ids = normalize_answer(prediction)
    
    if not pred_ids:
        return 0.0
    
    deeds = data.get('deeds', {})
    
    start_year = temporal_constraint.get('start_year', 0)
    end_year = temporal_constraint.get('end_year', 9999)
    specific_year = temporal_constraint.get('specific_year')
    
    if specific_year:
        start_year = specific_year
        end_year = specific_year
    
    valid_count = 0
    total_count = 0
    
    for pred_id in pred_ids:
        deed = deeds.get(pred_id)
        if deed:
            total_count += 1
            year = deed.get('signed_year', 0)
            if start_year <= year <= end_year:
                valid_count += 1
    
    if total_count == 0:
        return 0.0
    
    return valid_count / total_count


def spatial_accuracy(
    prediction: Any,
    ground_truth: Any,
    spatial_constraint: Dict[str, str],
    data: Dict[str, Any]
) -> float:
    """
    Compute spatial accuracy - how well predictions satisfy spatial constraints.
    
    Args:
        prediction: Predicted deed IDs
        ground_truth: Ground truth deed IDs
        spatial_constraint: Dict with subdivision_id or street_id
        data: Full dataset for looking up deed locations
    
    Returns:
        Fraction of predictions that satisfy spatial constraints
    """
    pred_ids = normalize_answer(prediction)
    
    if not pred_ids:
        return 0.0
    
    deeds = data.get('deeds', {})
    
    target_subdivision = spatial_constraint.get('subdivision_id')
    target_street = spatial_constraint.get('street_id')
    
    valid_count = 0
    total_count = 0
    
    for pred_id in pred_ids:
        deed = deeds.get(pred_id)
        if deed:
            total_count += 1
            
            if target_subdivision and deed.get('subdivision_id') == target_subdivision:
                valid_count += 1
            elif target_street and deed.get('street_id') == target_street:
                valid_count += 1
            elif not target_subdivision and not target_street:
                valid_count += 1  # No spatial constraint
    
    if total_count == 0:
        return 0.0
    
    return valid_count / total_count


def hop_coverage(
    prediction: Any,
    ground_truth: Any,
    required_hops: int = 1
) -> float:
    """
    Compute hop coverage - completeness of multi-hop retrieval.
    
    For questions requiring traversing multiple edges in the graph,
    this measures how many of the required entities were found.
    
    This is essentially recall for multi-hop questions.
    """
    pred_set = normalize_answer(prediction)
    truth_set = normalize_answer(ground_truth)
    
    if not truth_set:
        return 1.0 if not pred_set else 0.0
    
    # Hop coverage is recall
    true_positives = len(pred_set & truth_set)
    return true_positives / len(truth_set)


class MetricsCalculator:
    """
    Calculator for all evaluation metrics.
    
    Provides a unified interface for computing metrics across questions.
    """
    
    def __init__(self, data: Dict[str, Any]):
        """
        Initialize with dataset for constraint checking.
        
        Args:
            data: Full dataset from SyntheticDeedGenerator
        """
        self.data = data
    
    def compute_all(
        self,
        prediction: Any,
        ground_truth: Any,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """
        Compute all applicable metrics for a prediction.
        
        Args:
            prediction: Model prediction (answer text or list of IDs)
            ground_truth: Ground truth answer
            metadata: Question metadata with constraints
        
        Returns:
            Dict of metric names to scores
        """
        metadata = metadata or {}
        
        results = {
            "exact_match": exact_match(prediction, ground_truth),
            **f1_score(prediction, ground_truth)
        }
        
        # Temporal accuracy if temporal constraints present
        temporal_constraint = metadata.get('temporal_constraints', {})
        if not temporal_constraint:
            # Try to extract from metadata
            if 'start_year' in metadata or 'end_year' in metadata:
                temporal_constraint = {
                    'start_year': metadata.get('start_year'),
                    'end_year': metadata.get('end_year')
                }
            elif 'decade' in metadata:
                decade = metadata['decade']
                temporal_constraint = {
                    'start_year': decade,
                    'end_year': decade + 9
                }
            elif 'year' in metadata:
                temporal_constraint = {'specific_year': metadata['year']}
        
        if temporal_constraint:
            results["temporal_accuracy"] = temporal_accuracy(
                prediction, ground_truth, temporal_constraint, self.data
            )
        
        # Spatial accuracy if spatial constraints present
        spatial_constraint = metadata.get('spatial_constraints', {})
        if not spatial_constraint:
            if 'subdivision_id' in metadata:
                spatial_constraint = {'subdivision_id': metadata['subdivision_id']}
            elif 'street_id' in metadata:
                spatial_constraint = {'street_id': metadata['street_id']}
        
        if spatial_constraint:
            results["spatial_accuracy"] = spatial_accuracy(
                prediction, ground_truth, spatial_constraint, self.data
            )
        
        # Hop coverage (always computed, equals recall for list answers)
        results["hop_coverage"] = hop_coverage(prediction, ground_truth)
        
        return results
    
    def aggregate_metrics(self, all_results: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate metrics across multiple questions.
        
        Args:
            all_results: List of metric dicts from compute_all()
        
        Returns:
            Dict with aggregated (averaged) metrics
        """
        if not all_results:
            return {}
        
        # Collect all metric names
        metric_names = set()
        for result in all_results:
            metric_names.update(result.keys())
        
        # Compute averages
        aggregated = {}
        for metric in metric_names:
            values = [r.get(metric) for r in all_results if metric in r]
            if values:
                aggregated[metric] = sum(values) / len(values)
        
        aggregated["num_questions"] = len(all_results)
        
        return aggregated


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")
    
    # Test exact match
    assert exact_match(["deed_0001", "deed_0002"], ["deed_0001", "deed_0002"]) == 1.0
    assert exact_match(["deed_0001"], ["deed_0001", "deed_0002"]) == 0.0
    print("✓ exact_match")
    
    # Test F1
    f1_result = f1_score(["deed_0001", "deed_0002"], ["deed_0001", "deed_0003"])
    assert f1_result["precision"] == 0.5
    assert f1_result["recall"] == 0.5
    print("✓ f1_score")
    
    # Test normalize_answer
    assert normalize_answer("Found deeds: deed_0001, deed_0002") == {"deed_0001", "deed_0002"}
    assert normalize_answer(5) == {"5"}
    assert normalize_answer(["deed_0001"]) == {"deed_0001"}
    print("✓ normalize_answer")
    
    print("\nAll tests passed!")
