"""
Evaluation module for RAG system comparison.
"""

from .metrics import (
    normalize_answer,
    extract_deed_ids,
    exact_match,
    f1_score,
    temporal_accuracy,
    spatial_accuracy,
    hop_coverage,
    MetricsCalculator
)
from .evaluator import (
    Evaluator,
    QuestionResult,
    LevelResult,
    SystemResult
)

__all__ = [
    # Metrics
    'normalize_answer',
    'extract_deed_ids',
    'exact_match',
    'f1_score',
    'temporal_accuracy',
    'spatial_accuracy',
    'hop_coverage',
    'MetricsCalculator',
    
    # Evaluator
    'Evaluator',
    'QuestionResult',
    'LevelResult',
    'SystemResult'
]
