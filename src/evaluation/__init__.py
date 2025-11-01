"""
This submodule contains functions to compute the evaluation (performance) metrics to ascertain the
performance of the model.
- compute_metrics: Computes the various classification metrics (accuracy, precision, recall, f1-score, roc-auc)
- get_confustion_matrix: Compute and plots the confusion matrix
"""

from .metrics import compute_metrics, get_confusion_matrix

__all__ = [
    "compute_metrics",
    "get_confusion_matrix"
]