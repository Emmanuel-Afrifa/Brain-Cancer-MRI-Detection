from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from typing import Literal
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

avg_typ = Literal['micro', 'macro', 'samples', 'weighted']

def compute_metrics(targets: list, preds: list, pred_probs: list | None = None, class_names: list = [], average: avg_typ = "macro") -> tuple:
    """
    This function computes the classification metrics

    Args:
        targets (list): 
            True labels of the data
        preds (list): 
            Predicted labels (by the trained model) of the data
        pred_probs (list | None, optional): 
            Prediction probabilities. Defaults to None.
        class_names (list, optional): 
            Class (label) names. Defaults to [].
        average (avg_typ, optional): 
            Averaging method for precision, recall, f1 and roc-auc computation. Options: "macro",
            "micro", "samples", "weighted". Defaults to "macro".

    Raises:
        ValueError: 
            Raised when no pred_probs parameter is passed.

    Returns:
        tuple: 
            Overall metrics and per-class performance metrics.
    """
    
    logger.info("Computing accurary, precision, recall and f1 scores.")
    metrics = {
        "accuracy": float(accuracy_score(targets, preds)),
        "precision": float(precision_score(targets, preds, average=average)),
        "recall": float(recall_score(targets, preds, average=average)),
        "f1_score": float(f1_score(targets, preds, average=average))
    }
    
    if pred_probs is not None:
        try:
            logger.info("Computing ROC-AUC score.")
            metrics["roc_auc"] = float(roc_auc_score(targets, pred_probs, average=average, multi_class="ovr"))
        except:
            logger.error("ROC-AUC score not computed. No predicted probablities (pred_probs) provided.")
            raise ValueError("ROC-AUC score not computed. No predicted probablities (pred_probs) provided.")
        
    metrics_per_class = classification_report(targets, preds, labels=list(range(len(class_names))), 
                                              target_names=class_names, zero_division=0, output_dict=True)
    
    return metrics, metrics_per_class



def get_confusion_matrix(targets: list, predictions: list, class_names: list, save_name: str = "confusion_matrix") -> None:
    """
    This function computes and plots the confusion matrix.

    Args:
        targets (list): 
            True labels of the data
        predictions (list): 
            Predicted labels (by the trained model) of the data
        class_names (list): 
            Class (label) names.
        save_name (str, optional):  
            Name used to save the resulting graph. Defaults to "confusion_matrix".
    """
    cm = confusion_matrix(targets, predictions)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=class_names)
    cm_display.plot(cmap="Greens")
    plt.savefig(f"artifacts/graphs/{save_name}.png")
    plt.show()