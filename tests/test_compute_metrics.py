import torch
from src.evaluation.metrics import compute_metrics

def test_metrics_computation():
    y_true = torch.tensor([0, 1, 2, 1])
    y_pred = torch.tensor([0, 2, 1, 2])
    metrics, per_class_metrics = compute_metrics(y_true, y_pred, class_names=["a", "b", "b"])
    assert isinstance(metrics, dict)
    assert isinstance(per_class_metrics, dict)
    
    assert "accuracy" in metrics # Making sure accuracy is in the metrics 
    assert "precision" in metrics # Making sure precision is in the metrics
    assert "recall" in metrics # Making sure recall is in the metrics
    assert "f1_score" in metrics # Making sure f1_score is in the metrics
    assert 0 <= metrics["accuracy"] <= 1 # Making sure accuracy is in the range [0,1]
    assert 0 <= metrics["precision"] <= 1 # Making sure precision is in the range [0,1]
    assert 0 <= metrics["recall"] <= 1 # Making sure recall is in the range [0,1]
    assert 0 <= metrics["f1_score"] <= 1 # Making sure f1_score is in the range [0,1]