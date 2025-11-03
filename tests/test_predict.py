from src.inference.predict import predict
from src.models.base_model import BrainScanCNN
from torch.utils.data import DataLoader, TensorDataset
import torch

def test_predict():
    model = BrainScanCNN(num_classes=3)
    x = torch.randn([4, 3, 224, 224])
    x_labels = torch.randint(0, 3, (4,))
    dataloader = DataLoader(TensorDataset(x, x_labels), batch_size=2)
    
    preds, pred_probs = predict(dataloader, model)
    
    assert isinstance(preds, list)
    assert isinstance(pred_probs, list)
    assert len(pred_probs[0]) == 3
    assert len(preds) == 4
    assert not model.training
    
    preds_2, pred_probs_2 = predict(dataloader, model)
    assert torch.allclose(torch.tensor(preds), torch.tensor(preds_2))
    