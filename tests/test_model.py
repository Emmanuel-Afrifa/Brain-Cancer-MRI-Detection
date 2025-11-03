from src.models.base_model import BrainScanCNN
import torch

def test_model():
    model = BrainScanCNN(num_classes=3)
    x = torch.randn([3, 3, 224, 224])
    output = model(x)
    assert output.shape[0] == x.shape[0] 
    assert output.shape[1] == 3
    