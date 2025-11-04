from src.preprocessing.preprocessing import compute_mean_std
from torch.utils.data import DataLoader, TensorDataset
import torch

def test_compute_mean_std():
    x = torch.abs(torch.randn([4, 3, 224, 224]))
    labels = torch.randint(0, 3, (4,))
    loader = DataLoader(TensorDataset(x, labels), batch_size=2)
    
    mean, std = compute_mean_std(loader=loader)
    
    assert isinstance(mean, torch.Tensor)
    assert isinstance(std, torch.Tensor)
    assert mean.shape == (3,)
    assert std.shape == (3,)
    assert torch.all((0 <= mean) & (mean <= 1))
    assert torch.all(std >= 0)
    

