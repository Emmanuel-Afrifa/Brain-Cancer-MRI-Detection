from src.inference.inference_dataset import InferenceDataset
import torch

config = {
    "data": {
        "img_size": [224, 224],
        "preprocessed": "tests/test_data"
    }
}

def test_inference_dataset():
    mean, std = torch.randn(3), torch.randn(3)
    dataset = InferenceDataset("tests/test_data/test/brain_glioma", mean=mean, std=std, config=config)
    
    img, label = dataset[0]
    
    assert len(dataset) == 3
    assert isinstance(img, torch.Tensor)
    assert img.ndim == 3
    assert isinstance(label, str)