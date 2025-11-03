import pytest
from src.data.data_module import BrainMRIDataModule
from src.utils.file_io import load_config
import torch

config = {
    "data": {
        "img_size": [224, 224],
        "preprocessed": "tests/test_data"
    },
    "seed": 27,
    "train": {
        "batch_size": 3
    }    
}

def test_dataset():
    data_module = BrainMRIDataModule(config=config)
    with pytest.raises(ValueError, match="If `save_mean_std` is set to `True`, then `save_mean_std_path` must not be an empty string."):
        train_dataset, val_dataset, test_dataset = data_module.get_datasets()
    train_dataset, val_dataset, test_dataset = data_module.get_datasets("artifacts/preprocessing/test_norm.json")
    train_loader, val_loader, test_loader = data_module.get_dataloaders("artifacts/preprocessing/test_norm.json")

    assert len(train_dataset) == 9
    assert len(val_dataset) == 9
    assert len(test_dataset) == 9
    batch = next(iter(train_loader))
    images, labels = batch
    assert isinstance(images, torch.Tensor)
    assert images.shape[0] == 3 # Checking that the batch is 3 (config.train.batch_size)
    assert images.shape[1] == 3  # Checking that the images have 3 classes
    assert images[0].ndim == 3 # Checking that the image has 3 dimensions (C x H x W)
    assert images.ndim == 4 # Checking the dimensions of the batch (B x C x H x W)
    assert len(labels) == 3 # Checking that the labels match the batch size.
    assert isinstance(labels[0].item(), int)
    
