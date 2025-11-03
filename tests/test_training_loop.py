from src.models.base_model import BrainScanCNN
from src.training.optimizer import get_optimizer
from src.training.trainer import ModelTrainer
from torch.utils.data import DataLoader, TensorDataset
import torch

config = {
    "optimizer": {
        "name": "Adam",
        "lr": 0.001,
        "weight_decay": 0.0001
    },
    "train": {
        "batch_size": 3,
        "epochs": 1
    }
}

def test_training_one_epoch():
    model = BrainScanCNN(num_classes=3)
    
    x = torch.randn([4, 3, 224, 224])
    labels  = torch.randint(0, 3, (4,))
    loader = DataLoader(TensorDataset(x, labels), batch_size=2)
    
    optim = get_optimizer(config=config["optimizer"], model=model)
    loss = torch.nn.CrossEntropyLoss()
    
    trainer = ModelTrainer(config, model=model)
    training_loss = trainer._one_epoch_run(loader, model, optim, loss)
    assert isinstance(training_loss, float)
    assert torch.isfinite(torch.tensor(training_loss))