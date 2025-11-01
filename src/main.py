from src.data.data_module import BrainMRIDataModule
from src.data.data_splitter import DataSplitter
from src.models.base_model import BrainScanCNN
from src.training.callbacks import earlystopping
from src.training.optimizer import get_optimizer, get_lr_scheduler
from src.training.trainer import ModelTrainer
from src.utils.file_io import load_config
from src.utils.logger import setup_logging
from src.utils.seed import set_seed
from torch.nn import CrossEntropyLoss

import argparse
import logging
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["train", "eval"], help="Choose the mode to run the experiment in. Either to retrain the model or evaluate it's performance on the test set.")
    parser.add_argument("--config", required=True, type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    # setting up configurations
    configs = load_config(args.config)
    set_seed(configs["seed"])
    setup_logging("artifacts/logs/", filename="app.log")
    
    logger = logging.getLogger(__name__)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device detected: {device}")
    
    # Initializing model objects
    model = BrainScanCNN(num_classes=3)
    model.to(device=device)
    loss_func = CrossEntropyLoss()
    optimizer = get_optimizer(configs["optimizer"], model=model)
    lr_scheduler = get_lr_scheduler(config=configs["scheduler"], optimizer=optimizer)
    
    # Initializing data objects
    data_splitter = DataSplitter(config=configs)
    data_splitter.get_splits(configs["data"]["preprocessed"])
    data_module = BrainMRIDataModule(config=configs)
    # recheck the implementation of these functions. Will it be better to make them work for loading specific datasets, (highly reusable)
    train_dataset, val_dataset, test_dataset = data_module.get_datasets(save_mean_std_path="artifacts/preprocessing/normalization_mean_std.json")
    train_loader, val_loader, test_loader = data_module.get_dataloaders()
    
    # Initializing trainer object
    # Recheck design: Show I add .fit method to abstract some of the initializations?
    trainer = ModelTrainer(config=configs, model=model)
    epochs = configs["train"]["epochs"]
    
    if args.mode == "train":
        history = trainer.train(train_loader, val_loader, model, optimizer, loss_func, epochs=epochs,
                                lr_scheduler=lr_scheduler, training=True, early_stopping=earlystopping,
                                patience=5, device=device, save_history_path="artifacts/history/history_brain_cnn_baseline.pth",
                                checkpoint_path="artifacts/models/model_brain_cnn.pth")
        print(history)
    elif args.mode == "eval":
        pass
    
if __name__ == "__main__":
    main()