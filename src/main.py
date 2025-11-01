from src.data.data_module import BrainMRIDataModule
from src.data.data_splitter import DataSplitter
from src.evaluation.metrics import compute_metrics, get_confusion_matrix
from src.inference.predict import predict
from src.inference.inference_dataset import InferenceDataset
from src.inference.inference_loader import InferenceDataLoader
from src.models.base_model import BrainScanCNN
from src.training.callbacks import earlystopping
from src.training.optimizer import get_optimizer, get_lr_scheduler
from src.training.trainer import ModelTrainer
from src.utils.file_io import load_config, save_predictions
from src.utils.logger import setup_logging
from src.utils.seed import set_seed
from torch.nn import CrossEntropyLoss
import argparse
import json
import logging
import os
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["train", "eval", "predict"], help="Choose the mode to run the experiment in. Either to retrain the model or evaluate it's performance on the test set.")
    parser.add_argument("--config", required=True, type=str, default="configs/config.yaml")
    parser.add_argument("--input", type=str, help="Path to data to be predicted.")
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
    train_dataset, val_dataset, test_dataset = data_module.get_datasets(save_mean_std_path="artifacts/preprocessing/normalization_mean_std.json")
    train_loader, val_loader, test_loader = data_module.get_dataloaders()
    class_names = train_dataset.classes
    
    # Initializing trainer object
    trainer = ModelTrainer(config=configs, model=model)
    epochs = configs["train"]["epochs"]
    
    # Constants
    BASELINE_CNN_CHECKPOINT_PATH = "artifacts/models/model_brain_cnn.pth"
    BASELINE_MODEL_SAVED_HISTORY_PATH = "artifacts/history/history_brain_cnn_baseline.pth"
    BASELINE_TRAIN_NORMALIZATION_PATH = "artifacts/preprocessing/normalization_mean_std.json"
    
    if args.mode in ["eval", "predict"]:
        if os.path.exists(BASELINE_CNN_CHECKPOINT_PATH):
            checkpoint = torch.load(BASELINE_CNN_CHECKPOINT_PATH)
            logger.info("Loading saved model parameters")
            model.load_state_dict(checkpoint["model_state_dict"])    
    
    if args.mode == "train":
        history = trainer.train(train_loader, val_loader, model, optimizer, loss_func, epochs=epochs,
                                lr_scheduler=lr_scheduler, training=True, early_stopping=earlystopping,
                                patience=5, device=device, save_history_path=BASELINE_MODEL_SAVED_HISTORY_PATH,
                                checkpoint_path=BASELINE_CNN_CHECKPOINT_PATH)
        print(history)
    
    elif args.mode == "eval":
        targets = []
        for _, labels in test_loader:
            targets.extend(labels.numpy().tolist())
        image_paths = [os.path.basename(path) for path, _ in test_dataset.imgs]
        predictions, prediction_probs = predict(test_loader, model=model, device=device)
        metrics = compute_metrics(targets=targets, pred_probs=prediction_probs, preds=predictions, 
                                  class_names=class_names)
        save_predictions(predictions, prediction_probs, image_paths=image_paths, class_names=class_names, 
                         save_name="metrics_test_eval_baseline_brain_cnn", metrics=metrics)
        get_confusion_matrix(targets=targets, predictions=predictions, class_names=class_names, save_name="confusion_matrix_eval_basline_cnn")
        
    elif args.mode == "predict":
        if args.input:
            with open(BASELINE_TRAIN_NORMALIZATION_PATH, "r") as f:
                saved_mean_std = json.load(f)
            mean, std = saved_mean_std["mean"], saved_mean_std["std"]
            dataset = InferenceDataset(args.input, mean=mean, std=std, config=configs)
            dataloader = InferenceDataLoader(dataset=dataset, config=configs).get_inference_loaders()
            image_paths = [name for _, name in dataset]
            preds, pred_probs = predict(dataloader, model, device=device)
            save_predictions(preds, pred_probs, image_paths=image_paths, class_names=class_names, 
                             save_name="metrics_inference_baseline_cnn")
        else:
            logger.error("In `predict` mode, you must specify the input.")
            raise ValueError("In `predict` mode, you must specify the input.")

if __name__ == "__main__":
    main()