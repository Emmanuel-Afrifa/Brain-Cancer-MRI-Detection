from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score
from src.training.callbacks import checkpointing
from src.utils.file_io import save_objects
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
from typing import Callable
import logging
import torch


class ModelTrainer:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.epochs = self.config["train"]["epochs"]
        self.batch_size = self.config["train"]["batch_size"]
        self.logger = logging.getLogger(__name__)
        
    def _one_epoch_run(self, data_loader, model: torch.nn.Module, optimizer: Optimizer, 
                       loss_func: torch.nn.modules.loss._Loss,  device: str | torch.device = "cpu") -> float:
        """
        This method defines the training loop for one epoch

        Args:
            data_loader (torch.utils.data.DataLoader): 
                Data loader 
            model (torch.nn.Module): 
                Model to be trained
            optimizer (Optimizer): 
                Optimizer for tuning model parameters
            loss_func (torch.nn.modules.loss._Loss): 
                Loss function for computing model performance
            device (str | torch.device, optional): 
                Device on which to perform computations. Defaults to "cpu".

        Returns:
            float: 
                loss value.
        """
        train_loss = 0.0
        model.train() 
        
        for data, labels in tqdm(data_loader, desc="Training", leave=False):
            optimizer.zero_grad()
            data.to(device)
            labels.to(device)
            
            output = model(data)
            loss = loss_func(output, labels)            
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.data.item() * len(data)
        train_loss /= len(data_loader.dataset)
        return train_loss
    
    def _evaluate_model(self, data_loader, model: torch.nn.Module, loss_func: torch.nn.modules.loss._Loss, 
                        device: str | torch.device = "cpu") -> tuple[float, float, float, float]:
        """
        This method evaluates the model performance by computing metrics like accuracy, f1-macro, and f1-weighted

        Args:
            data_loader (torch.utils.data.DataLoader): 
                Data loader.
            model (torch.nn.Module): 
                Model whose performance is being evaluated.
            loss_func (torch.nn.modules.loss._Loss): 
                Loss function for computing model performance.
            device (str | torch.device, optional): 
                Device on which to perform computations. Defaults to "cpu".. Defaults to "cpu".

        Returns:
            tuple[float, float, float, float]: _description_
        """
        val_loss = 0
        all_preds = []
        all_labs = []
        
        model.eval()
        
        with torch.no_grad():
            for data, labels in tqdm(data_loader, desc="Evaluating", leave=False):
                data.to(device)
                labels.to(device)
                
                output = model(data)
                loss = loss_func(output, labels)
                
                val_loss += loss.data.item() * len(data)
                
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labs.extend(labels.cpu().numpy())
                
        val_loss /= len(data_loader)
        
        macro_f1 = float(f1_score(all_labs, all_preds, average="macro"))
        weighed_f1 = float(f1_score(all_labs, all_preds, average="weighted"))
        acc_score = float(accuracy_score(all_labs, all_preds))
        
        return (val_loss, acc_score, macro_f1, weighed_f1)
    
    def train(self, train_loader, val_loader, model: torch.nn.Module, optimizer: Optimizer, loss_func: torch.nn.modules.loss._Loss, 
               checkpoint_path: str, epochs: int = 20, lr_scheduler: LRScheduler | None = None, training: bool = True,
               early_stopping: Callable | None = None, patience: int = 5, device: str | torch.device = "cpu",
               save_history_path: str = "") -> dict:
        """
        This method defines the entire traing loop for the specified number of epochs

        Args:
            train_loader (torch.utils.data.DataLoader): 
                Data loader for the training set.
            val_loader (torch.utils.data.DataLoader): 
                Data loader for the validation set.
            model (torch.nn.Module): 
                Model to be trained.
            optimizer (Optimizer): 
                Optimizer for tuning model parameters
            loss_func (torch.nn.modules.loss._Loss): 
                Loss function for computing model performance.
            checkpoint_path (str): 
                Path to save the model parameters
            epochs (int, optional): 
                Specifies the maximum nuber of epochs to train the model. Defaults to 20.
            lr_scheduler (LRScheduler | None, optional): 
                Learning rate scheduler. Defaults to None.
            training (bool, optional): 
                Specifies whether the training metrics should be included (computed). Defaults to True.
            early_stopping (Callable | None, optional): 
                Early stopping function. Defaults to None.
            patience (int, optional): 
                Number of epochs to wait before triggering early stopping. Defaults to 5.
            device (str | torch.device, optional): 
                Device on which to perform computations. Defaults to "cpu".. Defaults to "cpu".
            save_history_path (str, optional): 
                Path to save the training history. Defaults to "".

        Returns:
            dict: 
                Training History
        """
        history = defaultdict(list)
        
        best_val_loss = float("inf")
        stopping_counter = 0
        
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            
            training_loss = self._one_epoch_run(train_loader, model, optimizer, loss_func, device)
            
            if training:
                train_loss, train_acc, train_f1_macro, train_f1_weighted = self._evaluate_model(
                    train_loader, model, loss_func, device)
            else:
                train_loss = training_loss
                train_acc, train_f1_macro, train_f1_weighted = 0, 0, 0
                
            val_loss, val_acc, val_f1_macro, val_f1_weighted = self._evaluate_model(
                val_loader, model, loss_func, device)
            
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["train_f1_macro"].append(train_f1_macro)
            history["train_f1_weighted"].append(train_f1_weighted)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1_macro"].append(val_f1_macro)
            history["val_f1_weighted"].append(val_f1_weighted)
            
            print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            print(f"Train Accuracy: {train_acc}, Validation Accuracy: {val_acc}")
            print(f"Train F1-macro: {train_f1_macro}, Validation F1-macro: {val_f1_macro}")
            print(f"Train F1-weighted: {train_f1_weighted}, Validation F1-weighted: {val_f1_weighted}")
        
            lr = optimizer.param_groups[0]["lr"]
            history["learning_rates"].append(lr)
            if lr_scheduler:
                lr_scheduler.step()
                
            if checkpoint_path:
                checkpointing(val_loss=val_loss, best_val_loss=best_val_loss, model=model, optimizer=optimizer, checkpoint_path=checkpoint_path)
                
            if early_stopping:
                stop, stopping_counter = early_stopping(val_loss=val_loss, best_val_loss=best_val_loss, counter=stopping_counter, patience=patience)
                if stop:
                    self.logger.info(f"Model trained has stopped after {epoch} epochs")
                    break
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
    
        save_objects(save_path=save_history_path, object=history)
        return history
