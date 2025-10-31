from torch.optim.optimizer import Optimizer
import logging
import torch

logger = logging.getLogger(__name__)

def checkpointing(val_loss: float, best_val_loss: float, model: torch.nn.Module, optimizer: Optimizer, 
                  checkpoint_path: str) -> None:
    """
    This model checkpoints (saves) the model state dictionaries if the new validation loss is lower than the previous 
    best validation loss

    Args:
        val_loss (float): 
            New validation loss.
        best_val_loss (float): 
            Previous validation loss.
        model (torch.nn.Module): 
            Model whose state dictionaries are to be savedt.
        optimizer (Optimizer): 
            Optimizer whose state dictionaries are to be saved.
        checkpoint_path (str): 
            Path to save the state dictionaries.
    """
    if val_loss < best_val_loss:
        logger.info(F"Saving checkpoint. Previous best val loss {best_val_loss}, new best val loss {val_loss}")
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": val_loss
        }, checkpoint_path)
        
        
def earlystopping(val_loss: float, best_val_loss: float, counter: int, patience: int = 5) -> tuple[bool, int]:
    """
    This function invocates early stopping to prevent the model from overfitting if the validation
    loss has not improved after `patience` epochs

    Args:
        val_loss (float): 
            New validation loss
        best_val_loss (float): 
            Previous best validation loss
        counter (int): 
            Indicator for number of epochs since validation loss improved. I resets if he validation 
            loss improves.
        patience (int, optional): 
            Number of epochs to wait before declaring early stopping. Defaults to 5.

    Returns:
        tuple[bool, int]: 
            A bool (which indicates whether early stopping should be invoked or not) and the counter.
    """
    
    if val_loss < best_val_loss:
        counter = 0
    else:
        counter += 1
            
    if counter > patience:
        stop = True
    else:
        stop = False
        
    return stop, counter