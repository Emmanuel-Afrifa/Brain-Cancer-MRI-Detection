from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import StepLR, LRScheduler
from torch.optim import Adam
import logging
import torch

logger = logging.getLogger(__name__)

def get_optimizer(config: dict, model: torch.nn.Module) -> Optimizer:
    """
    This function returns the optimizer object

    Args:
        config (dict): 
            Optimizer configuration
        model (torch.nn.Module): 
            Model whose parameters are to be optimized

    Raises:
        ValueError: 
            Raised when the specified optimizer config does not exist. Available options (for now) include "Adam".

    Returns:
        Optimizer: 
            Optimizer object
    """
    name = config["name"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    
    logger.info(f"Creating optimizer: {name}, lr={lr}, weight_decay={weight_decay}")
    
    if str(name).lower() == "adam":
        optimizer = Adam(model.parameters(), lr = lr, weight_decay=weight_decay)
        return optimizer
    else:
        logger.error("The specified optimizer does not exist. Available: ['Adam']")
        raise ValueError("The specified optimizer does not exist. Available: ['Adam']")
    
def get_lr_scheduler(config: dict, optimizer: Optimizer) -> LRScheduler:
    """
    This function returns the learning rate scheduler object

    Args:
        config (dict): 
            Optimizer configuration
        optimizer (Optimizer): 
            Wrapped optimizer

    Raises:
        ValueError: 
            Raised when the specified scheduler config does not exist. Available options (for now) include "StepLR".

    Returns:
        LRScheduler: 
            Learning rate scheduler object.
    """
    name = config["name"]
    gamma = config["gamma"]
    step_size = config["step_size"]
    
    logger.info(f"Create LR Scheduler {name}")
    
    if str(name).lower() == "steplr":
        return StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
    else:
        logger.error("The specified LR Scheduler does not exist. Available: ['StepLR']")
        raise ValueError("The specified optimizer does not exist. Available: ['StepLR']")