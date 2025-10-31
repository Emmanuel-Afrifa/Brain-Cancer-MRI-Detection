import torch
import numpy as np
import random

def set_seed(seed: int = 42) -> None:
    """
    Sets the seed for the random number generators.

    Args:
        seed (int, optional): 
            Seed set for the generators. Defaults to 42.
    """
    random.seed(seed)                
    np.random.seed(seed)             
    torch.manual_seed(seed)          
    torch.cuda.manual_seed(seed)     
    torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
