from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

def compute_mean_std(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """
    This function computes the mean and standard deviation (std) across the different image channels
    for the specified data loader

    Args:
        loader (DataLoader): 
            Data loader whose mean and std is to be computed.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 
            Tuple of mean and std across image channels
    """
    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    n_batches = 0
    
    for data, _ in tqdm(loader, desc="Computing mean and std"):
        channel_sum += torch.mean(data, dim=[0, 2, 3])
        channel_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        n_batches += 1
        
    mean = channel_sum / n_batches
    std = (channel_squared_sum / n_batches - (mean **2)).sqrt()
    
    return mean, std

