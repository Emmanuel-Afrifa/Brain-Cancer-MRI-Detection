from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import json
import logging
import os
import torch

logger = logging.getLogger(__name__)

class InferenceDataset(Dataset):
    """
    This function creates an Dataset object for the data on which we wish to make predictions.

    Attributes:
        image_dir (str): 
            Path to image directory.
        mean (list):
            Computed per-channel mean.
        std (list):
            Computed per-channed std.
        config (dict):
            Global configurations.
    """
    def __init__(self, image_dir: str, mean: list | torch.Tensor, std: list | torch.Tensor, config: dict) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.config = config
        self.img_size = config["data"]["img_size"]
        self.img_paths = [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        if not self.img_paths:
            logger.error(f"No valid image files found in: {image_dir}")
            raise ValueError(f"No valid image files found in: {image_dir}")
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, os.path.basename(img_path)