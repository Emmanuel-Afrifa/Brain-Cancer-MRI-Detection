from pathlib import Path
from src.preprocessing.convert_to_RGB import ConvertToRGB
from src.preprocessing.preprocessing import compute_mean_std
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import json
import logging
import os
import torch

logger = logging.getLogger(__name__)

class BrainMRIDataModule:
    def __init__(self, config) -> None:
        self.config = config
        self.img_size = self.config["data"]["img_size"]
        self.seed = self.config["seed"]
        self.batch_size = self.config["train"]["batch_size"]
        self.preprocessed_data_dir = Path(self.config["data"]["preprocessed"])
        self.generator = torch.Generator().manual_seed(self.seed)
        self.mean = None
        self.std = None
        
    def _build_transforms(self, mean: Tensor, std: Tensor,  training: bool = False):
        """
        This function builds the transforms to be applied to the datasets.

        Args:
            mean (Tensor): 
                Computed mean per image channel
            std (Tensor): 
                Computed standard deviation (std) per image channed
            training (bool, optional): 
                Determines if the training transform should  be used or not. It includes transformations
                like random rotations, flips, etc. to enhance model generalization. Defaults to False.

        Returns:
            A transform object.
        """
        if training:
            transform = transforms.Compose([
                ConvertToRGB(),
                transforms.Resize(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),
                transforms.RandomResizedCrop(self.img_size, (0.95, 1.05)),
                transforms.ColorJitter(contrast=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])  
        else: 
            transform = transforms.Compose([
                ConvertToRGB(),
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        return transform
        
    def compute_train_stats(self, train_path: str | Path, save_mean_std: bool, save_mean_std_path: str) -> tuple[Tensor, Tensor]:
        """
        This function computes the mean and standard deviation from the training set

        Args:
            train_path (str | Path): 
                Pathe to training data
            save_mean_std (bool): 
                Determines whether the computed mean and std should be saved or not.
            save_mean_std_path (str): 
                If `save_mean_std` is set to True, it specifies path to save the results.

        Raises:
            ValueError: 
                Raised when `save_mean_std` is set to `True`, but no path (`save_mean_std_path`) is specified.

        Returns:
            tuple[Tensor, Tensor]: 
                Computed mean and standard deviation.
        """
        transform = transforms.Compose([
            ConvertToRGB(),
            transforms.Resize(self.img_size),
            transforms.ToTensor()
        ])
        train_ds = ImageFolder(train_path, transform=transform)
        train_dl = DataLoader(train_ds, shuffle=False, batch_size=self.batch_size, generator=self.generator)
        logging.info(f"Computing mean and std for the training set ({train_path}).")
        self.mean, self.std = compute_mean_std(train_dl)
        
        if save_mean_std:
            if save_mean_std_path:
                with open(save_mean_std_path, "w") as f:
                    json.dump({"mean": self.mean.tolist(), "std": self.std.tolist()}, f)
            else:
                logging.error(f"If `save_mean_std` is set to `True`, then `save_mean_std_path` must not be an empty string.")
                raise ValueError(f"If `save_mean_std` is set to `True`, then `save_mean_std_path` must not be an empty string.")
        return self.mean, self.std

    def get_datasets(self, save_mean_std_path="") -> tuple:
        """
        Loads the training, validation and test datasets, applying the appropriate transformations.

        Args:
            save_mean_std_path (str, optional): 
                Specifies the path to load or save the computed mean and std. Defaults to "".

        Returns:
            tuple: 
                The training, validation and test datasets.
        """
        if os.path.exists(save_mean_std_path):
            with open(save_mean_std_path, "r") as f:
                logger.info("Loading the saved mean and std")
                saved_mean_std = json.load(f)
                self.mean, self.std = saved_mean_std["mean"], saved_mean_std["std"]
        else:
            self.mean, self.std = self.compute_train_stats(self.preprocessed_data_dir / "train", save_mean_std=True, save_mean_std_path=save_mean_std_path)
        
        transforms_training = self._build_transforms(self.mean, self.std, training=True)
        transform_other = self._build_transforms(self.mean, self.std, training=False)
        
        train_dataset = ImageFolder(self.preprocessed_data_dir / "train", transform=transforms_training)
        val_dataset = ImageFolder(self.preprocessed_data_dir / "val", transform=transform_other)
        test_dataset = ImageFolder(self.preprocessed_data_dir / "test", transform=transform_other)
        
        return train_dataset, val_dataset, test_dataset
        
    def get_dataloaders(self) -> tuple:
        """
        This function returns the data loaders for the three datasets (train, val and test sets).

        Returns:
            tuple: 
                Tuple of the data loaders for the train, val and test sets respectively.
        """
        train_ds, val_ds, test_ds = self.get_datasets(save_mean_std_path="artifacts/preprocessing/normalization_mean_std.json")
        return (
            DataLoader(train_ds, batch_size=self.batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, generator=self.generator),
            DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, generator=self.generator)
        )
    