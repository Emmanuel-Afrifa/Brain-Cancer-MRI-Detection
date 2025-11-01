"""
Brain Cancer MRI detection (ML-Approach)
============================================

This project seeks to train machine learning models capable for classifying brain cancer with a high
degree of accuracy.

This package provides tools for:
- Data preparation and loading
- Model definition
- Training with logging, checkpointing, and early stopping
- Evaluation and inference
"""

__version__ = "0.1.0"
__author__ = "Emmanuel Afrifa"
__description__ = "Brain Cancer MRI detection using Machine Learning"

# data
from .data.data_module import BrainMRIDataModule
from .data.data_splitter import DataSplitter

# evaluation
from .evaluation import metrics

# model
from .models import BrainScanCNN

# preprocessing
from .preprocessing.convert_to_RGB import ConvertToRGB
from .preprocessing.preprocessing import compute_mean_std


# training
from .training.trainer import ModelTrainer
from .training.callbacks import earlystopping, checkpointing
from .training.optimizer import get_optimizer, get_lr_scheduler

# inference


# utils
from .utils.file_io import save_objects, load_config, load_objects
from .utils.logger import setup_logging
from .utils.seed import set_seed


__all__ = [
    "__version__",
    "BrainMRIDataModule", 
    "DataSplitter",
    "metrics",
    "BrainScanCNN",
    "ConvertToRGB",
    "compute_mean_std",
    "ModelTrainer",
    "earlystopping",
    "checkpointing",
    "get_optimizer",
    "get_lr_scheduler",
    "save_objects",
    "load_config",
    "load_objects",
    "setup_logging",
    "set_seed"    
]