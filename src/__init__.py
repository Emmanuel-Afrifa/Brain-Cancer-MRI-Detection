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

__version__ = "1.0.0"
__author__ = "Emmanuel Afrifa"
__description__ = "Brain Cancer MRI detection using Machine Learning"

# data
from .data.data_module import BrainMRIDataModule
from .data.data_splitter import DataSplitter

# evaluation
from .evaluation import metrics

# inference
from .inference.inference_dataset import InferenceDataset
from .inference.inference_loader import InferenceDataLoader
from .inference.predict import predict
from .inference.uploaded_image_dataset import get_uploaded_image_data

# Interpret
from .interpret.interpret import BrainGradCAM

# model
from .models import BrainScanCNN

# preprocessing
from .preprocessing.convert_to_RGB import ConvertToRGB
from .preprocessing.preprocessing import compute_mean_std

# training
from .training.trainer import ModelTrainer
from .training.callbacks import earlystopping, checkpointing
from .training.optimizer import get_optimizer, get_lr_scheduler

# utils
from .utils.file_io import save_objects, load_config, load_objects, save_predictions
from .utils.logger import setup_logging
from .utils.seed import set_seed
from .utils.visualizations import class_counter, plot_raw_vs_transformed_imgs, plot_train_vs_val, plot_learning_rates


__all__ = [
    "__version__",
    "BrainMRIDataModule", 
    "DataSplitter",
    "metrics",
    "InferenceDataset",
    "InferenceDataLoader",
    "predict",
    "get_uploaded_image_data",
    "BrainGradCAM",
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
    "save_predictions",
    "setup_logging",
    "set_seed",
    "class_counter",
    "plot_raw_vs_transformed_imgs",
    "plot_train_vs_val",
    "plot_learning_rates"
]