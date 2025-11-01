"""
This submodule abstracts the creation of the dataset and data loaders for the data to be predicted. 
It also has the predict function, which can be used for making inference.
- InferenceDataset: Abstracts the creation of the inference dataset.
- InferenceDataLoader: Returns the data loader for the inference dataset.
- predict: Uses the trained model to make predictions of the data loader.
"""

from .inference_dataset import InferenceDataset
from .inference_loader import InferenceDataLoader
from .predict import predict

__all__ = [
    "InferenceDataset",
    "InferenceDataLoader",
    "predict"
]