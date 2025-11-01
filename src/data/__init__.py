"""
This submodule contains the classes and functions for managing and preparing data for analysis
- BrainMRIDataModule: Handles the loading of the data, transforming it and creation of data loaders.
- DataSplitter: Abstracts the logic for splitting the data into training, validation and test sets.
"""

from .data_module import BrainMRIDataModule
from .data_splitter import DataSplitter

__all__ = [
    "BrainMRIDataModule",
    "DataSplitter"
]