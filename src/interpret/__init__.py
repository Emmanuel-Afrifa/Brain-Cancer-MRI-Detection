"""
This submodule contains the class and methods for computing and visualizing the class activation maps
using the Gradient Weighted Class Activation Mapping (GRAD-CAM) approach.
- BrainGradCAM: The class abstracts the compuation of the class acivation mappings, and the needed utilities
    for plotting the results.
"""

from .interpret import BrainGradCAM