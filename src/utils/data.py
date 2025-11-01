import glob
import os
import logging

logger = logging.getLogger(__name__)

def get_image_paths(path: str) -> list:
    """
    This function parses the given path and returns all the image paths in the directory

    Args:
        path (str): 
            Path to image files

    Raises:
        ValueError: 
            Raised when the path provided does not exist.

    Returns:
        list: 
            List of names of image files in the specified path.
    """
    if os.path.join(path):
        exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(path, ext)))
        return files
    elif os.path.isfile(path):
        return [path]
    else:
        logger.error(f"The specified path does not exist: {path}")
        raise ValueError(f"The specified path does not exist: {path}")