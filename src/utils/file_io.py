from pickle import load, dump
import logging
import yaml

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """
    This function loads the configuration file from the specified path.

    Args:
        config_path (str): 
            Path to configuration file.

    Returns:
        dict: 
            loaded configurations dictionary.
    """
    logger.info(f"Loading config from {config_path}.")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Config loaded successfully.")
    return config
    

def save_objects(save_path: str, object):
    """
    This function saves the specified `object` in the destination path `save_path`.

    Args:
        save_path (str): 
            Path to save the object to (including file name).
        object (_type_): 
            Object to be saved.
    """
    logger.info(f"Saving object to {save_path}.")
    dump(object, open(save_path, "wb"))
    
def load_objects(file_path: str):
    """
    This function loads the object from the specified path

    Args:
        file_path (str): 
            Path to saved file
    """
    logger.info(f"Loading object from {file_path}")
    load(open(file_path, "rb")) 