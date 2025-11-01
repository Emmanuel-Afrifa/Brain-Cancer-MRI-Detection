from pathlib import Path
from pickle import load, dump
import csv
import json
import logging
import os
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
    
def save_predictions(pred_labels: list, pred_probs: list, image_paths: list[str], class_names: list, 
                     metrics: tuple | None = None, save_name: str = "") -> dict:
    """
    This function saves the predictions, predicted probabilities and metrics of the predictions on
    `image_paths` to `artifacts/results/{save_name}`

    Args:
        pred_labels (list): 
            Prediced labels
        pred_probs (list): 
            Predicted probabilities
        image_paths (list[str]): 
            Image paths
        class_names (list): 
            Names of labels in the dataset
        metrics (tuple | None, optional): 
            Metrics of the model evaluation. Defaults to None.
        save_name (str, optional): 
            Name used to save the results. Defaults to "".

    Returns:
        dict: 
            Returns csv and json saved file paths
    """
    
    out_dir = "artifacts/results"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{save_name}.csv")
    
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = ["image_path", "pred_label"] + [f"prob_{cls}" for cls in class_names]
        writer.writerow(header)

        logger.info(f"Saving results (in csv format) to {out_dir}")
        for path, label, probs in zip(image_paths, pred_labels, pred_probs):
            row = [path, class_names[label]] + [round(p, 4) for p in probs]
            writer.writerow(row)
    
    json_data = {
        "predictions": [
            {
                "image_path": path,
                "pred_label": class_names[label],
                "probabilities": {cls: round(p, 4) for cls, p in zip(class_names, probs)},
            }
            for (path, label, probs) in zip(image_paths, pred_labels, pred_probs)
        ]
    }

    
    if metrics:
        overall, per_class = metrics
        json_data.setdefault("metrics", []).append({
            "overall": overall,
            "per_class": per_class
        })
    
    logger.info(f"Saving the results (in json format) to {out_dir}")
    json_path = os.path.join(out_dir, f"{save_name}.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=4)

    return {"csv": csv_path, "json": json_path}