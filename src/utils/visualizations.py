from collections import Counter
from matplotlib.patches import Rectangle
from PIL import Image
from tqdm import tqdm
from typing import Literal
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch


def class_counter(dataset, title: str, savename: str = "") -> pd.Series:
    """
    This function counts the number of data points in each class for the specified dataset, plots the
    results and returns a pandas series of the class counts.

    Args:
        dataset (torch.utils.data.Dataset): 
            Dataset whose classes we wich to count
        title (str): 
            Title of the plot
        savename (str, optional): 
            Name used to save the graph. Defaults to "".

    Returns:
        pd.Series: 
            Pandas series of the class counts.
    """
    class_counts = Counter(labels for _, labels in tqdm(dataset))
    class_to_index = dataset.class_to_idx
    class_counts_df = pd.Series({cls: class_counts[idx] for cls, idx in class_to_index.items()})
    
    ax = class_counts_df.sort_values().plot(kind="bar", figsize=(5, 5))
    for p in ax.patches:
        if isinstance(p, Rectangle):
            height = p.get_height()
            ax.text(x=p.get_x() + p.get_width() / 2, y=height, s=f"{int(height)}", ha='center', va='bottom')
            
    plt.title(title)
    plt.xlabel("Labels")
    plt.ylabel("Counts")
    plt.xticks(rotation=15)
    if savename:
        plt.savefig(f"artifacts/graphs/{savename}.png")
    plt.show()
    return class_counts_df



def plot_raw_vs_transformed_imgs(dataset, mean: list, std: list, config: dict, n_samples: int = 10, save_name: str = "") -> None:
    """
    This function randomly choses `n_samples` from the dataset, plots the raw image next to the transformed
    image to visualize the augmentations applied to the images.

    Args:
        dataset (torch.utils.data.Dataset):
            Dataset 
        mean (list): 
            Computed mean during training (used for denormalizing transformed images)
        std (list): 
            Computed std during training (used for denormalizing transformed images)
        config (dict):
            Global configurations
        n_samples (int, optional): 
            Number of samples to choose from the dataset. Defaults to 10.
        save_name (str, optional): 
            Name used to save the model. Defaults to "".
    """
    g = torch.Generator().manual_seed(88)
    indices = torch.randint(0, len(dataset), (n_samples,), generator=g)
    
    print(mean, std)
    plt.figure(figsize=(6, n_samples*2))
    
    for i, idx in enumerate(indices):
        img_path, _ = dataset.samples[idx]
        transformed_img, _ = dataset[idx]
        
        transformed_img = transformed_img.permute(1, 2, 0)
        # possible convert std and mean to tensors
        transformed_img = transformed_img * torch.tensor(std) + torch.tensor(mean)
        transformed_img = transformed_img.clip(0, 1)
        
        raw_img = Image.open(img_path).convert("RGB")
        # plot both
        plt.subplot(n_samples, 2, 2*i + 1)
        plt.imshow(raw_img)
        plt.title(f"Raw: {os.path.basename(img_path)}")
        plt.axis("off")

        plt.subplot(n_samples, 2, 2*i + 2)
        plt.imshow(transformed_img)
        plt.title("Transformed")
        plt.axis("off")
    plt.tight_layout()
    if save_name:
        plt.savefig(f"artifacts/graphs/{save_name}.png")
    plt.show()
    

metric_literals = Literal["loss", "acc", "f1_macro", "f1_weighted"]
    
def plot_train_vs_val(history: dict, metric: metric_literals = "loss", title: str = "", save_name: str = "", 
                      y_label: str = "", color: tuple[str, str] | list[str] = ["r", "k"]) -> None:
    """
    This function plots the training vs validation history

    Args:
        history (dict): 
            Dictionary of training history
        metric (metric_literals, optional): 
            Suffice of the specific metric to be plotted. Defaults to "loss". Options: ["loss", "acc", "f1_macro", "f1_weighted"]
        title (str, optional): 
            Title of the graph. Defaults to "".
        save_name (str, optional): 
            Name used to save the graph. Defaults to "".
        y_label (str, optional): 
            Label of the y-axis of the graph. Defaults to "".
        color (tuple[str, str] | list[str], optional): 
            Colors used for the training and validation plots. Defaults to ["r", "k"].
    """
    plt.plot(history[f"train_{metric}"], color=color[0], label=f"training_{metric}")
    plt.plot(history[f"val_{metric}"], color=color[1], label=f"val_{metric}")
    plt.xlabel("Epochs")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    if save_name:
        plt.savefig(f"artifacts/graphs/{save_name}.png")
    plt.show()
    

def plot_learning_rates(history: dict, color: str = "b", marker: str = "*", markersize: int = 10, title: str = "", 
                        save_name: str = "") -> None:
    """
    This function plots the learning rates across the trained epochs.

    Args:
        history (dict): 
            Training history
        color (str, optional): 
            Color of the plot. Defaults to "b".
        marker (str, optional): 
            Point marker. Defaults to "*".
        markersize (int, optional): 
            Size of the marker. Defaults to 10.
        title (str, optional): 
            Title of the graph. Defaults to "".
        save_name (str, optional): 
            Name used to save the graph. Defaults to "".
    """
    plt.plot(history["learning_rates"], color=color, marker=marker, markersize=markersize)
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rates")
    plt.title(title)
    if save_name:
        plt.savefig(f"artifacts/graphs/{save_name}.png")
    plt.show()