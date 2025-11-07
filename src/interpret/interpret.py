from src.utils.file_io import load_saved_mean_std
import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)

class BrainGradCAM:
    """
    The class abstracts the compuation of the class acivation mappings, and the needed utilities
    for plotting the results.
    
    Attributes:
        model (torch.nn.Module):
            Model we wish to interpret
        target_layer_name (str):
            Name of the target convolutional layer. Usually, the last layer (conv5 in our case).
        mean_std_save_path (str):
            Path to the saved mean and std
        device:
            Device on which to perform the computations
            
    Methods:
        _get_target_layer(self) -> torch.nn.Module:
            Returns the layer with the specified name
            
        compute_grad_cam(self, img_tensor: torch.Tensor) -> tuple[np.ndarray, int]:
            Computes the Class Activation Mappings (CAM)
            
        def apply_colormap_on_img(self, orig_img: torch.Tensor, cam: np.ndarray, colormap = cv2.COLORMAP_JET, 
                              alpha: float = 0.4) -> np.ndarray:
            Superimposes the heatmap from the `CAM` onto the input image
            
        plot_overlays(self, orig_img: torch.Tensor, cam: np.ndarray, class_list: list, predicted_label: int | None = None, 
                      true_label: int | None = None, save_name: str = "") -> None:
            Plots the input image, computed `CAM` heatmap and the superimposed image.
        
    """
    def __init__(self, model: torch.nn.Module, target_layer_name: str, 
                 mean_std_save_path="artifacts/preprocessing/normalization_mean_std.json", 
                 device: str | torch.device = "cpu") -> None:
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer_name
        self.device = device
        self.activations = None
        self.gradients = None
        self.mean, self.std = load_saved_mean_std(mean_std_save_path=mean_std_save_path)
        
        
        self.model.to(self.device)
        self.target_layer = self._get_target_layer()
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradients)
        
    def _get_target_layer(self) -> torch.nn.Module:
        """
        Returns the target layer that matches the name specified

        Raises:
            ValueError: 
                Raised if no layer has the name specified

        Returns:
            torch.nn.Module: 
                Returns the target layer specified
        """
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                return module

        logger.error(f"The specified target layer ({self.target_layer_name}) does not exist")
        raise ValueError(f"The specified target layer ({self.target_layer_name}) does not exist")
    
    def _save_activation(self, model, input, output):
        """
        Assigns the output of the forward pass (feature maps) to the activations variable.
        """
        self.activations = output.detach()
        
    def _save_gradients(self, model, input, output):
        """
        Assigns the output of the backward pass (gradients) to the gradients variable
        """
        self.gradients = output[0].detach()
        
    def compute_grad_cam(self, img_tensor: torch.Tensor) -> tuple[np.ndarray, int]:
        """
        Computes and returns the class activation mapping (CAM).

        Args:
            img_tensor (torch.Tensor): 
                Tensor of the imput image

        Raises:
            ValueError: 
                Raised if `self.gradients` is `None`.

        Returns:
            tuple[np.ndarray, int]: 
                Returns the computed CAM.
        """
        img_tensor = img_tensor.to(self.device)
        img_tensor = img_tensor.unsqueeze(0)
        self.model.zero_grad()
        output = self.model(img_tensor)
        target_class = int(torch.argmax(output, dim=1).item())
        score = output[0, target_class]
        score.backward(retain_graph = True)
        
        activations = self.activations
        grads = self.gradients
        
        if grads is not None:
            logger.info("Computing Class Activation Mappings (CAM)")
            weights = grads.mean(dim=(2,3), keepdims=True)
            weighted_activations = (weights * activations).sum(dim=1, keepdims=True)
            cam = weighted_activations.squeeze().cpu().numpy()
            cam = np.maximum(cam, 0)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            return np.array(cam), target_class
        else:
            logger.error(f"Gradients cannot be None: {grads}")
            raise ValueError(f"Gradients cannot be None: {grads}")
        
    def apply_colormap_on_img(self, orig_img: torch.Tensor, cam: np.ndarray, colormap = cv2.COLORMAP_JET, 
                              alpha: float = 0.4) -> np.ndarray:
        """
        This method superimposes the computed class activation heatmap onto the input image

        Args:
            orig_img (torch.Tensor): 
                Input image tensor
            cam (np.ndarray): ,
                Computed class activation mapping
            colormap (cv2.COLORMAP_JET, optional): _
                Colormap of the CAM heatmap. Defaults to cv2.COLORMAP_JET.
            alpha (float, optional): 
                Superposition ratio. Defaults to 0.4.

        Returns:
            np.ndarray: 
                Array of input image superimposed with heatmap
        """
        logger.info("Superimposing CAM heatmap with input image")
        orig_img_np = orig_img.detach().cpu().permute(1, 2, 0).numpy()
        orig_img_np = orig_img_np * np.array(self.mean) + np.array(self.std)
        orig_img_np = np.clip(orig_img_np, 0, 1)
        orig_img_np = (255 * orig_img_np).astype(np.uint8)
        orig_img_np = cv2.cvtColor(orig_img_np, cv2.COLOR_RGB2BGR)     
        heatmap = cv2.resize((cam*255).astype(np.uint8), (orig_img_np.shape[0], orig_img_np.shape[1]))
        heatmap = cv2.applyColorMap(heatmap, colormap)
        overlay = cv2.addWeighted(orig_img_np, alpha, heatmap, 1 - alpha, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        return overlay
    
    def plot_overlays(self, orig_img: torch.Tensor, cam: np.ndarray, class_list: list, overlay: np.ndarray,
                      predicted_label: int | None = None, true_label: int | None = None, save_name: str = ""):
        """
        This method plots the original image, heatmap and superimposed image in the same figure.

        Args:
            orig_img (torch.Tensor): 
                Input image tensor
            cam (np.ndarray): 
                Class Activation Mapping array
            predicted_label (int): 
                Index of predicted class label
            true_label (int): 
                Index of actual class label
            class_list (list): 
                Class names
            save_name (str, optional): 
                Name used to save the resulting graph. Defaults to "".
        """        
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        orig_img = orig_img * std + mean
        orig_img = orig_img.clip(0, 1)
        
        orig_img_np = orig_img.detach().cpu().permute(1, 2, 0).numpy()

        heatmap = cv2.resize(cam, (orig_img_np.shape[1], orig_img_np.shape[0]))

        title = ""
        title += f"Predicted: {class_list[predicted_label]}" if predicted_label is not None else ""
        title =  title + f" | True: {class_list[true_label]}" if true_label is not None else title        

        logger.info("Plotting GRAD-CAM results")
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(orig_img_np)
        axes[0].set_title(f"Original Image")
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(f"Heatmap Superimposed Image")
        axes[2].axis("off")
        plt.suptitle(title, fontsize=14, fontweight='bold')

        plt.tight_layout()
        if save_name:
            logger.info(f"Saving GRAD-CAM visualization: artifacts/graphs/{save_name}.png")
            plt.savefig(f"artifacts/graphs/{save_name}.png")
        plt.show()

        return fig