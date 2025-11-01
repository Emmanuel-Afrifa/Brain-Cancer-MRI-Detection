from tqdm import tqdm
from torch.nn.functional import softmax
import torch

def predict(data_loader, model: torch.nn.Module, device: str | torch.device = "cpu") -> tuple:
    """
    This function uses the `model` to make predictions on the `data loader`.

    Args:
        data_loader (torch.utils.data.DataLoader): 
            Data loader whose labels are to be predicted.
        model (torch.nn.Module): 
            Model to be used for prediction.
        device (str | torch.device, optional): 
            Device on which to perform the computations. Defaults to "cpu".

    Returns:
        tuple: 
            Predicted classed and the prediction probabilities for all classes.
    """
    
    model.eval()
    
    all_preds = []
    all_pred_probs = []
    
    with torch.no_grad():
        for data, _ in tqdm(data_loader, desc="Predicing"):
            data.to(device)
            output = model(data)
            probs = softmax(output, dim=1)
            all_pred_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy().astype(int).tolist())
        
    return all_preds, all_pred_probs
            