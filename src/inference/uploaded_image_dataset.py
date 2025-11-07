from PIL import Image
from src.utils.file_io import load_saved_mean_std
from torch.utils.data import TensorDataset
from torchvision import transforms
import io
import torch

def get_uploaded_image_data(img, img_size=[224,224]): 
    img.seek(0)
    mean, std = load_saved_mean_std("artifacts/preprocessing/normalization_mean_std.json")
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    image = Image.open(io.BytesIO(img.read())).convert("RGB")
    img_tensor: torch.Tensor = transform(image).unsqueeze(0) # type: ignore
    return TensorDataset(img_tensor, torch.zeros((1,), dtype=torch.long))