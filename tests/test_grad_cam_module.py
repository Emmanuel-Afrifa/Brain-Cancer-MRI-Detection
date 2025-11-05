from src.interpret.interpret import BrainGradCAM
from src.models.base_model import BrainScanCNN
import numpy as np
import torch
import pytest

model = BrainScanCNN(num_classes=3)
brain_cam = BrainGradCAM(model, "conv5")

def test_grad_cam_layer():
    assert brain_cam._get_target_layer() == model.conv5
    
def test_grad_cam_hooks_internal_states():
    x = torch.randn([1, 3, 224, 224])
    
    outputs = model(x)
    pred_class = outputs.argmax()
    score = outputs[0, pred_class]
    score.backward()
    
    assert brain_cam.activations is not None
    assert brain_cam.gradients is not None
    assert brain_cam.activations.shape[1:] == brain_cam.gradients.shape[1:]
        
def test_grad_cam_compute():
    x = torch.randn([3, 224, 224])
    cam, target_class = brain_cam.compute_grad_cam(x)
    
    assert isinstance(cam, np.ndarray)
    assert isinstance(target_class, int)
    assert cam.ndim == 2
    assert 0 <= cam.min() <= cam.max() <= 1
    
def test_grad_cam_overlay():
    orig_img = torch.randn([3, 224, 224])
    cam = np.abs(np.random.randn(14, 14)).astype(np.float32)
    
    overlay = brain_cam.apply_colormap_on_img(orig_img, cam)
    
    assert overlay.shape == (224, 224, 3)
    

        