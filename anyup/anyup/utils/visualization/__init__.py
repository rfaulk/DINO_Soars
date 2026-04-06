import torch
from .attention_visualization import visualize_attention_oklab

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def unnormalize(t, mean=None, std=None):
    if mean is None: mean = IMAGENET_MEAN
    if std is None: std = IMAGENET_STD
    m = torch.as_tensor(mean, device=t.device, dtype=t.dtype).view(1, -1, 1, 1)
    s = torch.as_tensor(std, device=t.device, dtype=t.dtype).view(1, -1, 1, 1)
    return t * s + m