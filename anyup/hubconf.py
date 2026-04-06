dependencies = ['torch']

from anyup.model import AnyUp
import torch


def anyup(pretrained: bool = True, device='cpu'):
    """
    AnyUp model trained on DINOv2 ViT-S/14 features, used in most experiments of the paper.
    Note: If you want to use vis_attn, you also need to install matplotlib.
    """
    model = AnyUp().to(device)
    if pretrained:
        checkpoint = "https://github.com/wimmerth/anyup/releases/download/checkpoint/anyup_paper.pth"
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True, map_location=device))
    return model
