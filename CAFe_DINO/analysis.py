import sys
sys.path.append('..')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import cv2
import tifffile as tif
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TVT
import torchvision.transforms.functional as TVTF

import albumentations as A
from rich import print
import matplotlib.pyplot as plt
import argparse
from omegaconf import OmegaConf

from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l

from CAFe_DINO.modeling.cafedino import CAFe_DINO
from anyup.anyup.model import AnyUp
from val_data import *
from utils import build_text_embeddings, strided_inf, plot, overlay_segmentation
torch.set_float32_matmul_precision('high')

cfg = OmegaConf.load("./configs/config_cocostuff_subset.yaml")

device = "cuda"

INPUT_SIZE = 224

@torch.no_grad()
def get_raw_cost_vol(model, img, text_emb):
    B, _, imgH, imgW = img.shape
    H, W, D = imgH // 16, imgW // 16, 1024

    patch_tokens = model.encode_patches(img) # [P, D]
    patch_tokens = patch_tokens.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)

    # assert 1 == 0, (patch_tokens.shape, text_emb.shape)
    raw_cost_vol = torch.einsum('bdhw,bcd->bchw', F.normalize(patch_tokens, dim=1), F.normalize(text_emb, dim=-1))
    return raw_cost_vol

parser = argparse.ArgumentParser(description="Read a single string from the command line")
parser.add_argument("--weights", type=str, help="Input string", required=True)
parser.add_argument("--image", type=str, help="Input string", required=True)

args = parser.parse_args()

batch_size = 1
backbone, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l()
backbone.to(device).eval()

upsampler = torch.hub.load("wimmerth/anyup", "anyup", verbose=False).to(device).eval()

model = CAFe_DINO(backbone, tokenizer, upsampler, input_resolution=(INPUT_SIZE // 16, INPUT_SIZE // 16), device=device, aggregator_dim=cfg.aggregator_dim)

model.to(device)
# model = torch.compile(model)
sd = torch.load(args.weights)
print(sd.keys())
clean_state_dict = {
    k.replace("_orig_mod.", ""): v for k, v in sd["model"].items()
}
model.load_state_dict(clean_state_dict)
model.eval()

class ShortSideResize(nn.Module):
    def __init__(self, size: int, interpolation: TVT.InterpolationMode) -> None:
        super().__init__()
        self.size = size
        self.interpolation = interpolation

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = TVTF.get_dimensions(img)
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img
        if w < h:
            new_w = self.size
            new_h = int(self.size * h / w)
            return TVTF.resize(img, [new_h, new_w], self.interpolation)
        else:
            new_h = self.size
            new_w = int(self.size * w / h)
            return TVTF.resize(img, [new_h, new_w], self.interpolation)


resize = 512
strided = True
side = 128  # window size
stride = 32

NORMALIZE_IMAGENET = TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


transform = A.Compose([
    A.Resize(INPUT_SIZE, INPUT_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Have to ImageNet normalize for AnyUp
    A.pytorch.ToTensorV2()
])

transform_no_resize = A.Compose([
    A.Resize(resize, resize),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Have to ImageNet normalize for AnyUp
    A.pytorch.ToTensorV2()
])

OEM_CLASS_NAMES[0].remove("background")
VAIHINGEN_CLASS_NAMES[0].remove("background")
LOVE_DA_CLASS_NAMES[0].remove("background")

img_path = args.image

if "vaihingen" in img_path or "potsdam" in img_path:
    class_names = VAIHINGEN_CLASS_NAMES
    num_classes = VAIHINGEN_NUM_CLASSES - 1
elif "loveda" in img_path or "loveDA" in img_path:
    class_names = LOVE_DA_CLASS_NAMES
    num_classes = LOVEDA_NUM_CLASSES - 1
elif "openearthmap" in img_path:
    class_names = OEM_CLASS_NAMES
    num_classes = OEM_NUM_CLASSES - 1
elif "oem" in img_path:
    class_names = OEM_CLASS_NAMES
    num_classes = OEM_NUM_CLASSES - 1
else:
    class_names = VAIHINGEN_CLASS_NAMES
    num_classes = VAIHINGEN_NUM_CLASSES - 1

#OVERRIDE
class_names = [['boat', 'dock', 'tree', 'grass', 'building', 'pavement']]
num_classes = len(class_names[0])
    
print(class_names, num_classes)


with torch.no_grad():
    text_emb = build_text_embeddings(backbone, tokenizer, class_names_list=class_names)

img_raw = cv2.imread(img_path)
print(img_raw.shape)
# img_raw = img_raw[512:, :512, :]
full_res_img = transform_no_resize(image=img_raw)["image"]
full_res_img = full_res_img[None, :, :, :].to(device)
# raw_tensor = torch.tensor(img_raw, device='cuda', dtype=torch.float).moveaxis(-1, 0).unsqueeze(0)

img = transform(image=img_raw)["image"]
img = img[None, :, :, :].to(device)

with torch.no_grad():
    if strided:
        aggregated_cost_vol = strided_inf(model, full_res_img, text_emb, side, stride, num_classes)
    else:
        aggregated_cost_vol = model(img, text_emb, full_res_image=full_res_img)[0]


plot_img = full_res_img.detach().cpu().numpy()[0]
img_raw = cv2.resize(img_raw, (resize, resize))
plot(num_classes, class_names[0], img_raw, aggregated_cost_vol, aggregated_cost_vol, filename="comparison.png")

overlay_segmentation(img_raw, torch.argmax(aggregated_cost_vol, dim=0).squeeze(0).cpu().numpy(), class_names, num_classes)

def scale_to_255(x):
    x_min, x_max = x.min(), x.max()
    if x_max == x_min:
        return np.zeros_like(x, dtype=np.uint8)  # avoid division by zero
    x_scaled = (x - x_min) / (x_max - x_min) * 255
    return x_scaled.astype(np.uint8)

for i, costmap in enumerate(aggregated_cost_vol.cpu().numpy()):
    print(i)
    # print(np.unique(costmap))
    plt.imshow(costmap, cmap='magma')
    # plt.axis('off')  # hide axes
    # plt.colorbar()   # optional

    # Save figure
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'./plots/costmap_{i}.png', dpi=300, bbox_inches='tight', pad_inches=0)