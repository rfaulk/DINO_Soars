import sys
sys.path.append('/home/rfaulken/dinov3')
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import datetime
import torch
from torch.utils.data import DataLoader

import albumentations as A

from rich import print
from rich.console import Console

from omegaconf import OmegaConf

from data.vaihingen import VaihingenDataset
from data.potsdam import PotsdamDataset
from data.loveda import LoveDADataset
from data.oem import OpenEarthMapDataset

from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l

import argparse

from utils import log_print, logger, validate

from modeling.cafedino import CAFe_DINO
from anyup.anyup.model import AnyUp
from val_data import *
torch.set_float32_matmul_precision('high')

DEVICE = "cuda"

INPUT_SIZE = 224
STRIDED = True

def geobench_collate_fn(batch):
    images = torch.stack([torch.tensor(b.pack_to_3d(("red", "green", "blue"))[0]) for b in batch])
    labels = torch.stack([torch.tensor(b.label.data) for b in batch])
    return {"image": images, "label": labels}

# class GeoBenchTransform(torch.nn.Module):
#     def forward(self, img):
#         new_img = torch.tensor(img.pack_to_3d(("red", "green", "blue"))[0])
#         new_label = torch.tensor(img.label.data)
#         return new_img#, new_label

def val_suite(model):
    oem_num_classes = OEM_NUM_CLASSES# - 1
    vaihingen_num_classes = VAIHINGEN_NUM_CLASSES# - 1
    loveda_num_classes = LOVEDA_NUM_CLASSES# - 1
    OEM_CLASS_NAMES[0]#.remove("background")
    VAIHINGEN_CLASS_NAMES[0]#.remove("background")
    LOVE_DA_CLASS_NAMES[0]#.remove("background")

    # print(OEM_CLASS_NAMES, oem_num_classes)
    # miou_agg = 0
    for thresh in [0.1, 0.2, 0.3, 0.4]:
        validate(model, tokenizer, val_loader_vaihingen, DEVICE, vaihingen_num_classes, class_names=VAIHINGEN_CLASS_NAMES, save_path="vaihingen_confmat.png", strided=STRIDED, ignore_index=[255], thresh=thresh, bg_idx=5)
    for thresh in [0.1, 0.2, 0.3, 0.4]:
        validate(model, tokenizer, val_loader_oem, DEVICE, oem_num_classes, class_names=OEM_CLASS_NAMES, save_path="oem_confmat.png", strided=STRIDED, ignore_index=[255], thresh=thresh, bg_idx=0)
    for thresh in [0.1, 0.2, 0.3, 0.4]:
        validate(model, tokenizer, val_loader_potsdam, DEVICE, vaihingen_num_classes, class_names=VAIHINGEN_CLASS_NAMES, save_path="potsdam_confmat.png", strided=STRIDED, ignore_index=[255], thresh=thresh, bg_idx=5)
    for thresh in [0.5, 0.6, 0.7]:
        validate(model, tokenizer, val_loader_loveda, DEVICE, loveda_num_classes, class_names=LOVE_DA_CLASS_NAMES, save_path="loveda_confmat.png", strided=STRIDED, ignore_index=[255], thresh=thresh, bg_idx=0)

parser = argparse.ArgumentParser(description="Read a single string from the command line")
parser.add_argument("--config", type=str, help="Input string")
parser.add_argument("--model", type=str, help="weights")

args = parser.parse_args()

config_file = args.config
weights_path = args.model
print("Config:", config_file)
print("Weights:", weights_path)

cfg = OmegaConf.load(config_file)

cfg = OmegaConf.load(config_file)

batch_size = 1
backbone, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l()
backbone.to(DEVICE)

upsampler = AnyUp()
upsampler.load_state_dict(torch.load("/home/rfaulken/dinov3/weights/anyup_paper.pth"))
upsampler.eval()
model = CAFe_DINO(backbone, tokenizer, upsampler, input_resolution=(INPUT_SIZE // 16, INPUT_SIZE // 16), device=DEVICE, aggregator_dim=cfg.aggregator_dim)
model.to(DEVICE)
sd = torch.load(weights_path)
# print(sd.keys())
clean_state_dict = {
    k.replace("_orig_mod.", ""): v for k, v in sd["model"].items()
}
model.load_state_dict(clean_state_dict)
# model = torch.compile(model)


val_transform = A.Compose([
    A.Resize(INPUT_SIZE, INPUT_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),  # Have to ImageNet normalize for AnyUp
                std=(0.229, 0.224, 0.225)),
    A.pytorch.ToTensorV2()
])

# Resizing to 512 before we do strided inference so that there is enough global context for the smaller inference window
# E.g. a 1024 size image with a 224 size window is going to miss key context
if STRIDED:
    RESIZE = 512
    val_transform_no_resize = A.Compose([
        A.Resize(RESIZE, RESIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406),  # Have to ImageNet normalize for AnyUp
                    std=(0.229, 0.224, 0.225)),
        A.pytorch.ToTensorV2()
    ])
else:
    val_transform_no_resize = None

val_dataset_vaihingen = VaihingenDataset(split="val", transform=val_transform, transform_no_resize=val_transform_no_resize)
val_loader_vaihingen = DataLoader(val_dataset_vaihingen, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=False, drop_last=True)

val_dataset_potsdam = PotsdamDataset(split="val", transform=val_transform, transform_no_resize=val_transform_no_resize)
val_loader_potsdam = DataLoader(val_dataset_potsdam, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=False, drop_last=True)

val_dataset_loveda = LoveDADataset(split="val", transform=val_transform, transform_no_resize=val_transform_no_resize, include_bg=True)
val_loader_loveda = DataLoader(val_dataset_loveda, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=False, drop_last=True)

val_dataset_oem = OpenEarthMapDataset(split_file="val_noxd.txt", transform=val_transform, transform_no_resize=val_transform_no_resize, include_bg=True)
val_loader_oem = DataLoader(val_dataset_oem, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=False, drop_last=True)

torch.manual_seed(42)

log_dir = "/home/rfaulken/dinov3/output_val"
writer, version, new_log_dir = logger(log_dir)
print("Version:", version)

terminal_console = Console()  # Terminal output
file_name = f"train.log"
file_console = Console(
    file=open(file_name, "w"),
)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_print(f"\n[bold blue]{'='*50}[/bold blue]")
log_print(f"[bold blue]Starting at {timestamp}[/bold blue]")
log_print(f"[bold green]Configuration:[/bold green]")

val_suite(model)

writer.flush()
file_console.file.close()
