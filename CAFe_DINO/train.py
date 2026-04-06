import sys
sys.path.append('/home/rfaulken/dinov3/CVPR')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
from torch.profiler import profile, record_function, ProfilerActivity
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import albumentations as A

from rich import print
from rich.console import Console

from tqdm import tqdm
from omegaconf import OmegaConf

from data.cocostuff import COCOStuffDatasetSubset
from data.vaihingen import VaihingenDataset
from data.potsdam import PotsdamDataset
from data.oem import OpenEarthMapDataset
from data.loveda import LoveDADataset

from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l

from CAFe_DINO.modeling.cafedino import CAFe_DINO
from anyup.anyup.model import AnyUp
from utils import log_print, logger, validate
from val_data import *
torch.set_float32_matmul_precision('high')

DEVICE = "cuda"

FREQ = 100
VALIDATE_FREQ = 3600 #100
INPUT_SIZE = 224

def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

def enable_peft(cfg, model):
    """Freeze everything but the aggregator"""
    for name, param in model.backbone.named_parameters():
        param.requires_grad = False

    if cfg.freeze != "blocks" and cfg.freeze != "both":
        for name, param in model.backbone.visual_model.backbone.blocks.named_parameters():
            if '22' in name:
                param.requires_grad = True
            elif '23' in name:
                param.requires_grad = True

    if cfg.freeze != "text" and cfg.freeze != "both":
        for param in model.backbone.text_model.parameters():
            param.requires_grad = True

    if cfg.upsample_thaw and cfg.upsample_thaw == True:
        for param in model.upsampler.parameters():
            param.requires_grad = True
    else:
        for param in model.upsampler.parameters():
            param.requires_grad = False
            
    return model

def val_suite(model, writer, tokenizer, batch_idx):
    val_transform = A.Compose([
        A.Resize(INPUT_SIZE, INPUT_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406),  # Have to ImageNet normalize for AnyUp
                    std=(0.229, 0.224, 0.225)),
        A.pytorch.ToTensorV2()
    ])
    batch_size = 1

    val_dataset_vaihingen = VaihingenDataset(split="val", transform=val_transform)
    val_loader_vaihingen = DataLoader(val_dataset_vaihingen, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=False, drop_last=True)

    val_dataset_potsdam = PotsdamDataset(split="val", transform=val_transform)
    val_loader_potsdam = DataLoader(val_dataset_potsdam, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=False, drop_last=True)

    val_dataset_isaid = ISAIDDataset(split="val", transform=val_transform)
    val_loader_isaid = DataLoader(val_dataset_isaid, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=False, drop_last=True)

    val_dataset_loveda = LoveDADataset(split="val", transform=val_transform)
    val_loader_loveda = DataLoader(val_dataset_loveda, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=False, drop_last=True)

    val_dataset_dlrsd = DLRSDDataset(split="val", transform=val_transform)
    val_loader_dlrsd = DataLoader(val_dataset_dlrsd, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=False, drop_last=True)

    val_dataset_oem = OpenEarthMapDataset(split_file="val_noxd.txt", transform=val_transform)
    val_loader_oem = DataLoader(val_dataset_oem, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=False, drop_last=True)

    # miou, confmat = validate(model, tokenizer, val_loader_oem, DEVICE, OEM_NUM_CLASSES, class_names=OEM_CLASS_NAMES, save_path="oem_confmat.png")
    # writer.add_scalar("val_oem", miou, batch_idx)

    miou, confmat = validate(model, tokenizer, val_loader_vaihingen, DEVICE, VAIHINGEN_NUM_CLASSES, class_names=VAIHINGEN_CLASS_NAMES, save_path="vaihingen_confmat.png", full_res=False)
    writer.add_scalar("val_vaihingen_miou", miou, batch_idx)

    miou, confmat = validate(model, tokenizer, val_loader_potsdam, DEVICE, VAIHINGEN_NUM_CLASSES, class_names=VAIHINGEN_CLASS_NAMES, save_path="potsdam_confmat.png", full_res=False)
    writer.add_scalar("val_potsdam_miou", miou, batch_idx)

    # miou, confmat = validate(model, tokenizer, val_loader_dlrsd, DEVICE, DLRSD_NUM_CLASSES, class_names=DLRSD_CLASS_NAMES, save_path="dlrsd_confmat.png")
    # writer.add_scalar("val_dlrsd", miou, batch_idx)

    # # miou, confmat = validate(model, tokenizer, val_loader_isaid, DEVICE, DLRSD_NUM_CLASSES, class_names=ISAID_CLASS_NAMES, save_path="isaid_confmat.png")
    # # writer.add_scalar("val_isaid", miou, batch_idx)

    # miou, confmat = validate(model, tokenizer, val_loader_loveda, DEVICE, LOVEDA_NUM_CLASSES, class_names=LOVE_DA_CLASS_NAMES, save_path="loveda_confmat.png")
    # writer.add_scalar("val_loveda_miou", miou, batch_idx)
    return writer

def main():
    # rank = setup()
    # device = torch.device(f"cuda:{rank}")
    device = DEVICE

    parser = argparse.ArgumentParser(description="Read a single string from the command line")
    parser.add_argument("--config", type=str, help="Input string", required=True)

    args = parser.parse_args()
    print(args.config)

    config_file = args.config
    cfg = OmegaConf.load("/home/rfaulken/dinov3/CAFe_DINO/configs/" + config_file + ".yaml")

    batch_size = cfg.batch_size
    backbone, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l()

    upsampler = AnyUp()
    upsampler.load_state_dict(torch.load("/home/rfaulken/dinov3/weights/anyup_paper.pth", map_location="cpu"))

    if 'linear' in cfg:
        use_linear_transformer = cfg.linear
    else:
        use_linear_transformer = False
    model = CAFe_DINO(backbone, tokenizer, upsampler, device=device, input_resolution=(INPUT_SIZE//16, INPUT_SIZE//16), aggregator_dim=cfg.aggregator_dim, use_linear_transformer=use_linear_transformer)
    model = enable_peft(cfg, model)
    model.to(device)

    model = torch.compile(model)
    # model = DDP(model, device_ids=[rank])

    criterion = torch.nn.CrossEntropyLoss(ignore_index=255, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler("cuda")

    train_transform = A.Compose([
        # A.SquareSymmetry(p=1.0),  # apply one of {rot90, rot180, rot270, flips}
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.RandomGamma(p=0.3),
        A.CoarseDropout(p=0.3),
        A.Resize(INPUT_SIZE, INPUT_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406),  # Have to ImageNet normalize for AnyUp
                    std=(0.229, 0.224, 0.225)),
        A.pytorch.ToTensorV2()
    ])

    train_dataset = COCOStuffDatasetSubset(split="train", transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=False, drop_last=True)

    # train_dataset = DLRSDDataset(split="train", transform=train_transform)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=False, drop_last=True)

    accum_steps = cfg.accum_steps  # number of microbatches per update
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.lr,                   # peak LR
        epochs=cfg.epochs,             # total number of epochs
        steps_per_epoch=len(train_loader),  # num batches
        pct_start=0.05,                 # 30% of cycle increasing LR
        anneal_strategy='cos',         # cosine decay (smooth)
        div_factor=25.0,               # initial_lr = max_lr / div_factor
        final_div_factor=1e4,          # final_lr = initial_lr / final_div_factor
    )

    torch.manual_seed(42)

    log_dir = "/home/rfaulken/dinov3/output/" + config_file + "/"
    writer, version, new_log_dir = logger(log_dir)
    print("Version:", version)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_print(f"\n[bold blue]{'='*50}[/bold blue]")
    log_print(f"[bold blue]Starting at {timestamp}[/bold blue]")
    log_print(f"[bold green]Configuration:[/bold green]")

    print(len(train_loader))

    
    batch_idx = 0
    true_batch_idx = 0
    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for image_batch, label_batch in tqdm(train_loader):
            batch_idx += 1

            image_batch = image_batch.to(device, non_blocking=True)
            label_batch = label_batch.to(device, non_blocking=True, dtype=torch.long)

            with torch.amp.autocast("cuda"):
                logits = model(image_batch, cfg.class_names)
                loss = criterion(logits, label_batch)
                loss = loss / accum_steps  # normalize loss to average gradients

            scaler.scale(loss).backward()

            # only update weights every accum_steps iterations
            if batch_idx % accum_steps == 0:
                true_batch_idx += 1
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            # Optional: Update tqdm with loss information
            if batch_idx % (FREQ * accum_steps) == 0:
                # Log all loss components to tensorboard
                l = loss.item()
                if l != 0:
                    writer.add_scalar(
                        f"Loss/",
                        l,
                        true_batch_idx,
                    )

                # Log learning rates
                writer.add_scalar(
                    "Learning Rate",
                    optimizer.param_groups[0]["lr"],
                    true_batch_idx,
                )

                # Build concise log message
                loss_str = f"loss: {l:.4f}"
                log_print(
                    f"Step={true_batch_idx}/{cfg.epochs * len(train_loader)} | "
                    f"Epoch={epoch}/{cfg.epochs} | "
                    # f"Progress: {overall_progress:.1f}% | "
                    f"{loss_str}"
                )
            if batch_idx % (VALIDATE_FREQ * accum_steps) == 0:
                model.eval()
                # writer = val_suite(model, writer, tokenizer, true_batch_idx)
                model.train()
                # Save checkpoint at every epoch
                checkpoint_path = os.path.join(new_log_dir, f"model_{batch_idx}iter.pth")

                save_dict = {
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "model": model.state_dict(),
                }

                torch.save(save_dict, checkpoint_path)
                log_print(f"Saved checkpoint: {checkpoint_path}")

        # handle leftover gradients if len(train_loader) not multiple of accum_steps
        if batch_idx % accum_steps != 0:
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        torch.cuda.empty_cache()

    writer.flush()
    # cleanup()

if __name__ == "__main__":
    main()