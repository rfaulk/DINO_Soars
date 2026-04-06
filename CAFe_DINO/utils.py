import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console
import torchvision.utils as vutils

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

import numpy as np
import seaborn as sns
from tqdm import tqdm


DEVICE = torch.device("cuda")

PROMPT_TEMPLATES = (
    "a photo of {}", "an image of {}", "a photograph of {}", "a picture of {}",
    "a photo of a {}", "an image of a {}", "a photo of the {}", "an image of the {}",
    "a close-up photo of {}", "a cropped image featuring {}",
)

terminal_console = Console()  # Terminal output
file_name = f"train.log"
file_console = Console(
    file=open(file_name, "w"),
)

def logger(base_log_dir):
    os.makedirs(base_log_dir, exist_ok=True)
    existing_versions = [
        int(d.split("_")[-1])
        for d in os.listdir(base_log_dir)
        if os.path.isdir(os.path.join(base_log_dir, d)) and d.startswith("version_")
    ]
    new_version = max(existing_versions, default=-1) + 1
    new_log_dir = os.path.join(base_log_dir, f"version_{new_version}")

    # Create the SummaryWriter with the new log directory
    writer = SummaryWriter(log_dir=new_log_dir)
    return writer, new_version, new_log_dir

def text_embed(backbone, tokenizer, class_names) -> torch.Tensor:
    # Build prompts
    prompts = [tpl.format(name) for name in class_names for tpl in PROMPT_TEMPLATES]

    toks = tokenizer.tokenize(prompts).to(DEVICE)

    # Encode and keep patch part
    embs = backbone.encode_text(toks)[:, 1024:]  # [C*K, D]

    C = len(class_names)
    K = len(PROMPT_TEMPLATES)
    D = embs.size(1)

    # Group prompts by class and average
    embs = embs.view(C, K, D).mean(dim=1)       # [C, D]

    return F.normalize(embs, p=2, dim=1)


def build_text_embeddings(backbone, tokenizer, class_names_list) -> torch.Tensor:
    # Compute once per list item, stack, mean over lists
    stack = torch.stack([
        text_embed(backbone, tokenizer, class_names)
        for class_names in class_names_list
    ])  # [L, C, D]

    return stack.mean(dim=0)  # [C, D]

def log_print(*args, **kwargs):
    """Log to both terminal and file with immediate flushing"""
    # Print to terminal
    terminal_console.print(*args, **kwargs)
    # Print to file and flush immediately
    file_console.print(*args, **kwargs)
    file_console.file.flush()  # Force immediate write to disk

def confusion_matrix(preds, labels, num_classes):
    mask = (labels >= 0) & (labels < num_classes)
    hist = torch.bincount(
        num_classes * labels[mask].to(torch.int64) + preds[mask],
        minlength=num_classes ** 2
    )
    return hist.reshape(num_classes, num_classes)

def overlay_segmentation(image, labelmap, class_names, num_classes, alpha=0.55, save_path="segmentation.png"):

    # print(image.shape, np.unique(image))

    cmap = plt.get_cmap('tab10')
    color_map = cmap(np.arange(0, num_classes*2, 2))[:, :3]  # (num_classes, 3)

    blended = image / 255
    color_overlay = np.zeros_like(blended)
    
    valid = labelmap != 255
    color_overlay[valid] = color_map[labelmap[valid]]

    blended[valid] = (1 - alpha) * blended[valid] + alpha * color_overlay[valid]

    plt.figure(figsize=(10, 10))
    plt.imshow(blended)
    plt.axis('off')

    # # Add class labels
    # if class_names:
    #     for class_idx, name in enumerate(class_names):
    #         # Compute approximate label position (center of region)
    #         ys, xs = np.where(labelmap == class_idx)
    #         if len(xs) == 0:
    #             continue
    #         x_mean, y_mean = np.mean(xs), np.mean(ys)
    #         plt.text(
    #             x_mean, y_mean, name,
    #             color='white', fontsize=10, ha='center', va='center',
    #             bbox=dict(facecolor='black', alpha=0.5, pad=1)
    #         )

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=1000)
    print(f"Overlay saved to {save_path}")
    return blended

def plot(num_classes, class_names, img, cost_vol_hr: torch.Tensor, aggregated_cost_vol: torch.Tensor, filename="comparison.png"):
    # cost_vols: (C, H, W)
    fig, axes = plt.subplots(2, num_classes + 2, figsize=(15, 6))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    cmap = "tab20" if num_classes > 10 else "tab10"
    for col in range(num_classes):
        # Top row: raw
        ax = axes[0, col]
        ax.imshow(cost_vol_hr[col].cpu().numpy())
        ax.axis("off")
        ax.set_title(class_names[col], fontsize=12, pad=8)

        # Bottom row: processed
        ax = axes[1, col]
        ax.imshow(aggregated_cost_vol[col].cpu().numpy())
        ax.axis("off")

    ax = axes[0, num_classes]
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Image/Pred", fontsize=12, pad=8)

    ax = axes[1, num_classes]
    ax.imshow(torch.argmin(aggregated_cost_vol, dim=0).squeeze(0).cpu().numpy(), cmap=cmap, norm=mcolors.BoundaryNorm(range(num_classes), num_classes-1))
    ax.axis("off")

    ax = axes[0, num_classes + 1]
    ax.imshow(torch.argmax(cost_vol_hr, dim=0).squeeze(0).cpu().numpy(), cmap=cmap, norm=mcolors.BoundaryNorm(range(num_classes), num_classes-1))
    ax.axis("off")

    # === Add legend ===
    labels = [f"Class {i}" for i in range(num_classes)]  # or your actual class names

    # Create legend handles with colors from the cmap
    cmap = plt.get_cmap("tab20") if num_classes > 10 else plt.get_cmap("tab10")
    handles = [mpatches.Patch(color=cmap(i), label=c) for i, c in enumerate(class_names)]
    ax = axes[1, num_classes+1]
    ax.legend(
        handles=handles,
        loc="upper right",  # choose placement
        bbox_to_anchor=(1.2, 1),  # moves legend outside the subplot
        frameon=False
    )

    plt.savefig(filename, bbox_inches="tight", dpi=300)
    overlay_segmentation(img, torch.argmax(cost_vol_hr, dim=0).squeeze(0).cpu().numpy(), class_names, num_classes)

def strided_inf(model, img: torch.Tensor, text_features: torch.Tensor, side: int, stride: int, num_classes: int) -> torch.Tensor:
    # Iterate over overlapping windows, accumulate predictions at the image resolution
    _, _, H, W = img.shape
    # num_classes, _ = text_features.shape
    probs = torch.zeros([num_classes, H, W], device="cuda")
    counts = torch.zeros([H, W], device="cuda")
    h_grids = max(H - side + stride - 1, 0) // stride + 1  # 2
    w_grids = max(W - side + stride - 1, 0) // stride + 1  # 2
    for i in range(h_grids):
        for j in range(w_grids):
            y1 = i * stride  # 0, 192
            x1 = j * stride  # 0, 192
            y2 = min(y1 + side, H)
            x2 = min(x1 + side, W)
            y1 = max(y2 - side, 0)
            x1 = max(x2 - side, 0)

            # Compute cosine similarities for this window, same logic as predict_whole
            with torch.no_grad():
                img_window = img[:, :, y1:y2, x1:x2]  # [3, H_win, W_win]

                cost_vol_up = model(img_window, text_features, pre_text_emb=True)

            probs[:, y1:y2, x1:x2] += cost_vol_up.squeeze(0)  # [num_classes, h, w]
            counts[y1:y2, x1:x2] += 1
    probs /= counts

    # Return "probabilities" at the img resolution, they will be upsampled to the target resolution later
    return probs  # [num_classes, H, W]

def validate(model, tokenizer, val_loader, device, num_classes, class_names, ignore_index=[255], save_path="confusion_matrix.png", strided=False, full_res=True, thresh=None, bg_idx=None):
    """
    Validate model and save confusion matrix as a PNG.

    Args:
        model: The trained model to evaluate.
        val_loader: DataLoader for validation data.
        device: Torch device (cuda or cpu).
        num_classes: Number of classes.
        text_emb: Text embedding input to the model.
        class_names: Optional list of class names for labeling axes.
        save_path: Path to save the confusion matrix image.
        normalize: Whether to normalize confusion matrix by row.
    """
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64, device='cuda')

    with torch.no_grad():
        # Pre-compute a text embedding for the whole validation since it's not going to change
        text_emb = build_text_embeddings(model.backbone, tokenizer, class_names)

        for batch in tqdm(val_loader, desc="Validation"):
            images, labels = batch["img"], batch["mask"]
            labels = labels.to(device, non_blocking=True, dtype=torch.long)
            # print(images.shape, labels.shape)
            
            with torch.amp.autocast("cuda"):

                if full_res:
                    full_res_img = batch["full_res_img"].to(device, non_blocking=True)
                else:
                    full_res_img = None
                    
                if strided:
                    if full_res:
                        input_img = full_res_img
                    else:
                        input_img = images.to(device, non_blocking=True)
                    cost_vol_up = strided_inf(model, input_img, text_emb, 224, 112, num_classes)
                    softmax_dim = 0
                else:
                    images = images.to(device, non_blocking=True)
                    cost_vol_up = model(images, text_emb, upsample_img=full_res_img, pre_text_emb=True)
                    softmax_dim = 1

                sm = torch.softmax(cost_vol_up, dim=softmax_dim)
                maxvals, preds = torch.max(sm, dim=softmax_dim)
            # --- Ignore invalid labels ---

            if thresh is not None:
                assert bg_idx is not None
                preds = preds.masked_fill(maxvals < thresh, bg_idx)

            # Assuming batch size is always 1
            if len(labels.shape) == 3:
                labels = labels[0]
            if len(preds.shape) == 3:
                preds = preds[0]

            mask = ~torch.isin(labels, torch.tensor(ignore_index, device='cuda'))
            preds_masked = preds[mask]
            labels_masked = labels[mask]
            conf_matrix += confusion_matrix(preds_masked, labels_masked, num_classes)

    # Compute IoU per class
    intersection = torch.diag(conf_matrix)
    union = conf_matrix.sum(0) + conf_matrix.sum(1) - intersection
    iou = intersection.float() / union.float().clamp(min=1)
    miou = iou.mean().item()

    # Those compromises are full res upsampling (a lack thereof) and a lack of striding
    print(f"Validation mIoU: {miou:.4f} (don't rely on this if training, there are speed compromises)")

    # Convert to numpy for plotting
    conf_cpu = conf_matrix.cpu().numpy().astype(np.float64)

    # --- Plot confusion matrix ---
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(conf_cpu, annot=True, cmap='Blues', xticklabels=class_names[0], yticklabels=class_names[0], cbar=True, square=True, annot_kws={"size": 8})

    # Force all ticks to appear
    ax.set_xticks(np.arange(len(class_names[0])) + 0.5)
    ax.set_yticks(np.arange(len(class_names[0])) + 0.5)
    ax.set_xticklabels(class_names[0], rotation=45, ha='right')
    ax.set_yticklabels(class_names[0], rotation=0)

    plt.title(f'Confusion Matrix (mIoU={miou:.4f})')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Confusion matrix saved to {save_path}")

    return miou

####### DEBUG ######## DELETEME ######################################################
# For worst 10 images in dataset
import cv2
import heapq

def compute_image_iou_per_class(pred, gt, c):
    pred_c = pred == c
    gt_c = gt == c
    union = (pred_c | gt_c).sum()
    if union == 0:
        return 1.0  # Assumption: this function is for finding small ious
    inter = (pred_c & gt_c).sum()
    return inter.float() / union.float()

def compute_false_px_count_per_class(pred, gt, classnum):
    ignore_index = [i for i in range(256)]
    ignore_index.pop(classnum)
    mask = ~torch.isin(gt, torch.tensor(ignore_index, device=gt.device))
    pred = pred[mask]
    gt = gt[mask]
    return torch.count_nonzero(pred != gt)

def compute_false_px_count(pred, gt, ignore_index):
    mask = ~torch.isin(gt, torch.tensor(ignore_index, device=gt.device))
    pred = pred[mask]
    gt = gt[mask]
    return torch.count_nonzero(pred != gt)

def compute_image_iou(pred, gt, num_classes, ignore_index):
    mask = ~torch.isin(gt, torch.tensor(ignore_index, device=gt.device))
    pred = pred[mask]
    gt = gt[mask]

    ious = []
    for c in range(num_classes):
        pred_c = pred == c
        gt_c = gt == c
        union = (pred_c | gt_c).sum()
        if union == 0:
            continue
        inter = (pred_c & gt_c).sum()
        ious.append((inter.float() / union.float()).item())

    return float(np.mean(ious)) if len(ious) else 0.0

def scale_to_255(x):
    x = x - x.min()
    x_min, x_max = x.min(), x.max()
    if x_max == x_min:
        return np.zeros_like(x, dtype=np.uint8)  # avoid division by zero
    x_scaled = (x - x_min) / (x_max - x_min) * 255
    return x_scaled.astype(np.uint8)

def worst10(
    model, tokenizer, val_loader, device, num_classes,
    class_names=None, ignore_index=[255],
    save_path="confusion_matrix.png",
    worst_k=20,
    worst_dir="worst_miou",
    strided=False, full_res=True,
    build_text_emb=False, sup=None, bg_idx=None
):
    """Find the worst 10 mIOU images in the val set and save for failure analysis
        This suppresses 0.0 mIOU images"""
    os.makedirs(worst_dir, exist_ok=True)

    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64, device='cuda')

    # min-heap of (miou, index, image, pred)
    worst_heap = []
    img_counter = 0

    with torch.no_grad():
        text_emb = build_text_embeddings(model.backbone, tokenizer, class_names)

        for images, labels, full_res_img, idx in tqdm(val_loader, desc="Validation"):
            labels = labels.to(device, non_blocking=True, dtype=torch.long)[0]
            full_res_img = full_res_img.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                if strided:
                    if full_res:
                        cost_vol_up = strided_inf(model, full_res_img, text_emb, 224, 112, num_classes)
                    else:
                        images = images.to(device, non_blocking=True)
                        cost_vol_up = strided_inf(model, images, text_emb, 224, 112, num_classes)
                    sm = torch.softmax(cost_vol_up, dim=0)
                    vals, preds = torch.max(sm, dim=0)

                elif full_res:
                    cost_vol_up = model(images, text_emb, upsample_img=full_res_img)
                    sm = torch.softmax(cost_vol_up, dim=1)
                    vals, preds = torch.max(sm, dim=1)

                else:
                    images = images.to(device, non_blocking=True)
                    cost_vol_up = model(images, text_emb)
                    sm = torch.softmax(cost_vol_up, dim=1)
                    vals, preds = torch.max(sm, dim=1)[0]

            if sup is not None:
                preds = preds.masked_fill(vals < sup, bg_idx)

            mask = ~torch.isin(labels, torch.tensor(ignore_index, device='cuda'))
            conf_matrix += confusion_matrix(preds[mask], labels[mask], num_classes)

            # ---------- NEW: per-image mIoU ----------
            # img_miou = compute_image_iou(preds, labels, num_classes, ignore_index)
            # img_miou = -compute_false_px_count(preds, labels, ignore_index)
            classnum = 4
            img_miou = compute_image_iou_per_class(preds, labels, classnum)

            # Keep worst K
            item = (-img_miou, idx, labels.clone(), preds.clone())
            if img_miou > -0.01:
                continue
            if len(worst_heap) < worst_k:
                heapq.heappush(worst_heap, item)
            else:
                heapq.heappushpop(worst_heap, item)

    # ---------- SAVE WORST IMAGES ----------
    worst_heap = sorted(worst_heap, key=lambda x: x[0])  # lowest mIoU first

    for rank, (miou_val, idx, gt, pred) in enumerate(worst_heap):
        filepath = val_loader.dataset.get_path(idx)
        img_save = cv2.imread(filepath)
        img_save = cv2.resize(img_save, (512, 512))

        overlay = overlay_segmentation(img_save, pred.cpu().detach().numpy(), class_names, num_classes, alpha=0.55)
        cv2.imwrite(os.path.join(worst_dir, f"rank_{rank:02d}_miou_{miou_val:.3f}.png"), scale_to_255(overlay))

        gt[gt == 5] = 255
        overlay_gt = overlay_segmentation(img_save, gt.cpu().detach().numpy(), class_names, num_classes, alpha=0.55)
        cv2.imwrite(os.path.join(worst_dir, f"rank_{rank:02d}_miou_{miou_val:.3f}_gt.png"), scale_to_255(overlay_gt))
        # print(np.unique(overlay), np.unique(overlay_gt))

        cv2.imwrite(os.path.join(worst_dir, f"rank_{rank:02d}_miou_{miou_val:.3f}_img.png"), img_save)


    # ---------- GLOBAL mIoU ----------
    intersection = torch.diag(conf_matrix)
    union = conf_matrix.sum(0) + conf_matrix.sum(1) - intersection
    iou = intersection.float() / union.float().clamp(min=1)
    miou = iou.mean().item()

    return miou
