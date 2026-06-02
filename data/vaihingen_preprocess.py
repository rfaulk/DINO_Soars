#!/usr/bin/env python3
import os
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import tifffile as tif
import numpy as np

# === CONFIG ===
IMG_DIR = "/home/rfaulken/datasets/vaihingen/top"
LAB_DIR = "/home/rfaulken/datasets/vaihingen/labels"
OUTPUT_DIR = "/home/rfaulken/datasets/vaihingen/preprocessed_nobg"
TILE_SIZE = 1000
OVERLAP = 0   # Optional manual overlap (in addition to auto edge overlap)

VAL_SET = ["area2.", "area2_", "area4", "area6", "area8", "area10", "area12", "area14",
           "area16", "area20", "area22", "area24", "area27", "area29", "area31", "area33",
           "area35", "area38"]

MAP = {
    (255, 255, 255): 0,
    (0, 0, 255): 1,
    (0, 255, 255): 2,
    (0, 255, 0): 3,
    (255, 255, 0): 4,
    (255, 0, 0): 5,
    (0, 0, 0): 255
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_offsets(length, tile_size, overlap):
    """
    Compute starting offsets for tiles along one dimension,
    ensuring full coverage and overlap at the end if needed.
    """
    step = tile_size - overlap
    offsets = list(range(0, max(1, length - tile_size + 1), step))

    # If the last tile doesn't end exactly at the image boundary, add one more tile
    if offsets[-1] + tile_size < length:
        offsets.append(length - tile_size)

    return offsets

def remap_rgb_to_label(rgb_patch, mapping):
    """Convert a 3-channel RGB patch to single-channel label map."""
    h, w, _ = rgb_patch.shape
    label_map = np.zeros((h, w), dtype=np.uint8)
    rgb_flat = rgb_patch.reshape(-1, 3)
    label_flat = label_map.reshape(-1)
    for rgb, label in mapping.items():
        mask = np.all(rgb_flat == rgb, axis=1)
        label_flat[mask] = label
    # print("WARNING removing bg label")
    label_map[label_map == 5] = 255
    return label_map


def split_tiff(input_path, output_dir, tile_size=TILE_SIZE, overlap=OVERLAP, is_label=False):
    fname = os.path.splitext(os.path.basename(input_path))[0]
    with rasterio.open(input_path) as src:
        w, h = src.width, src.height
        profile = src.profile.copy()

        profile.update(count=1, dtype=rasterio.uint8)  # single-channel output

        x_offsets = compute_offsets(w, tile_size, overlap)
        y_offsets = compute_offsets(h, tile_size, overlap)

        for i, y_off in enumerate(y_offsets):
            for j, x_off in enumerate(x_offsets):
                window = Window(x_off, y_off, tile_size, tile_size)
                transform = src.window_transform(window)

                rgb = np.transpose(src.read([1, 2, 3], window=window), (1, 2, 0))
                label_map = remap_rgb_to_label(rgb, MAP)

                profile.update({
                    "width": label_map.shape[1],
                    "height": label_map.shape[0],
                    "transform": transform
                })

                out_name = f"{fname}_y{i:02d}_x{j:02d}.tif".replace("_noBoundary.", ".")
                out_path = os.path.join(output_dir, out_name)

                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(label_map, 1)


def process(dataset, split, tiftype, is_label):
    in_dir = IMG_DIR if tiftype == "images" else LAB_DIR
    outdir = os.path.join(OUTPUT_DIR, split, tiftype)
    os.makedirs(outdir, exist_ok=True)
    for tif in tqdm(dataset, desc=f"{split}/{tiftype}"):
        in_path = os.path.join(in_dir, tif)
        split_tiff(in_path, outdir, is_label=is_label)


def main():
    all_imgs = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(".tif")]
    train_imgs = sorted([img for img in all_imgs if not any(substring in img for substring in VAL_SET)])
    val_imgs = sorted([img for img in all_imgs if any(substring in img for substring in VAL_SET)])

    all_labels = [f for f in os.listdir(LAB_DIR) if f.lower().endswith(".tif")]
    train_labs = sorted([lab for lab in all_labels if not any(substring in lab for substring in VAL_SET)])
    val_labs = sorted([lab for lab in all_labels if any(substring in lab for substring in VAL_SET)])

    print(all_imgs)
    print("")
    print(all_labels)
    assert len(all_labels) == len(all_imgs), (len(all_labels), len(all_imgs))
    # Process both images and labels
    process(train_imgs, "train", "images", is_label=False)
    process(val_imgs, "val", "images", is_label=False)
    process(train_labs, "train", "labels", is_label=True)
    process(val_labs, "val", "labels", is_label=True)


if __name__ == "__main__":
    main()
