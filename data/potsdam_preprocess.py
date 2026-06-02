#!/usr/bin/env python3
import os
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from tqdm import tqdm
import numpy as np

# === CONFIG ===
IMG_DIR = "/home/rfaulken/datasets/potsdam/2_Ortho_RGB"   # folder containing large .tif files
LAB_DIR = "/home/rfaulken/datasets/potsdam/labels"
OUTPUT_DIR = "/home/rfaulken/datasets/potsdam/preprocessed_RGB"   # where to save smaller tiles
TILE_SIZE = 1000                           # pixels per side
OVERLAP = 0                                # optional, can set >0 for overlap

VAL_SET = ["2_13", "2_14", "3_13", "3_14", "4_13", "4_14", "4_15", "5_13", "5_14", "5_15", "6_13", "6_14", "6_15", "7_13"]

MAP = {
    (255, 255, 255): 0,
    (0, 0, 255): 1,
    (0, 255, 255): 2,
    (0, 255, 0): 3,
    (255, 255, 0): 4,
    (255, 0, 0): 5,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def remap_rgb_to_label(rgb_patch, mapping):
    """Convert a 3-channel RGB patch to single-channel label map."""
    h, w, _ = rgb_patch.shape
    label_map = np.zeros((h, w), dtype=np.uint8)
    rgb_flat = rgb_patch.reshape(-1, 3)
    label_flat = label_map.reshape(-1)
    for rgb, label in mapping.items():
        mask = np.all(rgb_flat == rgb, axis=1)
        label_flat[mask] = label
    # print(label_map.shape)
    return label_map

import os
import cv2
import numpy as np

def split_tiff(input_path, output_dir, tiftype, tile_size=TILE_SIZE, overlap=OVERLAP):
    fname = os.path.splitext(os.path.basename(input_path))[0]

    # Read full image (unchanged preserves bit depth and channels)
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Could not read {input_path}")

    # OpenCV loads as H x W x C (BGR for color)
    if img.ndim == 2:
        h, w = img.shape
        channels = 1
    else:
        h, w, channels = img.shape

    step = tile_size - overlap
    x_tiles = (w + step - 1) // step
    y_tiles = (h + step - 1) // step

    for i in range(y_tiles):
        for j in range(x_tiles):
            x_off = j * step
            y_off = i * step

            win_width = min(tile_size, w - x_off)
            win_height = min(tile_size, h - y_off)

            tile = img[y_off:y_off + win_height,
                       x_off:x_off + win_width]

            if tiftype == "labels":
                # Convert BGR -> RGB before remapping
                if tile.ndim == 3 and tile.shape[2] == 3:
                    rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                else:
                    raise ValueError("Label TIFF must have 3 channels")

                label_map = remap_rgb_to_label(rgb, MAP).astype(np.uint8)

                out_tile = label_map  # single channel
            elif tiftype == "images":
                out_tile = tile
            else:
                raise ValueError("tiftype must be 'labels' or 'images'")

            out_name = f"{fname}_y{i:02d}_x{j:02d}.tif"
            out_path = os.path.join(output_dir, out_name)

            cv2.imwrite(out_path, out_tile)

# def split_tiff(input_path, output_dir, tiftype, tile_size=TILE_SIZE, overlap=OVERLAP):
#     fname = os.path.splitext(os.path.basename(input_path))[0]

#     with rasterio.open(input_path) as src:
#         w, h = src.width, src.height
#         profile = src.profile.copy()

#         # Change output to single channel, uint8
#         if tiftype == "labels":
#             profile.update(count=1, dtype=rasterio.uint8)

#         x_tiles = (w + tile_size - 1) // tile_size
#         y_tiles = (h + tile_size - 1) // tile_size

#         for i in range(y_tiles):
#             for j in range(x_tiles):
#                 x_off = j * (tile_size - overlap)
#                 y_off = i * (tile_size - overlap)
#                 win_width = min(tile_size, w - x_off)
#                 win_height = min(tile_size, h - y_off)
#                 window = Window(x_off, y_off, win_width, win_height)
#                 transform = src.window_transform(window)

#                 # Read 3-band RGB and convert to label map
#                 if tiftype == "labels":
#                     rgb = np.transpose(src.read([1, 2, 3], window=window), (1, 2, 0))
#                     label_map = remap_rgb_to_label(rgb, MAP)

#                 profile.update({
#                     "width": win_width,
#                     "height": win_height,
#                     "transform": transform
#                 })

#                 out_name = f"{fname}_y{i:02d}_x{j:02d}.tif"
#                 out_path = os.path.join(output_dir, out_name)

#                 with rasterio.open(out_path, "w", **profile) as dst:
#                     if tiftype == "labels":
#                         dst.write(label_map, 1)
#                     elif tiftype == "images":
#                         dst.write(src.read(window=window))

def process(dataset, split, tiftype):
    if tiftype == "images":
        in_dir = IMG_DIR
    elif tiftype == "labels":
        in_dir = LAB_DIR
    outdir = os.path.join(OUTPUT_DIR, split, tiftype)
    os.makedirs(outdir, exist_ok=True)
    for tif in tqdm(dataset):
        in_path = os.path.join(in_dir, tif)
        split_tiff(in_path, outdir, tiftype)


def main():
    all_imgs = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(".tif")]
    train_imgs = sorted([img for img in all_imgs if not any(substring in img for substring in VAL_SET)])
    val_imgs = sorted([img for img in all_imgs if any(substring in img for substring in VAL_SET)])

    all_labels = [f for f in os.listdir(LAB_DIR) if f.lower().endswith(".tif")]
    train_labs = sorted([lab for lab in all_labels if not any(substring in lab for substring in VAL_SET)])
    val_labs = sorted([lab for lab in all_labels if any(substring in lab for substring in VAL_SET)])

    
    # Train data
    process(train_imgs, "train", "images")
    process(val_imgs, "val", "images")
    process(train_labs, "train", "labels")
    process(val_labs, "val", "labels")


if __name__ == "__main__":
    main()
