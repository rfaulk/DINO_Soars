"""For building the RS subset of COCO-Stuff"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil

class_subset = [1, 2, 3, 4, 5, 6, 7, 8,
                94, 95, 96,
                110, 112,
                123, 124, 125, 126, 127, 128,
                131,
                134, 135, 139,
                141, 143, 144, 145, 146, 147, 148, 149, 150,
                153, 154,
                157, 158,
                161, 163,
                168,
                177,
                181
]

def convert_coco_labels(src_dir, dst_dir):
    """
    Convert COCO-Stuff labels so that only the given subset of classes are kept.
    Others are mapped to ignore label 255.

    selected_classes: list of original coco class ids (AFTER any coco-stuff offsets)
                      example: [0, 5, 12] means only keep those classes.
    """

    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(dst_dir.replace("labels_subset", "images_subset"), exist_ok=True)
    mask_files = [f for f in os.listdir(src_dir) if f.endswith('.png')]

    # Build mapping: original_label -> new_label (0..K-1)
    # Everything else maps to 255 (ignore)
    remap = np.full(256, 255, dtype=np.uint8)
    for new_id, orig_id in enumerate(class_subset):
        remap[orig_id] = new_id

    for fname in tqdm(mask_files, desc=f"Converting {os.path.basename(src_dir)}"):
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)

        mask = np.array(Image.open(src_path), dtype=np.uint8)

        # Remap labels
        new_mask = remap[mask]

        if np.all(new_mask == 255):
            continue

        Image.fromarray(new_mask).save(dst_path)

        # Also do images
        src_path = src_path.replace("labels_unfixed", "images").replace(".png", ".jpg")
        dst_path = dst_path.replace("labels_subset", "images_subset").replace(".png", ".jpg")
        shutil.copy2(src_path, dst_path)


if __name__ == "__main__":

    splits = "val2017", "train2017"
    src_root = "/home/rfaulken/datasets/cocostuff/labels_unfixed"
    dst_root = "/home/rfaulken/datasets/cocostuff/labels_subset"
    for split in splits:
        src_dir = os.path.join(src_root, split)
        dst_dir = os.path.join(dst_root, split)
        convert_coco_labels(src_dir, dst_dir)

class_names = [
  # "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  # "traffic light",
  # "fire hydrant",
  "street sign",
  "stop sign",
  "parking meter",
  "bench",
  # "bird",
  # "cat",
  # "dog",
  # "horse",
  # "sheep",
  # "cow",
  # "elephant",
  # "bear",
  # "zebra",
  # "giraffe",
  # "hat",
  # "backpack",
  # "umbrella",
  # "shoe",
  # "eye glasses",
  # "handbag",
  # "tie",
  # "suitcase",
  # "frisbee",
  # "skis",
  # "snowboard",
  # "sports ball",
  # "kite",
  # "baseball bat",
  # "baseball glove",
  # "skateboard",
  # "surfboard",
  # "tennis racket",
  # "bottle",
  # "plate",
  # "wine glass",
  # "cup",
  # "fork",
  # "knife",
  # "spoon",
  # "bowl",
  # "banana",
  # "apple",
  # "sandwich",
  # "orange",
  # "broccoli",
  # "carrot",
  # "hot dog",
  # "pizza",
  # "donut",
  # "cake",
  # "chair",
  # "couch",
  # "potted plant",
  # "bed",
  # "mirror",
  # "dining table",
  # "window",
  # "desk",
  # "toilet",
  # "door",
  # "tv",
  # "laptop",
  # "mouse",
  # "remote",
  # "keyboard",
  # "cell phone",
  # "microwave",
  # "oven",
  # "toaster",
  # "sink",
  # "refrigerator",
  # "blender",
  # "book",
  # "clock",
  # "vase",
  # "scissors",
  # "teddy bear",
  # "hair drier",
  # "toothbrush",
  # "hair brush",
  "banner",
  # "blanket",
  "branch",
  "bridge",
  "building", #"building-other",
  "bush",
  # "cabinet",
  # "cage",
  # "cardboard",
  # "carpet",
  # "ceiling", #"ceiling-other",
  # "ceiling tile",
  # "cloth",
  # "clothes",
  # "clouds",
  # "counter",
  # "cupboard",
  # "curtain",
  # "desk", #"desk-stuff",
  "dirt",
  # "door",  # "door-stuff",
  "fence",
  # "marble floor", # "floor-marble"
  # "floor", #"floor-other",
  # "stone floor", #"floor-stone",
  # "tile floor", #"floor-tile",
  # "wood floor", #"floor-wood",
  "flower",  # More subs follow, but I don't really see the point in noting them all
  # "fog",
  # "food",
  # "fruit",
  # "furniture",
  "grass",
  "gravel",
  "ground",
  "hill",
  "house",
  "leaves",
  # "light",
  # "mat",
  "metal",
  # "mirror",
  # "moss",
  "mountain",
  "mud",
  # "napkin",
  "net",
  # "paper",
  "pavement",
  # "pillow",
  "plant",
  # "plastic",
  "platform",
  "playing field",
  "railing",
  "railroad",
  "river",
  "road",
  "rock",
  "roof",
  # "rug",
  # "salad",
  "sand",
  "sea",
  # "shelf",
  # "sky",
  "skyscraper",
  "snow",
  # "solid",
  # "stairs",
  "stone",
  # "straw",
  "structural",
  # "table",
  # "tent",
  # "textile",
  # "towel",
  "tree",
  # "vegetable",
  # "brick wall",
  # "concrete wall",
  # "other wall",
  # "panel wall",
  # "stone wall",
  # "tile wall",
  # "wood wall",
  "water",
  # "waterdrops",
  # "window blind",
  # "window",
  "wood"
]
