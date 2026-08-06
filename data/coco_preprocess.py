import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def convert_coco_labels(src_dir, dst_dir):
    """
    Converts COCO-Stuff label masks:
      - 0 becomes 255 (ignore)
      - everything else is decremented by 1

    Args:
        src_dir (str): Path to the directory with original label .png files.
        dst_dir (str): Path to the directory to save converted labels.
    """
    os.makedirs(dst_dir, exist_ok=True)
    mask_files = [f for f in os.listdir(src_dir) if f.endswith('.png')]

    for fname in tqdm(mask_files, desc=f"Converting {os.path.basename(src_dir)}"):
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)

        mask = np.array(Image.open(src_path), dtype=np.uint8)

        # Apply remapping
        new_mask = np.where((mask == 0) | (mask == 255), 255, mask - 1).astype(np.uint8)

        Image.fromarray(new_mask).save(dst_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Remap COCO-Stuff semantic labels")
    parser.add_argument("--src_root", type=str, required=True,
                        help="Root directory containing original COCO-Stuff annotations")
    parser.add_argument("--dst_root", type=str, required=True,
                        help="Output directory for remapped annotations")

    parser.add_argument("--splits", nargs="+", default=["train2017", "val2017"],
                        help="Which splits to process (default: train2017 val2017)")

    args = parser.parse_args()

    print(args.splits)
    for split in args.splits:
        src_dir = os.path.join(args.src_root, split)
        dst_dir = os.path.join(args.dst_root, split)
        convert_coco_labels(src_dir, dst_dir)
