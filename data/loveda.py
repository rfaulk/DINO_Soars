import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import random

imagenet_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean values
                         std=[0.229, 0.224, 0.225])  # ImageNet std values
])
imagenet_transform_DINO = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
totensor_transform = transforms.ToTensor()

class LoveDADataset(Dataset):
    def __init__(self, root="/home/rfaulken/datasets/loveDA/preprocessed", split="train", transform=None, transform_no_resize=None, include_bg=False):
        """
        Args:
            root (str): Root directory of COCOStuff dataset.
            split (str): 'train' or 'val'.
            transform (albumentations.Compose, optional): Transformations applied to image+mask.
        """
        self.root = root
        self.split = split
        self.include_bg = include_bg

        self.image_dir = os.path.join(root, split, "images")
        self.mask_dir = os.path.join(root, split, "labels")

        # print("WARNING: truncated loveDA")
        self.images = sorted(os.listdir(self.image_dir))
        self.masks = sorted(os.listdir(self.mask_dir))

        # indices = list(range(len(os.listdir(self.image_dir))))
        # random.seed(42)
        # random.shuffle(indices)
        # indices = indices[:500]

        # self.images = [sorted(os.listdir(self.image_dir))[i] for i in indices]
        # self.masks = [sorted(os.listdir(self.mask_dir))[i] for i in indices]

        assert len(self.images) == len(self.masks), "Number of images and masks must match."
        self.transform = transform
        self.transform_no_resize = transform_no_resize

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        img = np.array(Image.open(img_path).convert("RGB"))
        label = np.array(Image.open(mask_path))
        # print(np.unique(label))

        if not self.include_bg:
            label[label == 0] = 255
            label = label - 1
            label[label == 254] = 255
        # print(np.unique(label))

        if self.transform:
            aug = self.transform(image=img, mask=label)

        if self.transform_no_resize:
            full_res = self.transform_no_resize(image=img, mask=label)
            return {"img": aug["image"], "mask": full_res["mask"], "full_res_img": full_res["image"]}
        return {"img": aug["image"], "mask": aug["mask"], "full_res_img": imagenet_transform(img)}
