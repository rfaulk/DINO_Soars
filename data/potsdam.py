import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
import tifffile as tif
from torchvision import transforms

ROOT = '/home/rfaulken/datasets/potsdam/preprocessed_RGB'

imagenet_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean values
                         std=[0.229, 0.224, 0.225])  # ImageNet std values
])
imagenet_transform_DINO = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
totensor_transform = transforms.ToTensor()

class PotsdamDataset(Dataset):
    def __init__(self, root_dir=ROOT, split="train", transform=None, transform_no_resize=None, include_bg=True):
        """
        Args:
            root_dir (str or Path): Root directory of potsdam dataset.
            split (str): 'train' or 'val'.
            transform (callable, optional): Transformations on both image and label.
            normalize (bool): Whether to apply ImageNet normalization.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_dir = self.root_dir / split / "images" 
        if include_bg:
            self.lbl_dir = self.root_dir / split / "labels"
        else:
            self.lbl_dir = self.root_dir / split / "labels_nobg"
        self.transform = transform
        self.transform_no_resize = transform_no_resize

        self.images = sorted(self.img_dir.glob("*.tif"))
        self.labels = sorted(self.lbl_dir.glob("*.tif"))
        assert len(self.images) == len(self.labels), (len(self.images), len(self.labels))
        assert len(self.images) > 0

        # ISPRS potsdam approximate ImageNet normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        lbl_path = self.labels[idx]

        img = tif.imread(img_path)
        label = Image.open(lbl_path)

        img = np.array(img)
        label = np.array(label, dtype=np.long)

        # print("@", lbl_path, label.shape, np.unique(label))

        if self.transform:
            aug = self.transform(image=img, mask=label)

        # Dickered code to get a full-res image out if we need it (sometimes we want to process a 224x224 img and upsample to orig res e.g. 512x512 with AnyUp)
        if self.transform_no_resize:
            full_res = self.transform_no_resize(image=img, mask=label)
            return {"img": aug["image"], "mask": full_res["mask"], "full_res_img": full_res["image"]}
        return {"img": aug["image"], "mask": aug["mask"], "full_res_img": imagenet_transform(img)}
