import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch
import tifffile as tif
from torchvision import transforms
import numpy as np

imagenet_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean values
                         std=[0.229, 0.224, 0.225])  # ImageNet std values
])
imagenet_transform_DINO = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
totensor_transform = transforms.ToTensor()

class OpenEarthMapDataset(Dataset):
    def __init__(self, root='/home/rfaulken/datasets/openearthmap/OpenEarthMap_wo_xBD', split_file="val.txt", transform=None, transform_no_resize=None, include_bg=False):
        self.root = root
        self.include_bg = include_bg

        # Load split list (e.g. Kigali/000123.png)
        self.split_file = split_file
        if split_file is None:
            
            self.files = sorted([f for f in os.listdir(os.path.join(root, "images")) if os.path.isfile(os.path.join(root, f))])
        else:
            split_path = os.path.join(root, split_file)
            with open(split_path, "r") as f:
                self.files = [line.strip() for line in f.readlines() if line.strip()]

            assert len(self.files) > 0, f"No entries found in {split_file}"

        self.transform = transform or T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        self.transform_no_resize = transform_no_resize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]


        # city = ""
        # for i in fname.split("_")[:-1]:
        #     city += i + "_"
        # city = city[:-1]
        city, fname  = fname.split('/')

        img_path = os.path.join(self.root, city, "images", fname)
        mask_path = os.path.join(self.root, city, "labels", fname)

        # img_path = os.path.join(self.root, "images", fname)
        # mask_path = os.path.join(self.root, "labels", fname)

        img = tif.imread(img_path)
        label = tif.imread(mask_path)  # uint8 label map
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
        return {"img": aug["image"], "mask": totensor_transform(label), "full_res_img": imagenet_transform(img)}
    
class OpenEarthMapDatasetDINO(OpenEarthMapDataset):
    def __init__(self, root='/home/rfaulken/datasets/openearthmap/OpenEarthMap_wo_xBD', split_file="val.txt", transform=None, target_transform=None, include_bg=False):
        super().__init__(root=root, split_file=split_file, transform=transform, include_bg=include_bg)

    def __getitem__(self, idx):
        fname = self.files[idx]
        # print("@", fname)

        if self.split_file is None:
            img_path = os.path.join(self.root, "images", fname)
            mask_path = os.path.join(self.root, "labels", fname)

        else:
            city, fname  = fname.split('/')

            img_path = os.path.join(self.root, city, "images", fname)
            mask_path = os.path.join(self.root, city, "labels", fname)

        # img_path = os.path.join(self.root, "images", fname)
        # mask_path = os.path.join(self.root, "labels", fname)

        img = tif.imread(img_path)
        # img[img == 0] = 255
        # img = img - 1
        # img[img == 254] = 255
        label = tif.imread(mask_path)  # uint8 label map
        if not self.include_bg:
            label[label == 0] = 255
            label = label - 1
            label[label == 254] = 255

        if self.transform:
            img = self.transform(img)

        return {"img": img, "mask": label, "full_res_img": imagenet_transform_DINO(img)}
