import torch
from PIL import Image
import pandas as pd
import os
import json
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CustomImageFolder(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.images_list = os.listdir(root)
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        name = self.images_list[idx]
        image = Image.open(os.path.join(self.root, name))
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        return { 'name': name, 'image': image }

class AVAFeatDataset(Dataset):
    def __init__(self, csv_file, feat_file):
        self.annotations = pd.read_csv(csv_file)
        with open(feat_file, 'r') as f:
            self.feats_dict = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = str(self.annotations.iloc[idx, 0]) + '.jpg'
        feat = torch.as_tensor(self.feats_dict[img_name])
        annotations = self.annotations.iloc[idx, 1: 11].to_numpy()
        annotations = annotations.astype('float').reshape(-1, 1)
        tri_cls = int(self.annotations.iloc[idx, 11])
        bin_cls = int(self.annotations.iloc[idx, 12])
        sample = {'img_id': img_name, 'image': feat, 'annotations': annotations, 'tri_cls': tri_cls, 'bin_cls': bin_cls} # image: feat is for adapting trainer

        return sample

class AVADataset(Dataset):
    """AVA dataset

    Args:
        # csv_file: a 11-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings
        csv_file: a 12-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings, column 12 contains the bin_cls infomation
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.annotations.iloc[idx, 0]) + '.jpg')
        image = Image.open(img_name).convert('RGB')
        annotations = self.annotations.iloc[idx, 1: 11].to_numpy()
        annotations = annotations.astype('float').reshape(-1, 1)
        tri_cls = int(self.annotations.iloc[idx, 11])
        bin_cls = int(self.annotations.iloc[idx, 12])
        sample = {'img_id': img_name, 'image': image, 'annotations': annotations, 'tri_cls': tri_cls, 'bin_cls': bin_cls}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample
