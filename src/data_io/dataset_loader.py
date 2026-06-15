# -*- coding: utf-8 -*-
"""
Train / validation data loaders.

Uses standard torchvision transforms (modern API). Augmentations mirror the
original repo's intent (RandomResizedCrop, ColorJitter, small rotation, hflip)
plus ImageNet normalisation for the pretrained MobileNetV3 backbone.
"""

import torchvision.transforms as T
from torch.utils.data import DataLoader

from src.data_io.dataset_folder import DatasetFolderFT, DatasetFolderVal

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _train_transform(input_size):
    return T.Compose([
        T.ToPILImage(),
        T.RandomResizedCrop(tuple(input_size), scale=(0.9, 1.1)),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.RandomRotation(10),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _val_transform(input_size):
    return T.Compose([
        T.ToPILImage(),
        T.Resize(tuple(input_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_train_loader(conf):
    transform = _train_transform(conf.input_size)
    trainset = DatasetFolderFT(
        conf.train_root_path, transform,
        ft_width=conf.ft_width, ft_height=conf.ft_height)
    return DataLoader(
        trainset,
        batch_size=conf.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=conf.num_workers,
        drop_last=True)


def get_val_loader(conf):
    transform = _val_transform(conf.input_size)
    valset = DatasetFolderVal(conf.val_root_path, transform)
    return DataLoader(
        valset,
        batch_size=conf.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=conf.num_workers)
