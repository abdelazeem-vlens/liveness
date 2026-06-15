# -*- coding: utf-8 -*-
"""
Dataset definitions.

  * DatasetFolderFT  - training set; returns (image, ft_target, label).
                       The Fourier magnitude spectrum is computed from the raw
                       loaded image (before augmentation), exactly as in the
                       original repo, then resized to the FT-target size.
  * DatasetFolderVal - validation set; returns (image, label). No FT needed,
                       since the auxiliary branch is unused at eval time.

Folder layout expected (ImageFolder convention):
    root/0/xxx.jpg   -> spoof  (class index 0)
    root/1/yyy.jpg   -> real   (class index 1)
"""

import cv2
import torch
import numpy as np
from torchvision import datasets


def opencv_loader(path):
    """Load a BGR image with OpenCV (kept BGR here; converted later)."""
    return cv2.imread(path)


def generate_FT(image_bgr):
    """
    Magnitude spectrum (log-scaled, min-max normalised to ~[0, 1]) of the
    grayscale image. Phase is discarded. Identical in spirit to the original.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift) + 1)
    minn, maxx = fimg.min(), fimg.max()
    fimg = (fimg - minn + 1) / (maxx - minn + 1)
    return fimg


class DatasetFolderFT(datasets.ImageFolder):
    """Training dataset: image + Fourier-spectrum target + label."""

    def __init__(self, root, transform=None, target_transform=None,
                 ft_width=14, ft_height=14, loader=opencv_loader):
        super(DatasetFolderFT, self).__init__(
            root, transform, target_transform, loader)
        self.root = root
        self.ft_width = ft_width
        self.ft_height = ft_height

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)             # BGR numpy
        assert sample is not None, "image is None --> {}".format(path)

        # FT target from the RAW (un-augmented) image, then downsize.
        ft_sample = generate_FT(sample)
        ft_sample = cv2.resize(ft_sample, (self.ft_width, self.ft_height))
        ft_sample = torch.from_numpy(ft_sample).float().unsqueeze(0)  # (1, H, W)

        # Convert image to RGB for the (ImageNet-pretrained) backbone.
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, ft_sample, target


class DatasetFolderVal(datasets.ImageFolder):
    """Validation dataset: image + label (no FT branch at eval time)."""

    def __init__(self, root, transform=None, target_transform=None,
                 loader=opencv_loader):
        super(DatasetFolderVal, self).__init__(
            root, transform, target_transform, loader)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        assert sample is not None, "image is None --> {}".format(path)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
