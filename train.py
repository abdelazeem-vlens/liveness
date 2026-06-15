# -*- coding: utf-8 -*-
"""
Entry point.

Example:
    python train.py --device_ids 0 \
        --backbone mobilenet_v3_large \
        --input_size 224 \
        --train_root ./datasets/train \
        --val_root   ./datasets/val
"""

import os
import argparse

from src.train_main import TrainMain
from src.default_config import get_default_config, update_config


def parse_args():
    parser = argparse.ArgumentParser(description="Silence-FAS (MobileNetV3 + FT branch)")
    parser.add_argument("--device_ids", type=str, default="0", help="gpu ids, e.g. 0 or 01")
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_large",
                        choices=["mobilenet_v3_large", "mobilenet_v3_small"])
    parser.add_argument("--no_pretrained", action="store_true",
                        help="train backbone from scratch instead of ImageNet weights")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--input_size", type=int, default=224, help="square input size")
    parser.add_argument("--tap_min_spatial", type=int, default=14,
                        help="FT branch taps deepest backbone stage >= this spatial size")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-2, help="lr for new params (head + FT branch)")
    parser.add_argument("--backbone_lr", type=float, default=1e-3, help="lr for pretrained backbone")
    parser.add_argument("--no_discriminative_lr", action="store_true",
                        help="use a single lr for all params instead of split backbone/head lrs")
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "multistep"])
    parser.add_argument("--train_root", type=str, default="./datasets/train")
    parser.add_argument("--val_root", type=str, default="./datasets/val")
    args = parser.parse_args()

    cuda_devices = [int(x) for x in args.device_ids]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cuda_devices))
    args.devices = list(range(len(cuda_devices)))
    return args


if __name__ == "__main__":
    args = parse_args()
    conf = get_default_config()
    conf = update_config(args, conf)
    trainer = TrainMain(conf)
    trainer.train_model()
