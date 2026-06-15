# -*- coding: utf-8 -*-
"""
Default training configuration.

Note: FT-target dimensions (conf.ft_height / conf.ft_width) are NOT set here.
They depend on the backbone's tap feature-map size and are filled in by
TrainMain once the network is built.
"""

import torch
from datetime import datetime
from easydict import EasyDict

from src.utility import make_if_not_exist


def get_default_config():
    conf = EasyDict()

    # ---------------- training ----------------
    conf.lr = 1e-2                 # learning rate for the NEW params (head + FTGenerator)
    conf.epochs = 25
    conf.momentum = 0.9
    conf.weight_decay = 5e-4
    conf.batch_size = 64           # 224x224 is far heavier than the original 80x80
    conf.num_workers = 8

    # ---------------- discriminative learning rate ----------------
    # Pretrained backbone moves slower than the freshly-initialised head/FT branch.
    conf.use_discriminative_lr = True
    conf.backbone_lr = 1e-3        # lr for the pretrained MobileNetV3 features
    conf.no_wd_on_bn_bias = True   # exclude BatchNorm scales/biases from weight decay

    # ---------------- warmup + scheduler ----------------
    conf.warmup_epochs = 2         # linear warmup so the random branches settle first
    conf.warmup_start_factor = 0.01  # start warmup at 1% of each group's target lr
    conf.scheduler = "cosine"      # "cosine" or "multistep"
    conf.eta_min = 1e-5            # cosine floor
    conf.milestones = [10, 15, 22]  # used only when scheduler == "multistep"
    conf.gamma = 0.1               # used only when scheduler == "multistep"

    # ---------------- loss weights ----------------
    conf.cls_loss_weight = 0.5
    conf.ft_loss_weight = 0.5

    # ---------------- model ----------------
    conf.backbone = "mobilenet_v3_large"   # or "mobilenet_v3_small"
    conf.pretrained = True
    conf.num_classes = 2                    # real / spoof
    conf.real_label = 1                     # folder "1" == real -> positive class for AUC
    conf.input_channel = 3
    conf.input_size = [224, 224]            # [H, W]
    conf.tap_min_spatial = 14               # FT branch taps deepest stage >= this size

    # ---------------- dataset ----------------
    conf.train_root_path = "./datasets/train"
    conf.val_root_path = "./datasets/val"

    # ---------------- logging / checkpoints ----------------
    conf.snapshot_dir_path = "./saved_logs/snapshot"
    conf.log_path = "./saved_logs/jobs"
    conf.board_loss_every = 10

    return conf


def update_config(args, conf):
    conf.devices = args.devices
    conf.backbone = args.backbone
    conf.pretrained = not args.no_pretrained
    conf.num_classes = args.num_classes
    conf.batch_size = args.batch_size
    conf.epochs = args.epochs
    conf.lr = args.lr
    conf.backbone_lr = args.backbone_lr
    conf.use_discriminative_lr = not args.no_discriminative_lr
    conf.warmup_epochs = args.warmup_epochs
    conf.scheduler = args.scheduler
    conf.train_root_path = args.train_root
    conf.val_root_path = args.val_root
    conf.tap_min_spatial = args.tap_min_spatial
    conf.input_size = [args.input_size, args.input_size]

    conf.device = "cuda:{}".format(conf.devices[0]) if torch.cuda.is_available() else "cpu"

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    job_name = "AntiSpoofing_{}_{}".format(conf.backbone, args.input_size)
    log_path = "{}/{}/{}".format(conf.log_path, job_name, current_time)
    snapshot_dir = "{}/{}".format(conf.snapshot_dir_path, job_name)

    make_if_not_exist(snapshot_dir)
    make_if_not_exist(log_path)

    conf.model_path = snapshot_dir
    conf.log_path = log_path
    conf.job_name = job_name
    return conf
