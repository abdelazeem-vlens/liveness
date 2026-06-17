# -*- coding: utf-8 -*-
"""
Training loop.

Keeps the original two-term objective:
    loss = cls_w * CrossEntropy(cls, label) + ft_w * MSE(ft, ft_target)

Adds:
  * Per-epoch validation with ROC-AUC (positive class = `conf.real_label`).
  * Best-checkpoint saving by val AUC (best.pth), plus last.pth each epoch.
"""

import os
import torch
from torch import optim
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score

from src.logger import setup_logger
from src.model_lib.MultiFTNet import MultiFTNet
from src.data_io.dataset_loader import get_train_loader, get_val_loader


class TrainMain:
    def __init__(self, conf):
        self.conf = conf
        self.board_loss_every = conf.board_loss_every
        self.step = 0
        self.start_epoch = 0
        self.best_auc = -1.0
        self.best_epoch = -1

        # File logger (mirrors to console).
        self.log = setup_logger("train", os.path.join(conf.log_path, "train.log"))
        self.log.info("=" * 70)
        self.log.info("TRAINING RUN: %s", conf.job_name)
        self.log.info("=" * 70)
        self._log_config()

        # Build the network FIRST so we can read the FT-target spatial size,
        # then configure the loaders to produce matching FT targets.
        self.model = self._define_network()
        self.conf.ft_height, self.conf.ft_width = self.raw_model.ft_spatial
        n_params = sum(p.numel() for p in self.raw_model.parameters()) / 1e6
        self.log.info(
            "model: %s | params: %.2fM | tap channels: %d | FT target: %dx%d",
            conf.backbone, n_params, self.raw_model.tap_channels,
            conf.ft_height, conf.ft_width)

        self.train_loader = get_train_loader(self.conf)
        self.val_loader = get_val_loader(self.conf)
        self.log.info("train samples: %d | val samples: %d",
                      len(self.train_loader.dataset), len(self.val_loader.dataset))

    def _log_config(self):
        self.log.info("Configuration:")
        for k in sorted(self.conf.keys()):
            self.log.info("    %-22s : %s", k, self.conf[k])

    # ------------------------------------------------------------------ #
    def train_model(self):
        self._init_optimizer()
        self._train_stage()

    def _define_network(self):
        net = MultiFTNet(
            backbone=self.conf.backbone,
            pretrained=self.conf.pretrained,
            num_classes=self.conf.num_classes,
            input_size=tuple(self.conf.input_size),
            tap_min_spatial=self.conf.tap_min_spatial,
            img_channel=self.conf.input_channel,
        )
        self.raw_model = net                       # unwrapped reference
        net = net.to(self.conf.device)
        net = torch.nn.DataParallel(net, self.conf.devices)
        net.to(self.conf.device)
        return net

    def _build_param_groups(self):
        """
        Build optimizer param groups:
          * pretrained backbone (features_low + features_high) at backbone_lr
          * new params (classifier + FTGenerator) at conf.lr
        With BatchNorm scales/biases optionally excluded from weight decay.
        """
        conf = self.conf
        model = self.raw_model

        backbone_modules = [model.features_low, model.features_high]
        head_modules = [model.classifier, model.FTGenerator]

        def split(modules, lr):
            decay, no_decay = [], []
            for m in modules:
                for name, p in m.named_parameters():
                    if not p.requires_grad:
                        continue
                    if conf.no_wd_on_bn_bias and (p.ndim == 1 or name.endswith("bias")):
                        no_decay.append(p)
                    else:
                        decay.append(p)
            groups = []
            if decay:
                groups.append({"params": decay, "lr": lr,
                               "weight_decay": conf.weight_decay})
            if no_decay:
                groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
            return groups

        head_lr = conf.lr
        backbone_lr = conf.backbone_lr if conf.use_discriminative_lr else conf.lr
        return split(backbone_modules, backbone_lr) + split(head_modules, head_lr)

    def _build_scheduler(self):
        conf = self.conf
        opt = self.optimizer
        warmup = max(0, int(conf.warmup_epochs))

        if conf.scheduler == "cosine":
            main = optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, conf.epochs - warmup), eta_min=conf.eta_min)
        elif conf.scheduler == "multistep":
            # shift milestones so they still refer to absolute epoch numbers
            milestones = [m - warmup for m in conf.milestones if m - warmup > 0]
            main = optim.lr_scheduler.MultiStepLR(opt, milestones, conf.gamma)
        else:
            raise ValueError("Unknown scheduler: {}".format(conf.scheduler))

        if warmup > 0:
            warm = optim.lr_scheduler.LinearLR(
                opt, start_factor=conf.warmup_start_factor, end_factor=1.0,
                total_iters=warmup)
            return optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warm, main], milestones=[warmup])
        return main

    def _init_optimizer(self):
        self.cls_criterion = CrossEntropyLoss()
        self.ft_criterion = MSELoss()
        self.optimizer = optim.SGD(
            self._build_param_groups(),
            lr=self.conf.lr,
            momentum=self.conf.momentum)
        self.schedule_lr = self._build_scheduler()
        self.log.info(
            "optimizer: SGD | head_lr: %s | backbone_lr: %s | warmup: %d | sched: %s",
            self.conf.lr,
            self.conf.backbone_lr if self.conf.use_discriminative_lr else self.conf.lr,
            self.conf.warmup_epochs, self.conf.scheduler)

    # ------------------------------------------------------------------ #
    def _train_stage(self):
        self.writer = SummaryWriter(self.conf.log_path)
        running_loss = running_acc = running_cls = running_ft = 0.0

        for e in range(self.start_epoch, self.conf.epochs):
            self.model.train()
            lr_now = self.schedule_lr.get_last_lr()
            self.log.info("-" * 70)
            self.log.info("epoch %d started | lr: %s",
                          e, [round(x, 6) for x in lr_now])

            # epoch-level accumulators for the log file
            ep_loss = ep_acc = ep_cls = ep_ft = 0.0
            ep_batches = 0

            for sample, ft_sample, target in tqdm(self.train_loader):
                loss, acc, loss_cls, loss_ft = self._train_batch(sample, ft_sample, target)
                running_loss += loss
                running_acc += acc
                running_cls += loss_cls
                running_ft += loss_ft
                ep_loss += loss
                ep_acc += acc
                ep_cls += loss_cls
                ep_ft += loss_ft
                ep_batches += 1
                self.step += 1

                if self.step % self.board_loss_every == 0:
                    n = self.board_loss_every
                    self.writer.add_scalar("Training/Loss", running_loss / n, self.step)
                    self.writer.add_scalar("Training/Acc", running_acc / n, self.step)
                    self.writer.add_scalar("Training/Loss_cls", running_cls / n, self.step)
                    self.writer.add_scalar("Training/Loss_ft", running_ft / n, self.step)
                    lrs = [g["lr"] for g in self.optimizer.param_groups]
                    self.writer.add_scalar("Training/LR_backbone", min(lrs), self.step)
                    self.writer.add_scalar("Training/LR_head", max(lrs), self.step)
                    running_loss = running_acc = running_cls = running_ft = 0.0

            self.schedule_lr.step()

            # ---- validation + checkpointing (per epoch) ----
            auc = self._evaluate()
            self.writer.add_scalar("Val/AUC", auc, e)

            denom = max(1, ep_batches)
            self.log.info(
                "epoch %d done | train_loss: %.4f (cls: %.4f, ft: %.4f) | "
                "train_acc: %.4f | val_AUC: %.4f",
                e, ep_loss / denom, ep_cls / denom, ep_ft / denom,
                ep_acc / denom, auc)

            self._save_state("last")
            if auc > self.best_auc:
                self.best_auc = auc
                self.best_epoch = e
                self._save_state("best")
                self.log.info("  -> new best val AUC %.4f at epoch %d (saved best.pth)",
                              auc, e)

        self.writer.close()
        self.log.info("=" * 70)
        self.log.info("TRAINING COMPLETE | best val AUC: %.4f at epoch %d",
                      self.best_auc, self.best_epoch)
        self.log.info("best checkpoint: %s",
                      os.path.join(self.conf.model_path, "best.pth"))
        self.log.info("=" * 70)

    def _train_batch(self, sample, ft_sample, target):
        self.optimizer.zero_grad()
        sample = sample.to(self.conf.device)
        ft_target = ft_sample.to(self.conf.device)
        target = target.to(self.conf.device)

        cls, ft = self.model.forward(sample)
        loss_cls = self.cls_criterion(cls, target)
        loss_ft = self.ft_criterion(ft, ft_target)
        loss = self.conf.cls_loss_weight * loss_cls + self.conf.ft_loss_weight * loss_ft

        acc = self._accuracy(cls, target)
        loss.backward()
        self.optimizer.step()
        return loss.item(), acc, loss_cls.item(), loss_ft.item()

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        all_scores, all_labels = [], []
        for sample, target in tqdm(self.val_loader, desc="val"):
            sample = sample.to(self.conf.device)
            logits = self.model(sample)                       # (N, num_classes)
            prob_real = F.softmax(logits, dim=1)[:, self.conf.real_label]
            all_scores.append(prob_real.cpu())
            all_labels.append(target)

        scores = torch.cat(all_scores).numpy()
        labels = torch.cat(all_labels).numpy()
        bin_labels = (labels == self.conf.real_label).astype(int)
        return roc_auc_score(bin_labels, scores)

    def _accuracy(self, output, target):
        pred = output.argmax(dim=1)
        return (pred == target).float().mean().item()

    def _save_state(self, tag):
        path = os.path.join(self.conf.model_path, "{}.pth".format(tag))
        # Save the unwrapped state_dict (no DataParallel 'module.' prefix).
        torch.save(self.model.module.state_dict(), path)