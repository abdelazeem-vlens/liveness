# -*- coding: utf-8 -*-
"""
Evaluate a trained anti-spoofing model on a test set and write a detailed log.

Reports, for the "real" class as positive:
  * ROC-AUC, EER (+ EER threshold)
  * At each operating threshold (0.5 and the EER threshold):
      accuracy, confusion matrix, precision, recall(TPR), specificity(TNR), F1,
      and the PAD metrics APCER / BPCER / ACER (ISO/IEC 30107-3) and HTER
  * TPR at fixed FPR operating points (0.1, 0.01, 0.001)

Outputs (next to the checkpoint, or --out_dir):
  * eval_<timestamp>.log    - full human-readable report
  * preds_<timestamp>.csv   - per-image path,label,score,pred  (--save_predictions)
  * roc_<timestamp>.png + hist_<timestamp>.png  (--save_plots, needs matplotlib)

Example:
    python test.py --model_path ./saved_logs/snapshot/AntiSpoofing_mobilenet_v3_large_224/best.pth \
                   --test_root ./datasets/test --backbone mobilenet_v3_large --input_size 224
"""

import os
import csv
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

from src.logger import setup_logger
from src.metrics import compute_eer, tpr_at_fpr, binary_metrics
from src.model_lib.MultiFTNet import MultiFTNet
from src.data_io.dataset_folder import DatasetFolderVal
from src.data_io.dataset_loader import _val_transform


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def build_model(args, device):
    model = MultiFTNet(
        backbone=args.backbone,
        pretrained=False,                    # weights come from the checkpoint
        num_classes=args.num_classes,
        input_size=(args.input_size, args.input_size),
        tap_min_spatial=args.tap_min_spatial,
        img_channel=3,
    )
    state = torch.load(args.model_path, map_location=device)
    # checkpoints are now full dicts {model, optimizer, ...}; accept bare ones too
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    # tolerate both bare and DataParallel-prefixed checkpoints
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


@torch.no_grad()
def run_inference(model, loader, device, real_label):
    scores, labels = [], []
    from tqdm import tqdm
    for sample, target in tqdm(loader, desc="eval"):
        sample = sample.to(device)
        logits = model(sample)
        prob_real = F.softmax(logits, dim=1)[:, real_label]
        scores.append(prob_real.cpu())
        labels.append(target)
    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()
    return scores, labels


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def log_block(log, name, m):
    log.info("  [%s] threshold=%.4f", name, m["threshold"])
    log.info("      confusion:  TN=%d  FP=%d  FN=%d  TP=%d",
             m["tn"], m["fp"], m["fn"], m["tp"])
    log.info("      accuracy=%.4f  precision=%.4f  recall/TPR=%.4f  specificity/TNR=%.4f  F1=%.4f",
             m["accuracy"], m["precision"], m["recall_tpr"], m["specificity_tnr"], m["f1"])
    log.info("      FAR=%.4f  FRR=%.4f", m["far"], m["frr"])
    log.info("      APCER=%.4f  BPCER=%.4f  ACER=%.4f  HTER=%.4f",
             m["apcer"], m["bpcer"], m["acer"], m["hter"])


def maybe_save_plots(scores, y_true, stamp, out_dir, log):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        log.info("matplotlib not available - skipping plots")
        return

    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC"); plt.legend()
    roc_path = os.path.join(out_dir, "roc_{}.png".format(stamp))
    plt.savefig(roc_path, bbox_inches="tight"); plt.close()

    plt.figure()
    plt.hist(scores[y_true == 1], bins=40, alpha=0.6, label="real")
    plt.hist(scores[y_true == 0], bins=40, alpha=0.6, label="spoof")
    plt.xlabel("score (P(real))"); plt.ylabel("count")
    plt.title("Score distribution"); plt.legend()
    hist_path = os.path.join(out_dir, "hist_{}.png".format(stamp))
    plt.savefig(hist_path, bbox_inches="tight"); plt.close()
    log.info("saved plots: %s , %s", roc_path, hist_path)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available()
                          or "cpu" in args.device else "cpu")

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.model_path))
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    log = setup_logger("eval", os.path.join(out_dir, "eval_{}.log".format(stamp)))

    log.info("=" * 70)
    log.info("EVALUATION")
    log.info("=" * 70)
    log.info("checkpoint : %s", args.model_path)
    log.info("test_root  : %s", args.test_root)
    log.info("backbone   : %s | input_size: %d | num_classes: %d | real_label: %d",
             args.backbone, args.input_size, args.num_classes, args.real_label)
    log.info("device     : %s", device)

    # data
    transform = _val_transform([args.input_size, args.input_size])
    dataset = DatasetFolderVal(args.test_root, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    paths = [p for p, _ in dataset.samples]
    log.info("class_to_idx: %s", dataset.class_to_idx)

    # model + inference
    model = build_model(args, device)
    scores, labels = run_inference(model, loader, device, args.real_label)
    y_true = (labels == args.real_label).astype(int)

    n_real = int((y_true == 1).sum())
    n_spoof = int((y_true == 0).sum())
    log.info("samples: total=%d | real=%d | spoof=%d", len(y_true), n_real, n_spoof)

    # threshold-independent
    log.info("-" * 70)
    if n_real > 0 and n_spoof > 0:
        auc = roc_auc_score(y_true, scores)
        eer, eer_thr = compute_eer(y_true, scores)
        log.info("ROC-AUC : %.4f", auc)
        log.info("EER     : %.4f  (threshold=%.4f)", eer, eer_thr)
        for t in (0.1, 0.01, 0.001):
            log.info("TPR @ FPR=%-5s : %.4f", t, tpr_at_fpr(y_true, scores, t))
    else:
        log.info("Only one class present - AUC/EER undefined.")
        eer_thr = 0.5

    # threshold-dependent
    log.info("-" * 70)
    log.info("Point metrics:")
    log_block(log, "thr=0.50", binary_metrics(y_true, scores, 0.5))
    log_block(log, "thr=EER ", binary_metrics(y_true, scores, eer_thr))
    if args.threshold is not None:
        log_block(log, "thr=user", binary_metrics(y_true, scores, args.threshold))

    # per-image predictions
    if args.save_predictions:
        csv_path = os.path.join(out_dir, "preds_{}.csv".format(stamp))
        thr = args.threshold if args.threshold is not None else 0.5
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "score_real", "pred", "correct"])
            for p, lab, sc in zip(paths, y_true, scores):
                pred = int(sc >= thr)
                w.writerow([p, int(lab), "{:.6f}".format(sc), pred, int(pred == lab)])
        log.info("saved per-image predictions: %s", csv_path)

    if args.save_plots:
        maybe_save_plots(scores, y_true, stamp, out_dir, log)

    log.info("=" * 70)
    log.info("DONE")
    log.info("=" * 70)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate anti-spoofing model")
    p.add_argument("--model_path", type=str, required=True, help="path to best.pth")
    p.add_argument("--test_root", type=str, required=True,
                   help="test set root (ImageFolder: 0=spoof, 1=real)")
    # must match training
    p.add_argument("--backbone", type=str, default="mobilenet_v3_large",
                   choices=["mobilenet_v3_large", "mobilenet_v3_small"])
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--tap_min_spatial", type=int, default=14)
    p.add_argument("--real_label", type=int, default=1)
    # eval options
    p.add_argument("--threshold", type=float, default=None,
                   help="optional extra operating threshold to report")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--save_predictions", action="store_true")
    p.add_argument("--save_plots", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()