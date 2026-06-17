# -*- coding: utf-8 -*-
"""
Shared evaluation metrics (binary, positive class = real/bona-fide = 1).

Naming note for anti-spoofing:
    FAR (False Acceptance Rate) == APCER == FPR   (attack accepted as real)
    FRR (False Rejection Rate)  == BPCER == FNR   (real rejected as attack)
We report both name sets since they are the same quantities.
"""

import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix


def compute_eer(y_true, scores):
    """Equal Error Rate and the threshold at which it occurs."""
    fpr, tpr, thr = roc_curve(y_true, scores, pos_label=1)
    finite = np.isfinite(thr)               # roc_curve prepends an inf threshold
    fpr, tpr, thr = fpr[finite], tpr[finite], thr[finite]
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thr[idx])


def tpr_at_fpr(y_true, scores, target_fpr):
    """Highest TPR achievable while keeping FPR <= target_fpr."""
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    mask = fpr <= target_fpr
    return float(tpr[mask].max()) if mask.any() else 0.0


def binary_metrics(y_true, scores, threshold):
    """
    All point metrics at a decision threshold (score >= thr -> real=1).
    Returns a dict including precision, recall, F1, FAR, FRR, plus the
    ISO/IEC 30107-3 PAD metrics (APCER/BPCER/ACER) and HTER.
    """
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    eps = 1e-12
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)            # TPR
    specificity = tn / (tn + fp + eps)       # TNR
    f1 = 2 * precision * recall / (precision + recall + eps)

    far = fp / (fp + tn + eps)               # False Acceptance Rate (== APCER == FPR)
    frr = fn / (fn + tp + eps)               # False Rejection Rate  (== BPCER == FNR)
    apcer, bpcer = far, frr
    acer = (apcer + bpcer) / 2.0
    hter = (far + frr) / 2.0

    return {
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float(accuracy), "precision": float(precision),
        "recall_tpr": float(recall), "specificity_tnr": float(specificity),
        "f1": float(f1), "far": float(far), "frr": float(frr),
        "apcer": float(apcer), "bpcer": float(bpcer),
        "acer": float(acer), "hter": float(hter),
    }