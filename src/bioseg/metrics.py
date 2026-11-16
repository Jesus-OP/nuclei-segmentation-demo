"""Metricas de segmentacion de instancias para mascaras etiquetadas (0 = fondo).

Sin dependencias de ningun framework de DL: sirven para evaluar cualquier modelo.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """IoU entre cada objeto de `gt` (filas) y cada objeto de `pred` (columnas)."""
    if gt.shape != pred.shape:
        raise ValueError(f"shapes distintos: {gt.shape} vs {pred.shape}")
    gt_ids = np.unique(gt[gt > 0])
    pr_ids = np.unique(pred[pred > 0])
    if len(gt_ids) == 0 or len(pr_ids) == 0:
        return np.zeros((len(gt_ids), len(pr_ids)), dtype=float)
    gt_idx = np.searchsorted(gt_ids, gt.ravel())
    pr_idx = np.searchsorted(pr_ids, pred.ravel())
    both = (gt.ravel() > 0) & (pred.ravel() > 0)
    inter = np.zeros((len(gt_ids), len(pr_ids)), dtype=np.int64)
    np.add.at(inter, (gt_idx[both], pr_idx[both]), 1)
    gt_area = np.bincount(gt_idx[gt.ravel() > 0], minlength=len(gt_ids))
    pr_area = np.bincount(pr_idx[pred.ravel() > 0], minlength=len(pr_ids))
    union = gt_area[:, None] + pr_area[None, :] - inter
    return inter / union


def match(gt: np.ndarray, pred: np.ndarray, thr: float = 0.5) -> dict:
    """Emparejamiento uno-a-uno (hungaro) y recuento TP/FP/FN a un umbral de IoU."""
    iou = iou_matrix(gt, pred)
    n_gt, n_pred = iou.shape
    tp, matched_iou = 0, []
    if n_gt and n_pred:
        rows, cols = linear_sum_assignment(-iou)
        ok = iou[rows, cols] >= thr
        tp = int(ok.sum())
        matched_iou = iou[rows, cols][ok].tolist()
    fp, fn = n_pred - tp, n_gt - tp
    denom = tp + fp + fn
    return {
        "thr": thr, "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 1.0 if n_gt == 0 else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 1.0 if n_pred == 0 else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if denom else 1.0,
        "ap": tp / denom if denom else 1.0,  # "average precision" estilo DSB 2018
        "mean_matched_iou": float(np.mean(matched_iou)) if matched_iou else 0.0,
    }


def average_precision(gt, pred, thresholds=np.arange(0.5, 1.0, 0.05)) -> float:
    """Media de AP sobre umbrales de IoU 0.50:0.05:0.95."""
    return float(np.mean([match(gt, pred, float(t))["ap"] for t in thresholds]))
