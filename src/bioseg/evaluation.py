"""Dataset-level evaluation shared by all methods (see docs/notes_evaluate.md).

- TP/FP/FN are pooled over all images of a split for each IoU threshold
  (not a mean of per-image scores).
- Two variants: "all" (every reference object) and "ge25" (objects < 25 px removed from
  BOTH reference and prediction before matching).
- AP = TP / (TP + FP + FN) (DSB 2018 style). F1 is reported too because Caicedo et al.
  report mean F1 over IoU 0.50:0.95.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from bioseg.metrics import match

THRESHOLDS = tuple(np.round(np.arange(0.5, 1.0, 0.05), 2).tolist())
MIN_AREA_EVAL = 25
VARIANTS = ("all", "ge25")
METRICS = ("AP@0.5", "AP@0.5:0.95", "F1@0.5", "F1@0.5:0.95")


def drop_small(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Set objects with area < min_area to background (labels are not renumbered)."""
    areas = np.bincount(labels.ravel())
    out = labels.copy()
    out[areas[labels] < min_area] = 0
    return out


def image_counts(gt: np.ndarray, pred: np.ndarray) -> dict:
    """{(variant, thr): (tp, fp, fn)} for one image."""
    pairs = {"all": (gt, pred),
             "ge25": (drop_small(gt, MIN_AREA_EVAL), drop_small(pred, MIN_AREA_EVAL))}
    out = {}
    for variant, (g, p) in pairs.items():
        for t in THRESHOLDS:
            m = match(g, p, t)
            out[(variant, t)] = (m["tp"], m["fp"], m["fn"])
    return out


def summarize(counts: list[dict]) -> dict:
    """Pool per-image counts -> {variant: {metric: value}}."""
    res = {}
    for variant in VARIANTS:
        ap, f1 = [], []
        for t in THRESHOLDS:
            tp, fp, fn = np.sum([c[(variant, t)] for c in counts], axis=0)
            denom = tp + fp + fn
            ap.append(tp / denom if denom else 1.0)
            f1.append(2 * tp / (2 * tp + fp + fn) if denom else 1.0)
        res[variant] = {"AP@0.5": float(ap[0]), "AP@0.5:0.95": float(np.mean(ap)),
                        "F1@0.5": float(f1[0]), "F1@0.5:0.95": float(np.mean(f1))}
    return res


def flat(summary: dict) -> dict:
    """{variant: {metric: v}} -> {"all_AP@0.5": v, ...}"""
    return {f"{v}_{k}": x for v, r in summary.items() for k, x in r.items()}


def stems_of(data: Path, split: str) -> list[str]:
    df = pd.read_csv(Path(data) / "splits.csv")
    return df.loc[df["split"] == split, "stem"].tolist()


def _counts_from_files(args):
    gt_path, pred_path = args
    return image_counts(tifffile.imread(gt_path), tifffile.imread(pred_path))


def evaluate_dir(data: Path, pred_dir: Path, split: str, workers: int = 4):
    """Evaluate every `<stem>.tif` label image in pred_dir against the reference.

    Returns (summary, per_image DataFrame). Raises if a prediction is missing, so a
    partial folder can never be reported as a full-split result.
    """
    data, pred_dir = Path(data), Path(pred_dir)
    stems = stems_of(data, split)
    missing = [s for s in stems if not (pred_dir / f"{s}.tif").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} predicciones faltan en {pred_dir}, "
                                f"p. ej. {missing[0]}.tif")
    jobs = [(data / "labels" / f"{s}.tif", pred_dir / f"{s}.tif") for s in stems]
    with ProcessPoolExecutor(workers) as ex:
        counts = list(ex.map(_counts_from_files, jobs, chunksize=4))
    per_image = pd.DataFrame([{"stem": s, **flat(summarize([c]))}
                              for s, c in zip(stems, counts)])
    return summarize(counts), per_image


def format_summary(name: str, summary: dict) -> str:
    return "\n".join(f"  {name}/{v:5s} " + "  ".join(f"{k}={summary[v][k]:.3f}" for k in METRICS)
                     for v in VARIANTS)
