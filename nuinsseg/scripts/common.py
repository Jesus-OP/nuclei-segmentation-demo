"""Shared helpers for the NuInsSeg scripts (paths, H&E colour handling, evaluation)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))      # reuse the BBBC039 scripts (baseline_classical)

from bioseg.evaluation import evaluate_dir, format_summary  # noqa: E402

DATA = Path("data/nuinsseg")
RUNS = Path("runs/nuinsseg")


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


def hematoxylin(rgb: np.ndarray) -> np.ndarray:
    """Hematoxylin optical density (colour deconvolution, Ruifrok & Johnston stain
    vectors as in skimage.color.rgb2hed), stretched to [0, 1] with percentiles 1/99.8.
    Nuclei are bright."""
    from skimage.color import rgb2hed
    h = rgb2hed(rgb[..., :3])[..., 0]
    lo, hi = np.percentile(h, [1, 99.8])
    return np.clip((h - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def evaluate(pred_dir: Path, split: str, out: Path, workers: int = 4,
             data: Path = DATA) -> dict:
    """Evaluate a prediction folder twice: on every reference nucleus (main result) and
    ignoring objects inside the authors' "ambiguous area" masks. Writes
    metrics_<split>.csv (rows = ignore_vague x variant) and per_image_<split>.csv."""
    res, per = {}, []
    for ignore in (False, True):
        summary, per_image = evaluate_dir(data, pred_dir, split, workers,
                                          ignore_dir=data / "vague" if ignore else None)
        res[ignore] = summary
        per.append(per_image.assign(ignore_vague=ignore))
        print(("Sin zonas dudosas:" if ignore else "Todos los núcleos:"))
        print(format_summary(split, summary))
    rows = [{"ignore_vague": ig, "variant": v, **m} for ig, s in res.items()
            for v, m in s.items()]
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"metrics_{split}.csv", index=False)
    pd.concat(per).to_csv(out / f"per_image_{split}.csv", index=False)
    return res
