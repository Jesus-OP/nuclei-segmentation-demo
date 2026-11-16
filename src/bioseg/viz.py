"""Figures shared by the baselines and the report."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


def normalize(img: np.ndarray, lo_pct: float = 1, hi_pct: float = 99.8) -> np.ndarray:
    """Percentile contrast stretch to [0, 1] (display only)."""
    lo, hi = np.percentile(img, [lo_pct, hi_pct])
    return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)


def pick_examples(per_image: pd.DataFrame, col: str = "all_AP@0.5:0.95") -> dict:
    """Best, median and worst image by `col`."""
    ranked = per_image.sort_values(col, ascending=False).reset_index(drop=True)
    return {"Bueno": ranked.iloc[0], "Medio": ranked.iloc[len(ranked) // 2],
            "Peor": ranked.iloc[-1]}


def examples_figure(data: Path, pred_dir: Path, per_image: pd.DataFrame, path: Path,
                    method: str) -> None:
    """3x3 grid: image / reference / prediction for the best, median and worst image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.color import label2rgb

    data, pred_dir = Path(data), Path(pred_dir)
    fig, ax = plt.subplots(3, 3, figsize=(15, 11.5))
    colors = np.random.default_rng(0).random((64, 3))
    for row, (name, r) in enumerate(pick_examples(per_image).items()):
        img = normalize(tifffile.imread(data / "images" / f"{r['stem']}.tif"))
        gt = tifffile.imread(data / "labels" / f"{r['stem']}.tif")
        pred = tifffile.imread(pred_dir / f"{r['stem']}.tif")
        panels = [
            (img, f"{name}: AP@0.5:0.95 = {r['all_AP@0.5:0.95']:.2f} · "
                  f"AP@0.5 = {r['all_AP@0.5']:.2f}"),
            (label2rgb(gt, img, bg_label=0, alpha=0.4, colors=colors),
             f"Referencia ({len(np.unique(gt)) - 1} núcleos)"),
            (label2rgb(pred, img, bg_label=0, alpha=0.4, colors=colors),
             f"{method} ({len(np.unique(pred)) - 1} objetos)"),
        ]
        for col, (im, title) in enumerate(panels):
            ax[row, col].imshow(im, cmap="gray" if im.ndim == 2 else None)
            ax[row, col].set_title(title, fontsize=10)
            ax[row, col].axis("off")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
