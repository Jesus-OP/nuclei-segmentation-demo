"""README figure -> docs/figures/example_test.png (the only figure kept in git).

One test image of median difficulty for the best method (per-image AP@0.5:0.95 from
evaluate.py), cropped to its most crowded 256x256 window: image, reference, and each
method's instances as filled overlays. Reads runs/eval/*_per_image.csv and the pred folders.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from skimage.color import label2rgb

from bioseg.evaluation import slug
from bioseg.viz import normalize

METHODS = {"Otsu + watershed": "Otsu + watershed",
           "StarDist 2D_versatile_fluo (sin ajustar)": "StarDist (pretrained, untuned)",
           "U-Net (entrenada en BBBC039)": "U-Net (trained on BBBC039)"}


def crowded_window(gt: np.ndarray, size: int, step: int = 16) -> tuple[slice, slice]:
    best, pos = -1, (0, 0)
    for y in range(0, gt.shape[0] - size + 1, step):
        for x in range(0, gt.shape[1] - size + 1, step):
            n = len(np.unique(gt[y:y + size, x:x + size]))
            if n > best:
                best, pos = n, (y, x)
    return slice(pos[0], pos[0] + size), slice(pos[1], pos[1] + size)


def show(ax, img, labels, title):
    colors = np.random.default_rng(0).random((64, 3)) * 0.8 + 0.2
    ax.imshow(img if labels is None else
              label2rgb(labels, img, bg_label=0, alpha=0.45, colors=colors),
              cmap="gray", interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--results", type=Path, default=Path("runs/results.csv"))
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--out", type=Path, default=Path("docs/figures/example_test.png"))
    args = ap.parse_args()

    res = pd.read_csv(args.results)
    res = res[(res.status == "ok") & (res.variant == "all")]
    best = res.sort_values("AP@0.5:0.95").iloc[-1]["method"]
    per_image = pd.read_csv(args.results.parent / "eval" / f"{slug(best)}_per_image.csv")
    ranked = per_image.sort_values("all_AP@0.5:0.95").reset_index(drop=True)
    stem = ranked.loc[len(ranked) // 2, "stem"]

    img = normalize(tifffile.imread(args.data / "images" / f"{stem}.tif"))
    gt = tifffile.imread(args.data / "labels" / f"{stem}.tif")
    sl = crowded_window(gt, args.crop)
    panels = [("Image", None), ("Reference (manual)", gt)]
    for _, r in res.iterrows():
        panels.append((METHODS.get(r.method, r.method),
                       tifffile.imread(Path(r.pred_dir) / f"{stem}.tif")))
    fig, ax = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.5))
    for a, (title, lab) in zip(ax, panels):
        show(a, img[sl], None if lab is None else lab[sl], title)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"{stem} ({best}, median per-image AP) -> {args.out}")


if __name__ == "__main__":
    main()
