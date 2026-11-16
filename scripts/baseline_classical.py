"""Classical baseline: Gaussian smoothing + Otsu threshold + distance-transform watershed.

Two subcommands:
    tune     grid search on the train split; prints val metrics of the chosen setting and
             saves it to runs/classical/params.json. Never reads test.
    predict  segments a split with the saved params, writes label TIFFs, per-image
             metrics and (optionally) a figure with the best / median / worst image.

Metrics are pooled over the split (TP/FP/FN summed per IoU threshold), in two variants:
"all" uses every reference object; "ge25" drops objects < 25 px from BOTH reference and
prediction before matching (see docs/notes_evaluate.md).
"""
import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage import feature, filters, measure, segmentation

from bioseg.metrics import match

THRESHOLDS = np.round(np.arange(0.5, 1.0, 0.05), 2)
MIN_AREA_EVAL = 25
GRID = {  # searched on train only. A first, coarser grid (sigma 1-2, otsu_factor 0.6-1.0,
    # min_distance 5-11, min_size 10/25) peaked at its edge (sigma=1, min_distance=11) and
    # min_size=25 always beat 10. A second grid peaked at sigma=0.5, otsu_factor=0.9 (edges
    # again), so this final grid brackets every parameter around the optimum.
    "sigma": [0.0, 0.5, 1.0],
    "otsu_factor": [0.8, 0.9, 1.0],
    "min_distance": [9, 11, 13, 15],
    "min_size": [25],
}


def segment(img: np.ndarray, sigma: float, otsu_factor: float,
            min_distance: int, min_size: int) -> np.ndarray:
    """Return a uint16 label image (0 = background)."""
    smooth = filters.gaussian(img.astype(np.float32), sigma=sigma)
    fg = smooth > otsu_factor * filters.threshold_otsu(smooth)
    fg = ndi.binary_fill_holes(drop_small(measure.label(fg), min_size) > 0)
    dist = ndi.distance_transform_edt(fg)
    peaks = feature.peak_local_max(dist, min_distance=min_distance, labels=measure.label(fg),
                                   exclude_border=False)
    markers = np.zeros(fg.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = drop_small(segmentation.watershed(-dist, markers, mask=fg), min_size)
    return segmentation.relabel_sequential(labels)[0].astype(np.uint16)


def drop_small(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Set objects with area < min_area to background."""
    areas = np.bincount(labels.ravel())
    out = labels.copy()
    out[areas[labels] < min_area] = 0
    return out


def image_counts(gt: np.ndarray, pred: np.ndarray) -> dict:
    """TP/FP/FN per IoU threshold for both variants: {(variant, thr): (tp, fp, fn)}."""
    out = {}
    for variant, (g, p) in {"all": (gt, pred),
                            "ge25": (drop_small(gt, MIN_AREA_EVAL),
                                     drop_small(pred, MIN_AREA_EVAL))}.items():
        for t in THRESHOLDS:
            m = match(g, p, float(t))
            out[(variant, float(t))] = (m["tp"], m["fp"], m["fn"])
    return out


def summarize(counts: list[dict]) -> dict:
    """Pool counts over images -> AP@0.5, AP@0.5:0.95, F1@0.5, mean F1@0.5:0.95."""
    res = {}
    for variant in ("all", "ge25"):
        ap, f1 = [], []
        for t in THRESHOLDS:
            tp, fp, fn = np.sum([c[(variant, float(t))] for c in counts], axis=0)
            ap.append(tp / (tp + fp + fn))
            f1.append(2 * tp / (2 * tp + fp + fn))
        res[variant] = {"AP@0.5": float(ap[0]), "AP@0.5:0.95": float(np.mean(ap)),
                        "F1@0.5": float(f1[0]), "F1@0.5:0.95": float(np.mean(f1))}
    return res


def load(data: Path, stem: str):
    img = tifffile.imread(data / "images" / f"{stem}.tif")
    gt = tifffile.imread(data / "labels" / f"{stem}.tif")
    return img, gt


def _eval_one(args):
    data, stem, params = args
    img, gt = load(data, stem)
    return image_counts(gt, segment(img, **params))


def evaluate(data: Path, stems: list[str], params: dict, workers: int) -> dict:
    with ProcessPoolExecutor(workers) as ex:
        counts = list(ex.map(_eval_one, [(data, s, params) for s in stems]))
    return summarize(counts)


def stems_of(data: Path, split: str) -> list[str]:
    df = pd.read_csv(data / "splits.csv")
    return df.loc[df["split"] == split, "stem"].tolist()


def cmd_tune(args) -> None:
    train = stems_of(args.data, "train")
    rows = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        r = evaluate(args.data, train, params, args.workers)["all"]
        rows.append({**params, **r})
        print(params, f"AP@0.5:0.95={r['AP@0.5:0.95']:.3f}", flush=True)
    grid = pd.DataFrame(rows).sort_values("AP@0.5:0.95", ascending=False)
    args.out.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.out / "grid_train.csv", index=False)
    best = {k: grid.iloc[0][k] for k in GRID}
    best = {k: (int(v) if k in ("min_distance", "min_size") else float(v)) for k, v in best.items()}
    (args.out / "params.json").write_text(json.dumps(best, indent=2))
    print("\nMejores parámetros (train):", best)
    print_metrics("train", grid.iloc[0])
    val = evaluate(args.data, stems_of(args.data, "val"), best, args.workers)
    for variant, r in val.items():
        print_metrics(f"val/{variant}", r)


def print_metrics(name: str, r) -> None:
    print(f"  {name:10s} " + "  ".join(f"{k}={r[k]:.3f}" for k in
                                        ("AP@0.5", "AP@0.5:0.95", "F1@0.5", "F1@0.5:0.95")))


def _predict_one(args):
    data, stem, params, pred_dir = args
    img, gt = load(data, stem)
    pred = segment(img, **params)
    tifffile.imwrite(pred_dir / f"{stem}.tif", pred, compression="zlib")
    return image_counts(gt, pred)


def cmd_predict(args) -> None:
    params = json.loads((args.out / "params.json").read_text())
    stems = stems_of(args.data, args.split)
    pred_dir = args.out / "pred" / args.split
    pred_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as ex:
        counts = list(ex.map(_predict_one, [(args.data, s, params, pred_dir) for s in stems]))
    per_image = pd.DataFrame([{"stem": s, **{f"{v}_{k}": x for v, r in summarize([c]).items()
                                            for k, x in r.items()}}
                              for s, c in zip(stems, counts)])
    per_image.to_csv(args.out / f"per_image_{args.split}.csv", index=False)
    total = summarize(counts)
    pd.DataFrame(total).T.to_csv(args.out / f"metrics_{args.split}.csv")
    print("Parámetros:", params)
    for variant, r in total.items():
        print_metrics(f"{args.split}/{variant}", r)
    if args.figure:
        figure(args.data, pred_dir, per_image, args.out / f"examples_{args.split}.png")


def figure(data: Path, pred_dir: Path, per_image: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.color import label2rgb

    ranked = per_image.sort_values("all_AP@0.5:0.95", ascending=False).reset_index(drop=True)
    picks = {"Bueno": ranked.iloc[0], "Medio": ranked.iloc[len(ranked) // 2],
             "Peor": ranked.iloc[-1]}
    fig, ax = plt.subplots(3, 3, figsize=(15, 11.5))
    rng = np.random.default_rng(0)
    for row, (name, r) in enumerate(picks.items()):
        img, gt = load(data, r["stem"])
        pred = tifffile.imread(pred_dir / f"{r['stem']}.tif")
        lo, hi = np.percentile(img, [1, 99.8])
        norm = np.clip((img - lo) / (hi - lo), 0, 1)
        colors = rng.random((64, 3))
        panels = [(norm, "Imagen"),
                  (label2rgb(gt, norm, bg_label=0, alpha=0.4, colors=colors),
                   f"Referencia ({gt.max()} núcleos)"),
                  (label2rgb(pred, norm, bg_label=0, alpha=0.4, colors=colors),
                   f"Otsu + watershed ({pred.max()} objetos)")]
        for col, (im, title) in enumerate(panels):
            ax[row, col].imshow(im, cmap="gray" if im.ndim == 2 else None)
            ax[row, col].set_title(title, fontsize=10)
            ax[row, col].axis("off")
        ax[row, 0].set_title(f"{name}: AP@0.5:0.95 = {r['all_AP@0.5:0.95']:.2f}, "
                             f"AP@0.5 = {r['all_AP@0.5']:.2f}", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    print("Figura:", path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("runs/classical"))
    ap.add_argument("--workers", type=int, default=4)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tune")
    p = sub.add_parser("predict")
    p.add_argument("--split", choices=["train", "val", "test"], required=True)
    p.add_argument("--figure", action="store_true")
    args = ap.parse_args()
    {"tune": cmd_tune, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()
