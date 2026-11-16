"""Classical baseline: Gaussian smoothing + Otsu threshold + distance-transform watershed.

Two subcommands:
    tune     grid search on the train split; prints val metrics of the chosen setting and
             saves it to runs/classical/params.json. Never reads test.
    predict  segments a split with the saved params, writes label TIFFs to
             runs/classical/pred/<split>/, pooled + per-image metrics and, with --figure,
             the best / median / worst image.

Evaluation (pooled counts, "all" and "ge25" variants) lives in bioseg.evaluation.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # parallelism comes from --workers processes only

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

from bioseg.evaluation import (drop_small, evaluate_dir, format_summary, image_counts,
                               stems_of, summarize)
from bioseg.viz import examples_figure

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


def _eval_one(args):
    data, stem, params = args
    img = tifffile.imread(data / "images" / f"{stem}.tif")
    gt = tifffile.imread(data / "labels" / f"{stem}.tif")
    return image_counts(gt, segment(img, **params))


def evaluate_params(data: Path, split: str, params: dict, workers: int) -> dict:
    jobs = [(data, s, params) for s in stems_of(data, split)]
    with ProcessPoolExecutor(workers) as ex:
        return summarize(list(ex.map(_eval_one, jobs, chunksize=4)))


def cmd_tune(args) -> None:
    rows = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        r = evaluate_params(args.data, "train", params, args.workers)["all"]
        rows.append({**params, **r})
        print(params, f"AP@0.5:0.95={r['AP@0.5:0.95']:.3f}", flush=True)
    grid = pd.DataFrame(rows).sort_values("AP@0.5:0.95", ascending=False)
    args.out.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.out / "grid_train.csv", index=False)
    best = {k: grid.iloc[0][k] for k in GRID}
    best = {k: (int(v) if k in ("min_distance", "min_size") else float(v)) for k, v in best.items()}
    (args.out / "params.json").write_text(json.dumps(best, indent=2))
    print("\nMejores parámetros (train):", best)
    print(format_summary("val", evaluate_params(args.data, "val", best, args.workers)))


def _predict_one(args):
    data, stem, params, pred_dir = args
    img = tifffile.imread(data / "images" / f"{stem}.tif")
    tifffile.imwrite(pred_dir / f"{stem}.tif", segment(img, **params), compression="zlib")


def cmd_predict(args) -> None:
    params = json.loads((args.out / "params.json").read_text())
    pred_dir = args.out / "pred" / args.split
    pred_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(args.data, s, params, pred_dir) for s in stems_of(args.data, args.split)]
    with ProcessPoolExecutor(args.workers) as ex:
        list(ex.map(_predict_one, jobs, chunksize=4))
    summary, per_image = evaluate_dir(args.data, pred_dir, args.split, args.workers)
    per_image.to_csv(args.out / f"per_image_{args.split}.csv", index=False)
    pd.DataFrame(summary).T.to_csv(args.out / f"metrics_{args.split}.csv")
    print("Parámetros:", params)
    print(format_summary(args.split, summary))
    if args.figure:
        path = args.out / f"examples_{args.split}.png"
        examples_figure(args.data, pred_dir, per_image, path, "Otsu + watershed")
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
