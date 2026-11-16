"""Pretrained baseline: StarDist '2D_versatile_fluo', used as published (no fine-tuning,
default probability/NMS thresholds shipped with the model). CPU only.

Licenses (checked 2026-11-17, see LICENSES.md): code and weights BSD-3-Clause, training
data = subset of DSB 2018 stage1_train (BBBC038, CC0).

Input normalization is the one recommended by the StarDist authors for this model:
percentiles 1 / 99.8 per image. Outputs mirror scripts/baseline_classical.py:
runs/stardist/pred/<split>/<stem>.tif, metrics_<split>.csv, per_image_<split>.csv and,
with --figure, examples_<split>.png.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"       # never touch the GPU
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import time
from pathlib import Path


def set_threads(n: int) -> None:
    """Cap TensorFlow and OpenMP (StarDist NMS) to n threads. Call before importing TF."""
    os.environ["OMP_NUM_THREADS"] = str(n)
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(n)
    tf.config.threading.set_inter_op_parallelism_threads(1)


def load_model(StarDist2D):
    """from_pretrained() downloads and checks the sha256 of the zip, then creates a symlink
    to the extracted folder. On Windows without symlink privilege that last step fails
    (WinError 1314), so in that case load the already verified, extracted folder directly."""
    try:
        return StarDist2D.from_pretrained("2D_versatile_fluo")
    except OSError as e:
        cache = Path.home() / ".keras" / "models" / "StarDist2D" / "2D_versatile_fluo"
        extracted = cache / "2D_versatile_fluo_extracted"
        if not (extracted / "weights_best.h5").exists():
            raise
        print(f"Aviso: {e.__class__.__name__} al crear el enlace; cargo {extracted}")
        return StarDist2D(None, name=extracted.name, basedir=str(cache))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("runs/stardist"))
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--figure", action="store_true")
    args = ap.parse_args()

    set_threads(args.threads)
    import pandas as pd
    import tifffile
    from csbdeep.utils import normalize
    from stardist.models import StarDist2D

    from bioseg.evaluation import evaluate_dir, format_summary, stems_of
    from bioseg.viz import examples_figure

    model = load_model(StarDist2D)
    print("Umbrales del modelo (sin ajustar):", model.thresholds)

    pred_dir = args.out / "pred" / args.split
    pred_dir.mkdir(parents=True, exist_ok=True)
    stems = stems_of(args.data, args.split)
    t0 = time.time()
    for i, stem in enumerate(stems, 1):
        img = tifffile.imread(args.data / "images" / f"{stem}.tif")
        labels, _ = model.predict_instances(normalize(img, 1, 99.8), verbose=False)
        tifffile.imwrite(pred_dir / f"{stem}.tif", labels.astype("uint16"), compression="zlib")
        if i % 10 == 0:
            print(f"  {i}/{len(stems)} ({(time.time() - t0) / i:.1f} s/imagen)", flush=True)

    summary, per_image = evaluate_dir(args.data, pred_dir, args.split, args.threads)
    per_image.to_csv(args.out / f"per_image_{args.split}.csv", index=False)
    pd.DataFrame(summary).T.to_csv(args.out / f"metrics_{args.split}.csv")
    print(format_summary(args.split, summary))
    if args.figure:
        path = args.out / f"examples_{args.split}.png"
        examples_figure(args.data, pred_dir, per_image, path, "StarDist 2D_versatile_fluo")
        print("Figura:", path)


if __name__ == "__main__":
    main()
