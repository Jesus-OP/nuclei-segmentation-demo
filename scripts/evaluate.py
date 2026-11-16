"""Compare segmentation methods on one split (default: test) -> runs/results.csv.

Each method is a folder of label TIFFs named <stem>.tif (0 = background). Add any method
without touching code:
    python scripts/evaluate.py --method "U-Net (propia)=runs/unet/pred/test"

By default the three project methods are evaluated; a method whose folder does not exist
yet is written to results.csv with status "pendiente" and empty metrics.

Outputs (in --out, default runs/):
    results.csv                    one row per method x variant ("all", "ge25")
    eval/<slug>_per_image.csv      per-image metrics (for picking examples)
    eval/<slug>_objects.csv        per reference object: area, touching, at_border, IoU,
                                   found at IoU >= 0.5 (for the failure-case analysis)
Metric definitions: src/bioseg/evaluation.py and docs/notes_evaluate.md.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
from pathlib import Path

import pandas as pd

from bioseg.evaluation import (METRICS, VARIANTS, evaluate_dir, format_summary,
                               object_tables_dir, slug)

DEFAULT_METHODS = {  # name -> folder template ({split} is filled in)
    "Otsu + watershed": "runs/classical/pred/{split}",
    "StarDist 2D_versatile_fluo (sin ajustar)": "runs/stardist/pred/{split}",
    "U-Net (entrenada en BBBC039)": "runs/unet/pred/{split}",
}


def parse_methods(items: list[str] | None, split: str) -> dict[str, Path]:
    if not items:
        return {k: Path(v.format(split=split)) for k, v in DEFAULT_METHODS.items()}
    methods = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep:
            raise SystemExit(f"--method espera NOMBRE=CARPETA, recibido: {item}")
        methods[name.strip()] = Path(path.strip())
    return methods


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--method", action="append", metavar="NOMBRE=CARPETA",
                    help="repeatable; default: the three project methods")
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    (args.out / "eval").mkdir(parents=True, exist_ok=True)
    rows = []
    for name, folder in parse_methods(args.method, args.split).items():
        if not folder.is_dir():
            print(f"[pendiente] {name}: no existe {folder}")
            rows += [{"method": name, "variant": v, "status": "pendiente",
                      "split": args.split, "pred_dir": folder.as_posix()} for v in VARIANTS]
            continue
        summary, per_image = evaluate_dir(args.data, folder, args.split, args.workers)
        objects = object_tables_dir(args.data, folder, args.split, args.workers)
        per_image.to_csv(args.out / "eval" / f"{slug(name)}_per_image.csv", index=False)
        objects.to_csv(args.out / "eval" / f"{slug(name)}_objects.csv", index=False)
        print(name)
        print(format_summary(args.split, summary))
        rows += [{"method": name, "variant": v, "status": "ok", "split": args.split,
                  "n_images": len(per_image), "pred_dir": folder.as_posix(), **summary[v]}
                 for v in VARIANTS]
    cols = ["method", "variant", "status", *METRICS, "split", "n_images", "pred_dir"]
    pd.DataFrame(rows).reindex(columns=cols).to_csv(args.out / "results.csv", index=False)
    print("Tabla:", args.out / "results.csv")


if __name__ == "__main__":
    main()
