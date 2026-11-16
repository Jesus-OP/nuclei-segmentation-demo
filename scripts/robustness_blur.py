"""Stress test: how much do the baselines degrade on synthetically defocused images?

BBBC039 contains no clearly out-of-focus fields (checked visually on the train/val images
with the lowest Laplacian-variance score: they were sharp, the score was driven by
saturated objects and debris). Defocus is a common failure in real screens, so we
simulate it with a Gaussian blur of the raw image (sigma in pixels) and re-run both
baselines with their frozen settings (no re-tuning). The reference is unchanged.

Output: runs/robustness_blur.csv (method, sigma, pooled "all" and "ge25" metrics).
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0, 1, 2, 4])
    ap.add_argument("--classical-params", type=Path, default=Path("runs/classical/params.json"))
    ap.add_argument("--out", type=Path, default=Path("runs/robustness_blur.csv"))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    from baseline_pretrained import load_model, set_threads
    set_threads(args.threads)
    import numpy as np
    import pandas as pd
    import tifffile
    from csbdeep.utils import normalize
    from scipy import ndimage as ndi
    from stardist.models import StarDist2D

    from baseline_classical import segment
    from bioseg.evaluation import flat, image_counts, stems_of, summarize

    params = json.loads(args.classical_params.read_text())
    model = load_model(StarDist2D)
    stems = stems_of(args.data, args.split)
    data = [(tifffile.imread(args.data / "images" / f"{s}.tif").astype(np.float32),
             tifffile.imread(args.data / "labels" / f"{s}.tif")) for s in stems]
    rows = []
    for sigma in args.sigmas:
        counts = {"Otsu + watershed": [], "StarDist 2D_versatile_fluo (sin ajustar)": []}
        for img, gt in data:
            blurred = ndi.gaussian_filter(img, sigma) if sigma > 0 else img
            counts["Otsu + watershed"].append(image_counts(gt, segment(blurred, **params)))
            sd, _ = model.predict_instances(normalize(blurred, 1, 99.8), verbose=False)
            counts["StarDist 2D_versatile_fluo (sin ajustar)"].append(image_counts(gt, sd))
        for method, c in counts.items():
            r = flat(summarize(c))
            rows.append({"method": method, "sigma": sigma, **r})
            print(f"sigma={sigma:<4} {method:42s} AP@0.5={r['all_AP@0.5']:.3f} "
                  f"AP@0.5:0.95={r['all_AP@0.5:0.95']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("Tabla:", args.out)


if __name__ == "__main__":
    main()
