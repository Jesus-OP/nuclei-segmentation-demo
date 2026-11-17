"""Stress test: how much do the methods degrade on synthetically defocused images?

BBBC039 contains no clearly out-of-focus fields (checked visually on the train/val images
with the lowest Laplacian-variance score: they were sharp, the score was driven by
saturated objects and debris). Defocus is a common failure in real screens, so we
simulate it with a Gaussian blur of the raw image (sigma in pixels) and re-run each
method with its frozen settings (no re-tuning). The reference is unchanged.

    python scripts/robustness_blur.py                       # otsu + stardist
    python scripts/robustness_blur.py --methods unet        # adds the U-Net rows

Output: runs/robustness_blur.csv (method, sigma, pooled "all" and "ge25" metrics). Rows of
methods not run this time are kept, so methods can be added one at a time.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

NAMES = {"otsu": "Otsu + watershed",
         "stardist": "StarDist 2D_versatile_fluo (sin ajustar)",
         "unet": "U-Net (entrenada en BBBC039)"}


def make_otsu(args):
    from baseline_classical import segment
    params = json.loads(args.classical_params.read_text())
    return lambda img: segment(img, **params)


def make_stardist(args):
    from baseline_pretrained import load_model, set_threads
    set_threads(args.threads)
    from csbdeep.utils import normalize
    from stardist.models import StarDist2D
    model = load_model(StarDist2D)
    return lambda img: model.predict_instances(normalize(img, 1, 99.8), verbose=False)[0]


def make_unet(args):
    import torch
    from bioseg.unet import UNet, predict_labels
    torch.set_num_threads(args.threads)
    ckpt = torch.load(args.unet_weights, map_location="cpu", weights_only=False)
    model = UNet(base=int(ckpt["args"].get("base", 32)))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return lambda img: predict_labels(model, img, "cpu")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--methods", nargs="+", choices=list(NAMES), default=["otsu", "stardist"])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0, 1, 2, 4])
    ap.add_argument("--classical-params", type=Path, default=Path("runs/classical/params.json"))
    ap.add_argument("--unet-weights", type=Path, default=Path("runs/unet/best.pt"))
    ap.add_argument("--out", type=Path, default=Path("runs/robustness_blur.csv"))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import tifffile
    from scipy import ndimage as ndi

    from bioseg.evaluation import flat, image_counts, stems_of, summarize

    makers = {"otsu": make_otsu, "stardist": make_stardist, "unet": make_unet}
    segmenters = {NAMES[m]: makers[m](args) for m in args.methods}
    stems = stems_of(args.data, args.split)
    data = [(tifffile.imread(args.data / "images" / f"{s}.tif").astype(np.float32),
             tifffile.imread(args.data / "labels" / f"{s}.tif")) for s in stems]
    rows = []
    for sigma in args.sigmas:
        counts = {name: [] for name in segmenters}
        for img, gt in data:
            blurred = ndi.gaussian_filter(img, sigma) if sigma > 0 else img
            for name, seg in segmenters.items():
                counts[name].append(image_counts(gt, seg(blurred)))
        for method, c in counts.items():
            r = flat(summarize(c))
            rows.append({"method": method, "sigma": sigma, **r})
            print(f"sigma={sigma:<4} {method:42s} AP@0.5={r['all_AP@0.5']:.3f} "
                  f"AP@0.5:0.95={r['all_AP@0.5:0.95']:.3f}", flush=True)
    new = pd.DataFrame(rows)
    if args.out.exists():
        old = pd.read_csv(args.out)
        new = pd.concat([old[~old["method"].isin(new["method"])], new], ignore_index=True)
    new.to_csv(args.out, index=False)
    print("Tabla:", args.out)


if __name__ == "__main__":
    main()
