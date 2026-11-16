"""Run a trained U-Net checkpoint (from scripts/train.py) on a split and write label TIFFs.

CPU by default (local GPU is busy). Example, after downloading best.pt from Kaggle:
    python scripts/predict_unet.py --weights runs/unet/best.pt --split test
    python scripts/evaluate.py      # picks up runs/unet/pred/test automatically
"""
import argparse
from pathlib import Path

import tifffile
import torch

from bioseg.evaluation import stems_of
from bioseg.unet import UNet, predict_labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <weights folder>/pred/<split>")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None, help="only N images (smoke test)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    ckpt = torch.load(args.weights, map_location=args.device, weights_only=False)
    model = UNet(base=int(ckpt["args"].get("base", 32))).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    out = args.out or args.weights.parent / "pred" / args.split
    out.mkdir(parents=True, exist_ok=True)
    stems = stems_of(args.data, args.split)[:args.limit]
    for stem in stems:
        img = tifffile.imread(args.data / "images" / f"{stem}.tif")
        labels = predict_labels(model, img, args.device)
        tifffile.imwrite(out / f"{stem}.tif", labels, compression="zlib")
    print(f"{len(stems)} predicciones en {out} (checkpoint de la época {ckpt['epoch']})")


if __name__ == "__main__":
    main()
