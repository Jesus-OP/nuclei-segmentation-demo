"""Check which BBBC039 images also appear in DSB 2018 stage1_train (BBBC038), by pixels.

StarDist 2D_versatile_fluo was trained on a subset of DSB 2018 stage1_train. If a BBBC039
test image is in there, StarDist has seen it and its test score is optimistic.

Names differ between datasets and DSB stores 8-bit RGBA PNGs (BBBC039: 16-bit TIFF), so
hashes cannot match. Instead, every DSB image (grey channel) is searched inside every
BBBC039 image with normalised cross-correlation (as skimage.feature.match_template), which
is insensitive to the 16->8 bit intensity rescaling and also finds crops. A pair counts as
the same field if the peak NCC >= --thr (true duplicates give ~1.0; distinct fields of the
same cell line stay far below). Resized or rotated copies would not be found.

    python scripts/download_data.py ...               # BBBC039 in data/
    curl -O https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip
    python scripts/check_dsb_overlap.py --dsb-zip data/external/bbbc038/stage1_train.zip

Output: runs/dsb_overlap.csv (one row per BBBC039 image: best DSB match and its NCC).
The zip is read in place (not extracted).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import io
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.fft as sfft
import tifffile
from PIL import Image

_DSB: dict[str, np.ndarray] = {}


def load_dsb(zip_path: Path) -> dict[str, np.ndarray]:
    out = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if "/images/" in name and name.endswith(".png"):
                im = np.asarray(Image.open(io.BytesIO(zf.read(name))).convert("L"), np.float32)
                if im.std() > 0:
                    out[name.split("/")[0]] = im
    return out


def _init(zip_path: Path) -> None:
    _DSB.update(load_dsb(zip_path))


def window_sum(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Sum of `a` over every valid window of `shape` (summed-area table)."""
    c = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    h, w = shape
    return c[h:, w:] - c[:-h, w:] - c[h:, :-w] + c[:-h, :-w]


def ncc_max(img: np.ndarray, f_img: np.ndarray, stats: dict, tpl: np.ndarray) -> float:
    """Peak normalised cross-correlation of `tpl` over all valid positions in `img`.

    Same quantity as skimage.feature.match_template, but the image FFT and the window
    statistics are computed once per image (and per template shape) instead of per call.
    """
    h, w = tpl.shape
    if (h, w) not in stats:
        n = h * w
        s1 = window_sum(img, (h, w))
        var = window_sum(img ** 2, (h, w)) - s1 ** 2 / n
        stats[(h, w)] = np.sqrt(np.maximum(var, 0))
    t0 = tpl.astype(np.float64) - tpl.mean()
    num = sfft.irfft2(f_img * np.conj(sfft.rfft2(t0, img.shape)), img.shape)
    num = num[: img.shape[0] - h + 1, : img.shape[1] - w + 1]
    den = stats[(h, w)] * np.sqrt((t0 ** 2).sum())
    return float(np.max(np.where(den > 1e-6 * den.max(), num / np.maximum(den, 1e-12), 0)))


def best_match(image_path: Path) -> tuple[str, str, float, int, int]:
    """Return (stem, best DSB id, peak NCC, dsb height, dsb width) for one BBBC039 image."""
    img = tifffile.imread(image_path).astype(np.float64)
    f_img, stats = sfft.rfft2(img), {}
    best = ("", -1.0, 0, 0)
    for dsb_id, tpl in _DSB.items():
        if tpl.shape[0] > img.shape[0] or tpl.shape[1] > img.shape[1]:
            continue
        score = ncc_max(img, f_img, stats, tpl)
        if score > best[1]:
            best = (dsb_id, score, *tpl.shape)
    return (image_path.stem, *best)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--dsb-zip", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--thr", type=float, default=0.95)
    ap.add_argument("--out", type=Path, default=Path("runs/dsb_overlap.csv"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    splits = pd.read_csv(args.data / "splits.csv")
    splits = splits[splits["split"].isin(args.splits)]
    paths = [args.data / "images" / f"{s}.tif" for s in splits["stem"]]
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.dsb_zip,)) as ex:
        rows = list(ex.map(best_match, paths, chunksize=4))
    res = pd.DataFrame(rows, columns=["stem", "dsb_id", "ncc", "dsb_h", "dsb_w"])
    res = splits[["stem", "split"]].merge(res, on="stem")
    res["in_dsb"] = res["ncc"] >= args.thr
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    print(res.groupby("split")["in_dsb"].agg(["sum", "count"]))
    print("NCC por encima/debajo del umbral:",
          res.loc[res.in_dsb, "ncc"].min(), res.loc[~res.in_dsb, "ncc"].max())
    print("Tabla:", args.out)


if __name__ == "__main__":
    main()
