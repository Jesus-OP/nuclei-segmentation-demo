"""Download BBBC039v1 (images, masks, official splits) and convert masks to label images.

Dataset: https://bbbc.broadinstitute.org/BBBC039  (CC0 1.0, see LICENSES.md)

Output layout (under --out, default data/):
    raw/                     downloaded zips + extracted contents (untouched)
    images/<stem>.tif        16-bit DNA (Hoechst) images, copied as-is
    labels/<stem>.tif        uint16 instance labels (0 = background, 1..N = nuclei)
    splits.csv               stem,split  with split in {train, val, test} (official)

Mask format: the ground truth is an RGBA PNG where each nucleus is a flat region of one
color, and touching nuclei get different colors (in practice only the R channel is used,
with values 1, 2, occasionally 3; G=B=0, A=255). We label connected components of
*identical color* (4-connectivity): two adjacent nuclei never merge, and two separate
nuclei that happen to share a color still get different labels.

Small objects (< 25 px) are KEPT, on purpose. There are 894 of them (3.8 % of 23,617
objects; median nucleus area is ~625 px). Reviewed on train/val crops (2026-11-16):
- Almost all are small, deliberately annotated DNA bodies (micronuclei, apoptotic
  fragments, faint dots next to nuclei), drawn with a round brush. That is why their
  areas cluster at 12/16/21 px (4x4 and 5x5 stamps). Their mean intensity is 1.5-4x the
  background median (5th-95th percentile), so they contain real signal.
- Only ~2-3 are real annotation leftovers (e.g. a 5x1 px sliver on background, intensity
  ~1.0x background). That is too few to justify an extra cleaning rule that would edit
  the ground truth.
Filtering by area would delete real annotations and inflate every method's recall. The
eval should report metrics with all objects and also restricted to objects >= 25 px.
The "small" column in the summary counts them per split.
"""
import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage import io, measure

BASE_URL = "https://data.broadinstitute.org/bbbc/BBBC039/"
ARCHIVES = ["images.zip", "masks.zip", "metadata.zip"]
SPLIT_FILES = {"training.txt": "train", "validation.txt": "val", "test.txt": "test"}
SMALL_AREA = 25  # px; objects below this are kept but counted (see module docstring)


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ya existe: {dest.name}")
        return
    print(f"  descargando {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)


def safe_extract(zip_path: Path, dest: Path) -> None:
    """Extract a zip, refusing entries that would escape `dest`."""
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = (dest / member).resolve()
            if not target.is_relative_to(dest):
                raise ValueError(f"Ruta insegura en {zip_path.name}: {member}")
        zf.extractall(dest)


def rgb_mask_to_labels(mask: np.ndarray) -> np.ndarray:
    """RGB(A) instance mask -> uint16 label image (connected components of equal color)."""
    if mask.ndim == 2:
        code = mask.astype(np.int64)
    else:
        rgb = mask[..., :3].astype(np.int64)
        code = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    # measure.label joins neighbouring pixels only if they share the same value.
    labels = measure.label(code, background=0, connectivity=1)
    if labels.max() > np.iinfo(np.uint16).max:
        raise ValueError("Demasiados objetos para uint16")
    return labels.astype(np.uint16)


def read_splits(meta_dir: Path) -> pd.DataFrame:
    rows = []
    for fname, split in SPLIT_FILES.items():
        path = next(meta_dir.rglob(fname))
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append({"stem": Path(line.strip()).stem, "split": split})
    df = pd.DataFrame(rows)
    if df["stem"].duplicated().any():
        raise ValueError("Una imagen aparece en más de un split")
    return df


def find_one(root: Path, stem: str, suffix: str) -> Path:
    matches = list(root.rglob(stem + suffix))
    if len(matches) != 1:
        raise FileNotFoundError(f"{stem}{suffix}: {len(matches)} coincidencias en {root}")
    return matches[0]


def convert(raw: Path, out: Path, splits: pd.DataFrame) -> pd.DataFrame:
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    n_objects, n_small = [], []
    for stem in splits["stem"]:
        img_src = find_one(raw / "images", stem, ".tif")
        shutil.copy2(img_src, out / "images" / f"{stem}.tif")
        labels = rgb_mask_to_labels(io.imread(find_one(raw / "masks", stem, ".png")))
        tifffile.imwrite(out / "labels" / f"{stem}.tif", labels, compression="zlib")
        areas = np.bincount(labels.ravel())[1:]
        n_objects.append(int(labels.max()))
        n_small.append(int((areas < SMALL_AREA).sum()))
    return splits.assign(n_nuclei=n_objects, n_small=n_small)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()

    raw = args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    print("1/3 Descarga")
    for name in ARCHIVES:
        download(BASE_URL + name, raw / name)
        safe_extract(raw / name, raw / Path(name).stem)

    print("2/3 Splits oficiales")
    splits = read_splits(raw / "metadata")

    print("3/3 Conversión de máscaras a etiquetas")
    table = convert(raw, args.out, splits)
    table.to_csv(args.out / "splits.csv", index=False)

    summary = table.groupby("split", sort=False).agg(
        images=("stem", "size"), nuclei=("n_nuclei", "sum"), small=("n_small", "sum"))
    print(summary.to_string())
    print(f"Total: {len(table)} imágenes, {table['n_nuclei'].sum()} núcleos "
          f"({table['n_small'].sum()} con < {SMALL_AREA} px, conservados)")


if __name__ == "__main__":
    main()
