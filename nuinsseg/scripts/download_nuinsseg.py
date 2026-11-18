"""Download NuInsSeg (H&E nuclei instance segmentation) and build organ-level splits.

Dataset: Mahbod et al., Sci Data 11, 295 (2024). Zenodo 10.5281/zenodo.10518968,
CC BY 4.0 (attribution required, commercial use allowed; see LICENSES.md).

Output layout (under --out, default data/nuinsseg/), same as the BBBC039 demo so that
bioseg.evaluation works unchanged:
    raw/NuInsSeg.zip         original archive (untouched, not extracted)
    images/<stem>.tif        RGB uint8 H&E patches (512x512), from "tissue images/*.png"
    labels/<stem>.tif        uint16 instance labels, from "label masks modify/*.tif"
                             (overlaps resolved to one instance; the authors recommend
                             this folder for evaluation)
    vague/<stem>.tif         uint8 0/1 "ambiguous area" masks, from "vague areas/mask binary"
    splits.csv               stem, organ, species, split, train_rank, n_nuclei, ...

Splits (no official train/test split exists; the authors suggest an image-level 5-fold
KFold, which puts patches of the same slide on both sides). Each organ folder is one
whole-slide image (31 WSIs, 31 organs), so we split BY ORGAN: every patch of an organ
goes to the same split. Per species (human / mouse), organs are visited in a seeded
random order and each goes to the split furthest below its target share of patches
(20 % test, 15 % val, 65 % train). The table is also versioned as nuinsseg/splits.csv.

train_rank: a fixed random permutation (seed) of the train patches. "Train with N images"
means train_rank < N, so the 10-image subset is contained in the 25-image subset, which
is contained in the full train split.
"""
import argparse
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy import ndimage as ndi

URL = "https://zenodo.org/records/10518968/files/NuInsSeg.zip?download=1"
ZIP_SIZE = 1627427342          # bytes, from the Zenodo record (checked 2026-11-18)
SEED = 0
TEST_FRAC, VAL_FRAC = 0.20, 0.15


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size == ZIP_SIZE:
        print(f"  ya existe: {dest}")
        return
    print(f"  descargando {url} (1,6 GB)")
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 22)
    if tmp.stat().st_size != ZIP_SIZE:
        raise IOError(f"Tamaño inesperado: {tmp.stat().st_size} bytes")
    tmp.replace(dest)


def organ_split(table: pd.DataFrame, seed: int = SEED) -> dict:
    """{organ: split}. Per species, organs in a seeded random order; each goes to the split
    whose patch count is furthest below its target (20 % test, 15 % val, 65 % train)."""
    rng = np.random.default_rng(seed)
    fracs = {"test": TEST_FRAC, "val": VAL_FRAC, "train": 1 - TEST_FRAC - VAL_FRAC}
    out = {}
    for species, g in table.groupby("species", sort=True):
        sizes = g.groupby("organ").size().sort_index()
        organs = list(sizes.index[rng.permutation(len(sizes))])
        total, count = sizes.sum(), dict.fromkeys(fracs, 0)
        for organ in organs:
            split = max(fracs, key=lambda s: fracs[s] * total - count[s])
            out[organ], count[split] = split, count[split] + sizes[organ]
    return out


def train_ranks(table: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """Fixed random order of the train patches (nested subsets: first N = 'N images')."""
    train = table.index[table["split"] == "train"].sort_values()
    perm = np.random.default_rng(seed).permutation(len(train))
    rank = pd.Series(pd.NA, index=table.index, dtype="Int64")
    rank[train[perm]] = np.arange(len(train))
    return rank


def n_fragmented(labels: np.ndarray) -> int:
    """Instances made of more than one 8-connected piece (reported, not modified)."""
    objs = ndi.find_objects(labels)
    return sum(1 for i, sl in enumerate(objs, 1)
               if sl is not None and ndi.label(labels[sl] == i, np.ones((3, 3)))[1] > 1)


def convert(zip_path: Path, out: Path) -> pd.DataFrame:
    for d in ("images", "labels", "vague"):
        (out / d).mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        imgs = sorted(n for n in names if "/tissue images/" in n and n.endswith(".png"))
        for n in imgs:
            organ, stem = n.split("/")[0], Path(n).stem
            img = np.asarray(Image.open(io.BytesIO(zf.read(n))).convert("RGB"))
            lab = tifffile.imread(io.BytesIO(zf.read(f"{organ}/label masks modify/{stem}.tif")))
            lab = lab.astype(np.uint16)
            vname = f"{organ}/vague areas/mask binary/{stem}.png"
            vague = (np.asarray(Image.open(io.BytesIO(zf.read(vname)))) > 0) if vname in names \
                else np.zeros(lab.shape, bool)
            if vague.ndim == 3:
                vague = vague.any(-1)
            assert img.shape[:2] == lab.shape == vague.shape, n
            tifffile.imwrite(out / "images" / f"{stem}.tif", img, compression="zlib")
            tifffile.imwrite(out / "labels" / f"{stem}.tif", lab, compression="zlib")
            tifffile.imwrite(out / "vague" / f"{stem}.tif", vague.astype(np.uint8),
                             compression="zlib")
            ids = np.unique(lab[lab > 0])
            rows.append({"stem": stem, "organ": organ, "species": organ.split()[0],
                         "n_nuclei": len(ids), "n_fragmented": n_fragmented(lab),
                         "vague_frac": round(float(vague.mean()), 4)})
    table = pd.DataFrame(rows)
    if table["stem"].duplicated().any():
        raise ValueError("Nombres de imagen repetidos entre órganos")
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data/nuinsseg"))
    ap.add_argument("--repo-splits", type=Path, default=Path("nuinsseg/splits.csv"),
                    help="versioned copy of the split table ('' to skip)")
    args = ap.parse_args()

    (args.out / "raw").mkdir(parents=True, exist_ok=True)
    print("1/3 Descarga")
    download(URL, args.out / "raw" / "NuInsSeg.zip")
    print("2/3 Conversión")
    table = convert(args.out / "raw" / "NuInsSeg.zip", args.out)
    print("3/3 Splits por órgano")
    table["split"] = table["organ"].map(organ_split(table))
    table["train_rank"] = train_ranks(table)
    cols = ["stem", "organ", "species", "split", "train_rank", "n_nuclei", "n_fragmented",
            "vague_frac"]
    table = table[cols]
    table.to_csv(args.out / "splits.csv", index=False)
    if str(args.repo_splits):
        if args.repo_splits.exists():
            old = pd.read_csv(args.repo_splits)
            if not old[["stem", "split", "train_rank"]].equals(
                    pd.read_csv(args.out / "splits.csv")[["stem", "split", "train_rank"]]):
                raise ValueError(f"Los splits no coinciden con {args.repo_splits}")
        else:
            args.repo_splits.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(args.repo_splits, index=False)

    summary = table.groupby(["split", "species"]).agg(
        organs=("organ", "nunique"), images=("stem", "size"), nuclei=("n_nuclei", "sum"))
    print(summary.to_string())
    for split, g in table.groupby("split"):
        print(f"{split}: " + ", ".join(sorted(g["organ"].unique())))
    print(f"Total: {len(table)} imágenes, {table['n_nuclei'].sum()} núcleos; "
          f"instancias fragmentadas (sin modificar): {table['n_fragmented'].sum()}")


if __name__ == "__main__":
    main()
