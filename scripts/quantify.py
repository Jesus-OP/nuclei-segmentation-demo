"""Per-nucleus measurements from a folder of label images -> one CSV per image.

Uses bioseg.measure.per_object_table (area, centroid, eccentricity, solidity, mean / max
intensity on the raw 16-bit image) and adds `touches_border`: nuclei cut by the image
edge have truncated area and shape, so they are flagged (not removed) and the summary
statistics exclude them.

BBBC039 does not state a pixel size on its page, so areas are in pixels unless
--pixel-size-um is given (then `area_um2` is added).

Outputs (in --out, default runs/quant/<name>/):
    <stem>.csv     one row per nucleus
    summary.csv    one row per image: count, count excluding border, median area, ...
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from bioseg.evaluation import stems_of
from bioseg.measure import per_object_table


def border_labels(labels: np.ndarray) -> np.ndarray:
    edge = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    return np.unique(edge[edge > 0])


def quantify_image(img: np.ndarray, labels: np.ndarray, pixel_size_um) -> pd.DataFrame:
    t = pd.DataFrame(per_object_table(labels, img, pixel_size_um=pixel_size_um))
    t = t.rename(columns={"centroid-0": "centroid_y", "centroid-1": "centroid_x"})
    t["touches_border"] = t["label"].isin(border_labels(labels))
    return t


def summarize_image(stem: str, t: pd.DataFrame) -> dict:
    inner = t[~t["touches_border"]]
    return {"stem": stem, "n_nuclei": len(t), "n_inner": len(inner),
            "median_area_px": inner["area"].median(),
            "median_intensity": inner["intensity_mean"].median(),
            "median_eccentricity": inner["eccentricity"].median()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--labels", type=Path, default=Path("runs/stardist/pred/test"),
                    help="folder of <stem>.tif label images (any method)")
    ap.add_argument("--split", default="test", help="which stems to process")
    ap.add_argument("--out", type=Path, default=None, help="default runs/quant/<labels parent>")
    ap.add_argument("--pixel-size-um", type=float, default=None)
    args = ap.parse_args()

    out = args.out or Path("runs/quant") / args.labels.parent.parent.name
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for stem in stems_of(args.data, args.split):
        img = tifffile.imread(args.data / "images" / f"{stem}.tif")
        labels = tifffile.imread(args.labels / f"{stem}.tif")
        t = quantify_image(img, labels, args.pixel_size_um)
        t.to_csv(out / f"{stem}.csv", index=False)
        summary.append(summarize_image(stem, t))
    s = pd.DataFrame(summary)
    s.to_csv(out / "summary.csv", index=False)
    print(f"{len(s)} imágenes, {s['n_nuclei'].sum()} núcleos "
          f"({s['n_nuclei'].sum() - s['n_inner'].sum()} tocan el borde) -> {out}")


if __name__ == "__main__":
    main()
