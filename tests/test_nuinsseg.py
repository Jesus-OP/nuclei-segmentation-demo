"""Tests for the pieces added for the NuInsSeg demo (ignore masks, RGB U-Net, splits)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bioseg.evaluation import drop_in_mask

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nuinsseg" / "scripts"))


def test_drop_in_mask():
    lab = np.zeros((10, 10), np.uint16)
    lab[0:4, 0:4] = 1          # fully inside the mask
    lab[0:4, 4:8] = 2          # 1/4 inside
    lab[6:9, 6:9] = 3          # outside
    mask = np.zeros_like(lab, dtype=np.uint8)
    mask[0:4, 0:5] = 1
    out = drop_in_mask(lab, mask)
    assert set(np.unique(out)) == {0, 2, 3}
    assert (drop_in_mask(lab, np.zeros_like(mask)) == lab).all()


def test_organ_split_and_nested_ranks():
    dl = pytest.importorskip("download_nuinsseg")
    rows = [{"stem": f"{sp}_{o}_{i}", "organ": f"{sp} {o}", "species": sp}
            for sp, n_org in (("human", 12), ("mouse", 5)) for o in range(n_org)
            for i in range(10)]
    t = pd.DataFrame(rows)
    t["split"] = t["organ"].map(dl.organ_split(t))
    assert (t.groupby("organ")["split"].nunique() == 1).all()       # no organ leaks
    for sp, g in t.groupby("species"):
        assert set(g["split"]) == {"train", "val", "test"}
        assert abs((g["split"] == "test").mean() - dl.TEST_FRAC) <= 0.1
    t["train_rank"] = dl.train_ranks(t)
    tr = t[t["split"] == "train"]
    assert sorted(tr["train_rank"]) == list(range(len(tr)))
    assert t.loc[t["split"] != "train", "train_rank"].isna().all()
    assert t["split"].equals(t["organ"].map(dl.organ_split(t)))        # deterministic


def test_rgb_prediction_shape():
    pytest.importorskip("torch")
    from bioseg.unet import UNet, predict_probs
    model = UNet(in_ch=3, base=4, depth=2).eval()
    img = np.random.default_rng(0).integers(0, 255, (20, 30, 3), dtype=np.uint8)
    assert predict_probs(model, img, "cpu").shape == (3, 20, 30)
    gray = UNet(base=4, depth=2).eval()
    assert predict_probs(gray, img[..., 0], "cpu").shape == (3, 20, 30)
