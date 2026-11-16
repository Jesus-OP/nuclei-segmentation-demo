import numpy as np
import pandas as pd
import tifffile

from bioseg.evaluation import drop_small, evaluate_dir, image_counts, summarize


def _scene():
    """One 36 px nucleus and one 4 px micro-object."""
    m = np.zeros((20, 20), dtype=np.uint16)
    m[2:8, 2:8] = 1
    m[15:17, 15:17] = 2
    return m


def test_drop_small():
    out = drop_small(_scene(), 25)
    assert set(np.unique(out)) == {0, 1}


def test_ge25_filters_both_sides():
    gt = _scene()
    pred = np.zeros_like(gt)
    pred[2:8, 2:8] = 5          # matches the nucleus
    pred[0:3, 15:18] = 9        # small (9 px) spurious detection
    c = image_counts(gt, pred)
    assert c[("all", 0.5)] == (1, 1, 1)    # micro-object missed, small FP counted
    assert c[("ge25", 0.5)] == (1, 0, 0)   # both small objects ignored


def test_summarize_pools_counts():
    a = {(v, t): (1, 0, 0) for v in ("all", "ge25") for t in np.round(np.arange(0.5, 1, 0.05), 2)}
    b = {(v, t): (0, 1, 1) for v, t in a}
    s = summarize([a, b])
    assert abs(s["all"]["AP@0.5"] - 1 / 3) < 1e-9     # pooled 1/(1+1+1), not mean(1, 0)
    assert abs(s["all"]["F1@0.5"] - 0.5) < 1e-9


def test_evaluate_dir(tmp_path):
    data, pred = tmp_path / "data", tmp_path / "pred"
    (data / "labels").mkdir(parents=True)
    pred.mkdir()
    pd.DataFrame({"stem": ["a"], "split": ["test"]}).to_csv(data / "splits.csv", index=False)
    tifffile.imwrite(data / "labels" / "a.tif", _scene())
    tifffile.imwrite(pred / "a.tif", _scene())
    summary, per_image = evaluate_dir(data, pred, "test", workers=1)
    assert summary["all"]["AP@0.5:0.95"] == 1.0
    assert len(per_image) == 1
