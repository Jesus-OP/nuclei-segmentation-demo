import numpy as np
import pandas as pd
import tifffile

from bioseg.evaluation import drop_small, evaluate_dir, image_counts, object_table, summarize


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


def test_object_table():
    gt = np.zeros((20, 20), dtype=np.uint16)
    gt[0:5, 5:10] = 1          # at the top border
    gt[10:15, 2:6] = 2
    gt[10:15, 6:10] = 3        # touches 2
    pred = np.zeros_like(gt)
    pred[10:15, 2:10] = 7      # 2 and 3 merged -> at most one of them found
    t = object_table(gt, pred).set_index("label")
    assert list(t["touching"]) == [False, True, True]
    assert list(t["at_border"]) == [True, False, False]
    assert not t.loc[1, "found"] and t["found"].sum() <= 1


def test_matched_iou():
    gt = _scene()
    s = summarize([image_counts(gt, gt)])
    assert s["all"]["IoU_matched@0.5"] == 1.0


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


def test_evaluate_dir_exclude(tmp_path):
    data, pred = tmp_path / "data", tmp_path / "pred"
    (data / "labels").mkdir(parents=True)
    pred.mkdir()
    pd.DataFrame({"stem": ["a", "b"], "split": ["test", "test"]}).to_csv(
        data / "splits.csv", index=False)
    for s in "ab":
        tifffile.imwrite(data / "labels" / f"{s}.tif", _scene())
    tifffile.imwrite(pred / "a.tif", _scene())
    tifffile.imwrite(pred / "b.tif", np.zeros((20, 20), np.uint16))  # all missed
    full, _ = evaluate_dir(data, pred, "test", workers=1)
    kept, per_image = evaluate_dir(data, pred, "test", workers=1, exclude={"b"})
    assert full["all"]["AP@0.5"] == 0.5
    assert kept["all"]["AP@0.5"] == 1.0 and list(per_image["stem"]) == ["a"]
