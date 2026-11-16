import numpy as np
from bioseg.metrics import iou_matrix, match, average_precision
from bioseg.measure import per_object_table


def _two_squares():
    m = np.zeros((20, 20), dtype=int)
    m[2:8, 2:8] = 1
    m[12:18, 12:18] = 2
    return m


def test_perfect():
    gt = _two_squares()
    assert match(gt, gt)["f1"] == 1.0
    assert average_precision(gt, gt) == 1.0


def test_label_ids_irrelevant():
    gt = _two_squares()
    pred = gt.copy(); pred[pred == 1] = 7; pred[pred == 2] = 3
    assert match(gt, pred)["tp"] == 2


def test_missed_and_spurious():
    gt = _two_squares()
    pred = np.zeros_like(gt); pred[2:8, 2:8] = 1; pred[0:2, 15:19] = 2
    r = match(gt, pred)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1)
    assert abs(r["ap"] - 1 / 3) < 1e-9


def test_iou_value():
    gt = np.zeros((10, 10), int); gt[0:4, 0:4] = 1
    pred = np.zeros((10, 10), int); pred[0:4, 2:6] = 1
    assert abs(iou_matrix(gt, pred)[0, 0] - 8 / 24) < 1e-9
    assert match(gt, pred, 0.5)["tp"] == 0


def test_empty():
    z = np.zeros((5, 5), int)
    assert match(z, z)["f1"] == 1.0
    assert match(_two_squares(), np.zeros((20, 20), int))["fn"] == 2


def test_table():
    gt = _two_squares()
    t = per_object_table(gt, gt.astype(float) * 10, pixel_size_um=0.5)
    assert list(t["area"]) == [36, 36]
    assert list(t["area_um2"]) == [9.0, 9.0]
