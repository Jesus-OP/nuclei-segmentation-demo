import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bioseg.unet import BG, BOUNDARY, INTERIOR, UNet, labels_to_classes, probs_to_labels


def _touching():
    m = np.zeros((32, 32), dtype=np.uint16)
    m[4:20, 4:14] = 1
    m[4:20, 14:24] = 2        # touches object 1
    m[26:29, 26:29] = 3       # 3x3 micro-object
    return m


def test_classes():
    c = labels_to_classes(_touching())
    assert c[0, 0] == BG
    assert c[10, 8] == INTERIOR
    assert c[10, 13] == BOUNDARY and c[10, 14] == BOUNDARY   # 2 px between touching nuclei
    assert c[4, 8] == BOUNDARY                               # 1 px against background


def test_roundtrip_separates_touching_and_keeps_small():
    c = labels_to_classes(_touching())
    probs = np.stack([(c == k).astype(float) for k in (BG, INTERIOR, BOUNDARY)])
    out = probs_to_labels(probs, min_size=5)
    assert out.max() == 3                     # two touching nuclei + orphan micro-object
    left, right = np.unique(out[4:20, 4:14]), np.unique(out[4:20, 14:24])
    assert len(left) == 1 and len(right) == 1 and left[0] != right[0] and 0 not in (left[0], right[0])
    assert out[27, 27] > 0                    # micro-object (all boundary) kept


def test_unet_shape():
    model = UNet(base=4, depth=2).eval()
    assert model(torch.zeros(1, 1, 16, 24)).shape == (1, 3, 16, 24)
