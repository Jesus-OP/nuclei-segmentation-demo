"""3-class U-Net (background / interior / boundary) for nucleus instance segmentation.

Instances are recovered by watershed: seeds are connected components of the "interior"
class, flooded over the foreground (interior + boundary). Foreground blobs without any
interior seed (typically micro-objects only a few pixels wide, which are all "boundary")
are kept as their own instances so small nuclei are not silently dropped.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import measure, segmentation

BG, INTERIOR, BOUNDARY = 0, 1, 2


def labels_to_classes(labels: np.ndarray) -> np.ndarray:
    """Instance labels -> uint8 class map. Boundary = 'thick' boundaries inside objects,
    i.e. 1 px against background and 2 px (one per side) between touching nuclei."""
    classes = (labels > 0).astype(np.uint8) * INTERIOR
    classes[segmentation.find_boundaries(labels, mode="thick") & (labels > 0)] = BOUNDARY
    return classes


def probs_to_labels(probs: np.ndarray, interior_thr: float = 0.5,
                    min_size: int = 5) -> np.ndarray:
    """probs: (3, H, W) softmax output -> uint16 instance labels."""
    fg = probs.argmax(0) != BG
    seeds = measure.label((probs[INTERIOR] > interior_thr) & fg)
    labels = segmentation.watershed(-probs[INTERIOR], seeds, mask=fg)
    orphans = measure.label(fg & (labels == 0))
    labels[orphans > 0] = orphans[orphans > 0] + labels.max()
    areas = np.bincount(labels.ravel())
    labels[areas[labels] < min_size] = 0
    return segmentation.relabel_sequential(labels)[0].astype(np.uint16)


def normalize(img: np.ndarray, lo: float = 1, hi: float = 99.8) -> np.ndarray:
    """Per-image percentile normalization (same convention as StarDist)."""
    a, b = np.percentile(img, [lo, hi])
    return ((img.astype(np.float32) - a) / max(b - a, 1e-6)).astype(np.float32)


class _Block(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True))


class UNet(nn.Module):
    """Plain U-Net. Input H, W must be divisible by 2**depth (predict_image pads)."""

    def __init__(self, in_ch: int = 1, n_classes: int = 3, base: int = 32, depth: int = 4):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList([_Block(in_ch, chs[0])] +
                                  [_Block(chs[i], chs[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2)
                                 for i in reversed(range(depth))])
        self.dec = nn.ModuleList([_Block(2 * chs[i], chs[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(chs[0], n_classes, 1)
        self.depth = depth

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 1))
        return self.head(x)


@torch.no_grad()
def predict_probs(model: UNet, img: np.ndarray, device) -> np.ndarray:
    """Full-image softmax probabilities (3, H, W); reflect-pads to a multiple of 2**depth."""
    m = 2 ** model.depth
    h, w = img.shape
    ph, pw = (-h) % m, (-w) % m
    x = np.pad(normalize(img), ((0, ph), (0, pw)), mode="reflect")
    x = torch.from_numpy(x)[None, None].to(device)
    probs = torch.softmax(model(x), 1)[0, :, :h, :w]
    return probs.cpu().numpy()


def predict_labels(model: UNet, img: np.ndarray, device, **kw) -> np.ndarray:
    return probs_to_labels(predict_probs(model, img, device), **kw)
