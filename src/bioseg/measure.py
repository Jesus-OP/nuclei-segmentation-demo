"""Cuantificacion por objeto: lo que el cliente realmente quiere (la tabla, no la mascara)."""
from __future__ import annotations

import numpy as np
from skimage.measure import regionprops_table

PROPS = ("label", "area", "centroid", "eccentricity", "solidity",
         "intensity_mean", "intensity_max")


def per_object_table(labels: np.ndarray, intensity: np.ndarray,
                     pixel_size_um: float | None = None) -> dict:
    """Tabla (dict de columnas) con una fila por objeto. Pasar a pandas.DataFrame para CSV."""
    t = regionprops_table(labels, intensity_image=intensity, properties=PROPS)
    if pixel_size_um is not None:
        t["area_um2"] = t["area"] * pixel_size_um ** 2
    return t
