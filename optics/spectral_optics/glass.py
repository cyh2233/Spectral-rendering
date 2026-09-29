"""Refractive-index models shared with the renderer (data/glass/catalog.json)."""
from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np

_DEFAULT_CATALOG = os.path.join(os.path.dirname(__file__), "..", "..", "data", "glass", "catalog.json")


def catalog_path() -> str:
    env = os.environ.get("SPECTRAL_DATA_DIR")
    if env:
        return os.path.join(env, "glass", "catalog.json")
    return os.path.normpath(_DEFAULT_CATALOG)


@lru_cache(maxsize=None)
def _catalog(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def ior(name: str, wavelength_nm, path: str | None = None):
    """Refractive index of catalog glass `name` at wavelength(s) in nm. 'air' -> 1."""
    lam = np.asarray(wavelength_nm, dtype=np.float64)
    if name == "air":
        return np.ones_like(lam)
    g = _catalog(path or catalog_path())[name]
    l2 = (lam * 1e-3) ** 2
    if g["model"] == "sellmeier":
        n2 = 1.0 + sum(B * l2 / (l2 - C) for B, C in zip(g["B"], g["C"]))
        return np.sqrt(n2)
    if g["model"] == "power_series":
        l = lam * 1e-3
        return sum(c * l ** p for c, p in g["coefficients"])
    raise ValueError(f"unknown glass model {g['model']}")
