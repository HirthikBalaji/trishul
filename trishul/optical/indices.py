"""Spectral index calculation for Sentinel-2 MSI multispectral imagery.

In compliance with Competition Rules §6.3:
Calculates data-driven physical signatures:
- NDVI: Vegetation loss from scouring / deposition
- MNDWI / NDWI: Surface water ponding
- BSI: Bare sediment and gravel accretion
- NDSI: High-elevation snow/ice gate (prevents snow false alarms)
"""

from __future__ import annotations

import numpy as np


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index: (NIR - Red) / (NIR + Red)."""
    denom = nir + red
    denom = np.where(denom == 0, 1e-6, denom)
    return np.clip((nir - red) / denom, -1.0, 1.0).astype(np.float32)


def compute_mndwi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Modified Normalized Difference Water Index: (Green - SWIR) / (Green + SWIR)."""
    denom = green + swir
    denom = np.where(denom == 0, 1e-6, denom)
    return np.clip((green - swir) / denom, -1.0, 1.0).astype(np.float32)


def compute_bsi(blue: np.ndarray, red: np.ndarray, nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Bare Soil Index: ((SWIR + Red) - (NIR + Blue)) / ((SWIR + Red) + (NIR + Blue))."""
    num = (swir + red) - (nir + blue)
    denom = (swir + red) + (nir + blue)
    denom = np.where(denom == 0, 1e-6, denom)
    return np.clip(num / denom, -1.0, 1.0).astype(np.float32)


def compute_ndsi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Normalized Difference Snow Index: (Green - SWIR) / (Green + SWIR)."""
    denom = green + swir
    denom = np.where(denom == 0, 1e-6, denom)
    return np.clip((green - swir) / denom, -1.0, 1.0).astype(np.float32)
