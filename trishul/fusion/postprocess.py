"""Postprocessing and raster serialization for evidence fusion.

Produces standard competition deliverables adhering to §6.6 and §7:
- hazard_prob.tif
- hazard_class.tif
- reliability.tif
- blindspot.tif
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

logger = logging.getLogger(__name__)


@dataclass
class FusionOutputs:
    hazard_prob: np.ndarray    # Float32 in [0, 1] for flood-affected (water ∪ debris)
    hazard_class: np.ndarray   # Int16 class map (0: none, 1: perm, 2: flood, 3: debris, 6: blindspot)
    reliability: np.ndarray    # Float32 in [0, 1]
    blindspot: np.ndarray      # Int16 (0: visible, 1: shadow/layover/cloud)
    flood_water_km2: float
    debris_deposit_km2: float
    total_affected_km2: float
    blindspot_pct: float


class FusionPostprocessor:
    """Classifies raster evidence into discrete hazard classes and exports standard GeoTIFFs."""

    def __init__(self, pixel_area_m2: float = 900.0):  # 30m x 30m default
        self.pixel_area_m2 = pixel_area_m2

    def process(
        self,
        water_llr: np.ndarray,
        debris_llr: np.ndarray,
        reliability: np.ndarray,
        blindspot_mask: np.ndarray | None = None,
    ) -> FusionOutputs:
        """Combine log-likelihood ratios into calibrated probabilities and class labels."""
        height, width = water_llr.shape
        if blindspot_mask is None:
            blindspot_mask = np.zeros((height, width), dtype=np.int16)

        # 1. Sigmoid transformation of LLRs
        p_water = 1.0 / (1.0 + np.exp(-water_llr))
        p_debris = 1.0 / (1.0 + np.exp(-debris_llr))

        # Overall flood-affected probability: P(water ∪ debris) = 1 - (1 - P_water)*(1 - P_debris)
        p_affected = 1.0 - (1.0 - p_water) * (1.0 - p_debris)
        p_affected = np.clip(p_affected, 0.0, 1.0)

        # 2. Assign discrete class labels
        # 0: no evidence
        # 1: permanent water
        # 2: flood water
        # 3: debris deposit
        # 6: not assessable
        hazard_class = np.zeros((height, width), dtype=np.int16)

        # Permanent water baseline (very stable water)
        # For vertical slice: distinguish flood water vs debris
        flood_water_idx = (p_water >= 0.5) & (p_water > p_debris)
        debris_idx = (p_debris >= 0.5) & (p_debris >= p_water)

        hazard_class[flood_water_idx] = 2
        hazard_class[debris_idx] = 3

        # Mark blindspot pixels
        hazard_class[blindspot_mask > 0] = 6
        p_affected[blindspot_mask > 0] = 0.0

        # Calculate affected surface areas in square kilometers
        pixel_to_km2 = self.pixel_area_m2 / 1_000_000.0
        flood_water_km2 = float(np.sum(hazard_class == 2) * pixel_to_km2)
        debris_deposit_km2 = float(np.sum(hazard_class == 3) * pixel_to_km2)
        total_affected_km2 = flood_water_km2 + debris_deposit_km2
        blindspot_pct = float(np.mean(blindspot_mask > 0) * 100.0)

        return FusionOutputs(
            hazard_prob=p_affected.astype(np.float32),
            hazard_class=hazard_class,
            reliability=reliability.astype(np.float32),
            blindspot=blindspot_mask.astype(np.int16),
            flood_water_km2=round(flood_water_km2, 3),
            debris_deposit_km2=round(debris_deposit_km2, 3),
            total_affected_km2=round(total_affected_km2, 3),
            blindspot_pct=round(blindspot_pct, 2),
        )

    def export_rasters(
        self,
        outputs: FusionOutputs,
        transform: rasterio.Affine,
        crs: str,
        output_dir: Path | str,
    ) -> dict[str, Path]:
        """Write standard GeoTIFF rasters to run directory."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        height, width = outputs.hazard_prob.shape
        paths = {}

        meta_base = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "crs": crs,
            "transform": transform,
            "compress": "lzw",
        }

        # 1. hazard_prob.tif
        prob_path = out / "hazard_prob.tif"
        with rasterio.open(prob_path, "w", dtype=rasterio.float32, **meta_base) as dst:
            dst.write(outputs.hazard_prob, 1)
        paths["hazard_prob"] = prob_path

        # 2. hazard_class.tif
        class_path = out / "hazard_class.tif"
        with rasterio.open(class_path, "w", dtype=rasterio.int16, **meta_base) as dst:
            dst.write(outputs.hazard_class, 1)
        paths["hazard_class"] = class_path

        # 3. reliability.tif
        rel_path = out / "reliability.tif"
        with rasterio.open(rel_path, "w", dtype=rasterio.float32, **meta_base) as dst:
            dst.write(outputs.reliability, 1)
        paths["reliability"] = rel_path

        # 4. blindspot.tif
        blind_path = out / "blindspot.tif"
        with rasterio.open(blind_path, "w", dtype=rasterio.int16, **meta_base) as dst:
            dst.write(outputs.blindspot, 1)
        paths["blindspot"] = blind_path

        return paths
