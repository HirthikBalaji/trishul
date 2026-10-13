"""Sentinel-2 multispectral compositing and differential change evidence.

In compliance with Competition Rules §6.3:
Derives vegetation stripping (ΔNDVI), sediment accumulation (ΔBSI), and surface water (MNDWI)
with clear-sky reliability gating. Falls back gracefully to zero reliability under clouds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from trishul.optical.indices import compute_bsi, compute_mndwi, compute_ndvi
from trishul.optical.masks import generate_cloud_mask

logger = logging.getLogger(__name__)


@dataclass
class OpticalEvidenceResult:
    delta_ndvi: np.ndarray
    delta_bsi: np.ndarray
    mndwi_post: np.ndarray
    water_llr: np.ndarray
    debris_llr: np.ndarray
    clear_sky_mask: np.ndarray
    reliability: np.ndarray


class OpticalEvidenceAnalyzer:
    """Analyzes Sentinel-2 pre- and post-flood imagery for physical surface changes."""

    def __init__(
        self,
        ndvi_loss_threshold: float = -0.15,
        bsi_gain_threshold: float = 0.10,
    ):
        self.ndvi_loss_threshold = ndvi_loss_threshold
        self.bsi_gain_threshold = bsi_gain_threshold

    def analyze_change(
        self,
        pre_bands: dict[str, np.ndarray],
        post_bands: dict[str, np.ndarray],
    ) -> OpticalEvidenceResult:
        """Compute differential indices and optical LLRs."""
        # Pre indices
        ndvi_pre = compute_ndvi(pre_bands["B08"], pre_bands["B04"])
        bsi_pre = compute_bsi(pre_bands["B02"], pre_bands["B04"], pre_bands["B08"], pre_bands["B11"])

        # Post indices
        ndvi_post = compute_ndvi(post_bands["B08"], post_bands["B04"])
        bsi_post = compute_bsi(post_bands["B02"], post_bands["B04"], post_bands["B08"], post_bands["B11"])
        mndwi_post = compute_mndwi(post_bands["B03"], post_bands["B11"])

        delta_ndvi = ndvi_post - ndvi_pre
        delta_bsi = bsi_post - bsi_pre

        # Cloud and snow filtering
        scl_post = post_bands.get("SCL", np.full_like(delta_ndvi, 4))  # 4 = vegetation
        cloud_mask, snow_mask = generate_cloud_mask(scl_post, dilate_pixels=5)
        clear_sky_mask = (~cloud_mask) & (~snow_mask)

        # 1. Optical Water Evidence: high MNDWI (> 0.0) and severe NDVI loss
        water_score = np.clip(mndwi_post * 4.0 - delta_ndvi * 2.0, -4.0, 4.0)

        # 2. Debris / Sediment Evidence: severe vegetation loss (ΔNDVI << 0) + sharp BSI gain (ΔBSI >> 0)
        debris_score = np.clip(-delta_ndvi * 3.5 + delta_bsi * 4.0 - 0.5, -4.0, 4.0)

        # Reliability = 1 where clear sky, 0 where obscured
        reliability = np.where(clear_sky_mask, 0.95, 0.0).astype(np.float32)

        # Zero out LLRs where obscured
        water_score[~clear_sky_mask] = 0.0
        debris_score[~clear_sky_mask] = 0.0

        return OpticalEvidenceResult(
            delta_ndvi=delta_ndvi.astype(np.float32),
            delta_bsi=delta_bsi.astype(np.float32),
            mndwi_post=mndwi_post.astype(np.float32),
            water_llr=water_score.astype(np.float32),
            debris_llr=debris_score.astype(np.float32),
            clear_sky_mask=clear_sky_mask,
            reliability=reliability,
        )

    def generate_synthetic_bands(
        self,
        elevation: np.ndarray,
        is_post: bool = False,
    ) -> dict[str, np.ndarray]:
        """Generate physics-consistent synthetic multispectral bands for testing."""
        height, width = elevation.shape
        rng = np.random.default_rng(seed=42 if not is_post else 43)

        # Base mountainous reflectance
        b02 = rng.uniform(0.04, 0.08, size=(height, width)).astype(np.float32)  # Blue
        b03 = rng.uniform(0.06, 0.12, size=(height, width)).astype(np.float32)  # Green
        b04 = rng.uniform(0.05, 0.10, size=(height, width)).astype(np.float32)  # Red
        b08 = rng.uniform(0.25, 0.50, size=(height, width)).astype(np.float32)  # NIR (healthy alpine vegetation)
        b11 = rng.uniform(0.10, 0.20, size=(height, width)).astype(np.float32)  # SWIR
        scl = np.full((height, width), 4, dtype=np.uint8)  # Class 4 = vegetation

        # River channel
        y_indices = np.linspace(0, 1, height)[:, None]
        channel_x = 0.5 + 0.1 * np.sin(y_indices * 3 * np.pi)
        x_indices = np.linspace(0, 1, width)[None, :]
        dist_channel = np.abs(x_indices - channel_x)

        # Permanent water
        perm_water = dist_channel < 0.015
        b08[perm_water] = 0.03
        b11[perm_water] = 0.01
        b03[perm_water] = 0.15
        scl[perm_water] = 6  # Class 6 = water

        if is_post:
            # Flood and debris zone
            flood_water = (dist_channel >= 0.015) & (dist_channel < 0.04) & (elevation < 2200.0)
            debris = (dist_channel >= 0.04) & (dist_channel < 0.07) & (elevation < 2100.0)

            # Inundated: water absorbs NIR and SWIR
            b08[flood_water] = 0.04
            b11[flood_water] = 0.02
            b03[flood_water] = 0.14
            scl[flood_water] = 6

            # Debris deposit: high bare soil and rock reflectance
            b04[debris] = 0.28
            b11[debris] = 0.35
            b08[debris] = 0.12  # stripped vegetation
            b02[debris] = 0.18

        return {
            "B02": b02,
            "B03": b03,
            "B04": b04,
            "B08": b08,
            "B11": b11,
            "SCL": scl,
        }
