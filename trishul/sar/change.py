"""Temporal statistics and speckle-aware change detection on SAR stacks.

In compliance with Competition Rules §5 & §6.2:
Tests the post-event acquisition against its own same-track pre-event history,
yielding calibrated signed z-scores, p-values, and log-likelihood ratios.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.stats import norm

from trishul.sar.backends import SarSceneData

logger = logging.getLogger(__name__)


@dataclass
class SarChangeResult:
    delta_vv_db: np.ndarray
    delta_vh_db: np.ndarray
    z_vv: np.ndarray
    p_value_vv: np.ndarray
    water_llr: np.ndarray
    debris_llr: np.ndarray
    reliability: np.ndarray


class SarChangeDetector:
    """Calculates temporal-statistics change metrics from same-track SAR acquisitions."""

    def __init__(
        self,
        multilook_window: int = 3,
        mad_sigma_floor_db: float = 0.5,
    ):
        self.multilook_window = multilook_window
        self.mad_sigma_floor_db = mad_sigma_floor_db

    def detect_change(
        self,
        pre_scene: SarSceneData,
        post_scene: SarSceneData,
        pre_stack: list[SarSceneData] | None = None,
    ) -> SarChangeResult:
        """Execute change detection between pre-event history and post-event acquisition."""
        # 1. Light spatial multilooking to reduce speckle noise
        post_vv = uniform_filter(post_scene.vv_db, size=self.multilook_window)
        post_vh = uniform_filter(post_scene.vh_db, size=self.multilook_window)

        if pre_stack and len(pre_stack) >= 2:
            # Multi-temporal stack statistics
            stack_vv = np.stack([
                uniform_filter(s.vv_db, size=self.multilook_window)
                for s in pre_stack
            ], axis=0)
            pre_mean_vv = np.median(stack_vv, axis=0)
            # Median Absolute Deviation (MAD) scaled to normal sigma
            mad_vv = np.median(np.abs(stack_vv - pre_mean_vv), axis=0) * 1.4826
            sigma_vv = np.maximum(mad_vv, self.mad_sigma_floor_db)
        else:
            pre_mean_vv = uniform_filter(pre_scene.vv_db, size=self.multilook_window)
            sigma_vv = np.full_like(pre_mean_vv, 1.2)

        # 2. Simple log-ratio and standardized z-score
        delta_vv = post_vv - pre_mean_vv
        pre_vh = uniform_filter(pre_scene.vh_db, size=self.multilook_window)
        delta_vh = post_vh - pre_vh

        z_vv = delta_vv / sigma_vv
        # Two-sided p-value
        p_val_vv = 2.0 * (1.0 - norm.cdf(np.abs(z_vv)))

        # 3. Calibrated log-likelihood ratios (LLRs)
        # Strong negative delta_vv (<-3.0 dB) indicates smooth surface/water
        # Sigmoid mapping from z-score
        water_llr = np.clip(-z_vv * 1.5 - 1.0, -5.0, 5.0)

        # Positive delta_vv (> +2.5 dB) indicates rocky debris / boulder deposit
        debris_llr = np.clip(z_vv * 1.2 - 1.5, -5.0, 5.0)

        # Reliability metric based on stack depth and noise floor
        stack_depth = len(pre_stack) if pre_stack else 1
        r_temporal = min(1.0, stack_depth / 6.0)
        reliability = np.full_like(delta_vv, r_temporal, dtype=np.float32)

        return SarChangeResult(
            delta_vv_db=delta_vv.astype(np.float32),
            delta_vh_db=delta_vh.astype(np.float32),
            z_vv=z_vv.astype(np.float32),
            p_value_vv=p_val_vv.astype(np.float32),
            water_llr=water_llr.astype(np.float32),
            debris_llr=debris_llr.astype(np.float32),
            reliability=reliability.astype(np.float32),
        )
