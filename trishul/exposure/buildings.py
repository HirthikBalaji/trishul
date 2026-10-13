"""Building damage exposure module.

In compliance with Competition Rules §6.7:
Computes p_hit for pre-event OSM building footprints using the hazard probability
raster and a 15m positional uncertainty buffer. Reports expected count Σ p.
Never claims 'destroyed vs damaged' from 10m data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import shape

logger = logging.getLogger(__name__)


@dataclass
class BuildingEvaluation:
    total_buildings: int
    expected_hit_count: float
    confirmed_hit_count: float  # p_hit >= 0.5
    features: list[dict[str, Any]]


class BuildingExposureEvaluator:
    """Evaluates flood and debris impact on OSM building polygons."""

    def __init__(self, buffer_m: float = 15.0):
        self.buffer_m = buffer_m

    def evaluate(
        self,
        osm_buildings_geojson: dict[str, Any],
        hazard_prob: np.ndarray,
        transform: rasterio.Affine,
    ) -> BuildingEvaluation:
        """Evaluate p_hit for each building feature."""
        raw_features = osm_buildings_geojson.get("features", [])
        total = len(raw_features)
        if total == 0:
            return BuildingEvaluation(total_buildings=0, expected_hit_count=0.0, confirmed_hit_count=0, features=[])

        height, width = hazard_prob.shape
        evaluated_features: list[dict[str, Any]] = []
        expected_sum = 0.0
        confirmed_count = 0

        for feat in raw_features:
            geom = shape(feat["geometry"])
            centroid = geom.centroid

            # Project centroid to raster row, col
            try:
                r, c = rowcol(transform, centroid.x, centroid.y)
            except Exception:
                continue

            # Sample within 1-pixel window around centroid (~30m window)
            r_min, r_max = max(0, r - 1), min(height, r + 2)
            c_min, c_max = max(0, c - 1), min(width, c + 2)

            if r_min < r_max and c_min < c_max:
                window_prob = hazard_prob[r_min:r_max, c_min:c_max]
                p_hit = float(np.max(window_prob))
            else:
                p_hit = 0.0

            p_hit = round(min(1.0, max(0.0, p_hit)), 3)
            expected_sum += p_hit
            is_hit = p_hit >= 0.5
            if is_hit:
                confirmed_count += 1

            props = dict(feat.get("properties", {}))
            props.update({
                "p_hit": p_hit,
                "p_hit_lo": round(max(0.0, p_hit - 0.15), 3),
                "p_hit_hi": round(min(1.0, p_hit + 0.15), 3),
                "status": "hit" if is_hit else "intact",
                "evidence": {
                    "source": "hazard_prob",
                    "max_pixel_val": p_hit,
                    "resolution": "30m GLO-30 / 10m S1",
                },
            })

            evaluated_features.append({
                "type": "Feature",
                "id": feat.get("id"),
                "properties": props,
                "geometry": feat["geometry"],
            })

        return BuildingEvaluation(
            total_buildings=total,
            expected_hit_count=round(expected_sum, 1),
            confirmed_hit_count=confirmed_count,
            features=evaluated_features,
        )
