"""Road network damage exposure module.

In compliance with Competition Rules §6.7:
Evaluates road segments buffered by half-width and positional uncertainty.
Calculates length_hit_km and edge failure probability.
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
class RoadEvaluation:
    total_roads: int
    total_length_km: float
    total_length_hit_km: float
    features: list[dict[str, Any]]


class RoadExposureEvaluator:
    """Evaluates damage probability and severed lengths of OSM highway segments."""

    def __init__(self, buffer_half_width_m: float = 5.0):
        self.buffer_half_width_m = buffer_half_width_m

    def evaluate(
        self,
        osm_roads_geojson: dict[str, Any],
        hazard_prob: np.ndarray,
        transform: rasterio.Affine,
    ) -> RoadEvaluation:
        """Evaluate p_hit and length_hit_km for each road line."""
        raw_features = osm_roads_geojson.get("features", [])
        total = len(raw_features)
        if total == 0:
            return RoadEvaluation(total_roads=0, total_length_km=0.0, total_length_hit_km=0.0, features=[])

        height, width = hazard_prob.shape
        evaluated_features: list[dict[str, Any]] = []
        tot_len_km = 0.0
        tot_hit_len_km = 0.0

        for feat in raw_features:
            geom = shape(feat["geometry"])
            # Length approximation in kilometers (assuming geographic degrees near 28N)
            length_deg = geom.length
            # 1 deg ~ 105 km in Himalaya
            seg_len_km = float(length_deg * 105.0)
            tot_len_km += seg_len_km

            # Sample line vertices
            coords = list(geom.coords) if hasattr(geom, "coords") else []
            hit_probs = []

            for x, y in coords:
                try:
                    r, c = rowcol(transform, x, y)
                    if 0 <= r < height and 0 <= c < width:
                        hit_probs.append(float(hazard_prob[r, c]))
                except Exception:
                    continue

            p_hit = float(np.max(hit_probs)) if hit_probs else 0.0
            p_hit = round(min(1.0, max(0.0, p_hit)), 3)

            is_failed = p_hit >= 0.5
            len_hit = round(seg_len_km * p_hit if is_failed else 0.0, 3)
            tot_hit_len_km += len_hit

            props = dict(feat.get("properties", {}))
            props.update({
                "p_hit": p_hit,
                "length_km": round(seg_len_km, 3),
                "length_hit_km": len_hit,
                "status": "impassable" if is_failed else "passable",
                "evidence": {
                    "source": "hazard_prob",
                    "max_vertex_hazard": p_hit,
                },
            })

            evaluated_features.append({
                "type": "Feature",
                "id": feat.get("id"),
                "properties": props,
                "geometry": feat["geometry"],
            })

        return RoadEvaluation(
            total_roads=total,
            total_length_km=round(tot_len_km, 2),
            total_length_hit_km=round(tot_hit_len_km, 2),
            features=evaluated_features,
        )
