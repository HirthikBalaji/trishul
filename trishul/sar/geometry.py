"""SAR terrain-geometry gating and ray-marched blindspot mapping.

In compliance with Competition Rules §5 Innovation #1 and §6.1:
Calculates local incidence angle, layover mask, ray-marched radar shadow mask,
and per-track reliability r_geo. Prevents topographic false positives in steep Himalayan valleys.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GeometryGateOutputs:
    local_incidence_deg: np.ndarray
    layover_mask: np.ndarray      # True where slope > complement of incidence angle
    shadow_mask: np.ndarray       # True where terrain casts radar shadow
    blindspot_mask: np.ndarray    # Layover | Shadow | Extreme incidence
    r_geo: np.ndarray             # Geometric reliability factor in [0, 1]
    blindspot_reason: np.ndarray  # 0: visible, 1: shadow, 2: layover, 3: extreme incidence


class SarGeometryGate:
    """Computes radar geometric distortion masks and reliability weights."""

    def __init__(
        self,
        min_incidence_deg: float = 20.0,
        max_incidence_deg: float = 50.0,
        nominal_look_angle_deg: float = 38.0,
    ):
        self.min_incidence_deg = min_incidence_deg
        self.max_incidence_deg = max_incidence_deg
        self.nominal_look_angle_deg = nominal_look_angle_deg

    def compute_geometry(
        self,
        elevation: np.ndarray,
        resolution_m: float = 30.0,
        orbit_direction: str = "DESCENDING",
    ) -> GeometryGateOutputs:
        """Compute ray-marched geometric reliability and blindspot mask."""
        height, width = elevation.shape

        # 1. Compute surface gradients and terrain normal
        # dy: North-South gradient, dx: West-East gradient
        grad_y, grad_x = np.gradient(elevation, resolution_m)
        slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
        slope_deg = np.degrees(slope_rad)

        # 2. Sensor line-of-sight vector
        # Sentinel-1 is right-looking.
        # Descending pass (heading ~190° S): illuminates from East (azimuth ~100°)
        # Ascending pass (heading ~350° N): illuminates from West (azimuth ~280°)
        look_azimuth_deg = 100.0 if orbit_direction.upper() == "DESCENDING" else 280.0
        look_azimuth_rad = np.radians(look_azimuth_deg)
        look_inc_rad = np.radians(self.nominal_look_angle_deg)

        # Dot product between surface normal and incident ray
        # Normal vector: [-grad_x, -grad_y, 1] / sqrt(...)
        # Ray vector: [sin(inc)*cos(az), sin(inc)*sin(az), cos(inc)]
        norm_len = np.sqrt(grad_x**2 + grad_y**2 + 1.0)
        nx = -grad_x / norm_len
        ny = -grad_y / norm_len
        nz = 1.0 / norm_len

        rx = np.sin(look_inc_rad) * np.sin(look_azimuth_rad)
        ry = np.sin(look_inc_rad) * np.cos(look_azimuth_rad)
        rz = np.cos(look_inc_rad)

        cos_loc_inc = nx * rx + ny * ry + nz * rz
        cos_loc_inc = np.clip(cos_loc_inc, -1.0, 1.0)
        local_inc_deg = np.degrees(np.arccos(cos_loc_inc))

        # 3. Layover: slope facing sensor steeper than complement of incidence angle
        complement_angle = 90.0 - self.nominal_look_angle_deg
        layover_mask = (slope_deg > complement_angle) & (cos_loc_inc < 0.1)

        # 4. Ray-marched Radar Shadow along range direction
        shadow_mask = self._ray_march_shadow(
            elevation,
            resolution_m=resolution_m,
            look_angle_deg=self.nominal_look_angle_deg,
            orbit_direction=orbit_direction,
        )

        # 5. Blindspot classification and geometric reliability r_geo
        extreme_inc_mask = (local_inc_deg < self.min_incidence_deg) | (local_inc_deg > self.max_incidence_deg)
        blindspot_mask = layover_mask | shadow_mask | extreme_inc_mask

        # Assign reason codes:
        # 0: clear, 1: shadow, 2: layover, 3: extreme incidence
        blindspot_reason = np.zeros((height, width), dtype=np.int16)
        blindspot_reason[extreme_inc_mask] = 3
        blindspot_reason[layover_mask] = 2
        blindspot_reason[shadow_mask] = 1

        # Smooth r_geo reliability in [0, 1]
        r_geo = np.ones((height, width), dtype=np.float32)
        # Decay smoothly outside comfort incidence window
        too_steep = local_inc_deg < self.min_incidence_deg
        too_shallow = local_inc_deg > self.max_incidence_deg
        r_geo[too_steep] = np.clip(local_inc_deg[too_steep] / self.min_incidence_deg, 0.0, 1.0)
        r_geo[too_shallow] = np.clip((90.0 - local_inc_deg[too_shallow]) / (90.0 - self.max_incidence_deg), 0.0, 1.0)

        # Zero out in layover and shadow
        r_geo[layover_mask] = 0.0
        r_geo[shadow_mask] = 0.0

        return GeometryGateOutputs(
            local_incidence_deg=local_inc_deg.astype(np.float32),
            layover_mask=layover_mask,
            shadow_mask=shadow_mask,
            blindspot_mask=blindspot_mask,
            r_geo=r_geo.astype(np.float32),
            blindspot_reason=blindspot_reason,
        )

    def _ray_march_shadow(
        self,
        elevation: np.ndarray,
        resolution_m: float,
        look_angle_deg: float,
        orbit_direction: str,
    ) -> np.ndarray:
        """Ray-march across rows along the radar illumination direction to cast shadows."""
        height, width = elevation.shape
        shadow = np.zeros((height, width), dtype=bool)
        tan_look = np.tan(np.radians(90.0 - look_angle_deg))  # Elevation angle from horizon

        # If Descending: illuminates from East (right) toward West (left)
        # If Ascending: illuminates from West (left) toward East (right)
        if orbit_direction.upper() == "DESCENDING":
            for r in range(height):
                current_ray_elev = -1e9
                for c in range(width - 1, -1, -1):
                    elev = float(elevation[r, c])
                    current_ray_elev -= resolution_m * tan_look
                    if elev < current_ray_elev:
                        shadow[r, c] = True
                    else:
                        current_ray_elev = elev
        else:
            for r in range(height):
                current_ray_elev = -1e9
                for c in range(width):
                    elev = float(elevation[r, c])
                    current_ray_elev -= resolution_m * tan_look
                    if elev < current_ray_elev:
                        shadow[r, c] = True
                    else:
                        current_ray_elev = elev

        return shadow
