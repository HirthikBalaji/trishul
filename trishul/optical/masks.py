"""Cloud, shadow, and snow masking using Sentinel-2 Scene Classification Layer (SCL).

In compliance with Competition Rules §6.0 & §6.3:
Identifies unusable optical pixels and dilates cloud/shadow masks by 5 pixels
to eliminate edge contamination.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation


def generate_cloud_mask(
    scl: np.ndarray,
    dilate_pixels: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract cloud, shadow, and snow masks from SCL band.

    SCL Classes:
      3: Cloud shadows
      8: Cloud medium probability
      9: Cloud high probability
      10: Thin cirrus
      11: Snow or ice

    Returns:
      (unusable_mask, snow_gate_mask)
    """
    cloud_and_shadow = np.isin(scl, [3, 8, 9, 10])

    if dilate_pixels > 0:
        struct = np.ones((dilate_pixels * 2 + 1, dilate_pixels * 2 + 1), dtype=bool)
        cloud_dilated = binary_dilation(cloud_and_shadow, structure=struct)
    else:
        cloud_dilated = cloud_and_shadow

    snow_mask = (scl == 11)

    return cloud_dilated, snow_mask
