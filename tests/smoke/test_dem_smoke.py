"""Smoke test for Copernicus DEM 30m ingest."""

import numpy as np

from trishul.ingest.dem import DemManager


def test_dem_smoke():
    """Verify DEM manager yields valid 2D elevation grid with realistic Himalayan elevations."""
    manager = DemManager()
    bounds = [85.25, 28.10, 85.40, 28.25]
    elevation, transform, crs = manager.get_dem_for_bounds(bounds, resolution_m=30.0)

    assert isinstance(elevation, np.ndarray)
    assert elevation.ndim == 2
    assert elevation.shape[0] >= 50
    assert elevation.shape[1] >= 50
    assert crs == "EPSG:4326"

    # Himalayan relief check
    min_elev = np.min(elevation)
    max_elev = np.max(elevation)
    assert min_elev >= 1000.0, f"Valley floor elevation {min_elev}m should exceed 1000m in Trishuli"
    assert max_elev > min_elev + 500.0, "Steep terrain must exhibit significant vertical relief"
