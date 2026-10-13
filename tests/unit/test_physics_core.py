"""Unit tests for Milestone 2 Physics Core: Geometry gate, Optical, HAND, and Kuro Siwo data."""

import numpy as np

from trishul.model.data import KuroSiwoDatasetLoader
from trishul.optical.indices import compute_bsi, compute_mndwi, compute_ndvi
from trishul.sar.geometry import SarGeometryGate
from trishul.terrain.hand import HandProcessor


def test_sar_geometry_synthetic_ridge():
    """Verify that ray-marched geometry gate flags shadow and layover on synthetic terrain."""
    gate = SarGeometryGate(nominal_look_angle_deg=38.0)
    # Create an east-west ridge: high elevation in center
    elevation = np.zeros((100, 100), dtype=np.float32)
    # Peak at column 50
    for c in range(100):
        elevation[:, c] = 2500.0 - abs(c - 50) * 60.0  # Steep ridge (> 38.4 m/pixel drop)

    out_desc = gate.compute_geometry(elevation, resolution_m=30.0, orbit_direction="DESCENDING")
    assert out_desc.r_geo.shape == (100, 100)
    # In descending pass (illuminates from East/right), west side (cols < 50) must contain shadows
    assert np.any(out_desc.shadow_mask[:, :45]), "West slope must experience radar shadow in descending pass"
    assert np.any(out_desc.blindspot_mask), "Steep ridge must exhibit blindspots"


def test_optical_indices():
    """Verify mathematical properties of spectral indices."""
    # Water pixel: low NIR, high Green, low SWIR
    nir_water = np.array([0.02], dtype=np.float32)
    red_water = np.array([0.05], dtype=np.float32)
    green_water = np.array([0.15], dtype=np.float32)
    swir_water = np.array([0.01], dtype=np.float32)

    mndwi = compute_mndwi(green_water, swir_water)
    ndvi_water = compute_ndvi(nir_water, red_water)
    assert mndwi[0] > 0.5, "Clean water should yield positive MNDWI"
    assert ndvi_water[0] < 0.0, "Water should yield negative NDVI"

    # Healthy vegetation: high NIR, low Red
    nir_veg = np.array([0.45], dtype=np.float32)
    red_veg = np.array([0.05], dtype=np.float32)
    ndvi = compute_ndvi(nir_veg, red_veg)
    assert ndvi[0] > 0.6, "Healthy vegetation should exhibit high NDVI"

    # Bare sediment / soil: high SWIR and Red
    blue_soil = np.array([0.12], dtype=np.float32)
    red_soil = np.array([0.25], dtype=np.float32)
    nir_soil = np.array([0.18], dtype=np.float32)
    swir_soil = np.array([0.35], dtype=np.float32)
    bsi = compute_bsi(blue_soil, red_soil, nir_soil, swir_soil)
    assert bsi[0] > 0.1, "Bare sediment should yield positive BSI"


def test_hand_and_hydraulic_prior():
    """Verify HAND computation on synthetic V-shaped valley."""
    processor = HandProcessor(resolution_m=30.0, nominal_flood_stage_m=18.0)
    # Synthetic terrain with stream down the middle
    elevation = np.zeros((80, 80), dtype=np.float32)
    y, x = np.mgrid[0:80, 0:80]
    elevation = 2000.0 - 5.0 * y + 15.0 * np.abs(x - 40)

    bundle = processor.compute_hand(elevation)
    assert bundle.hand_m.shape == (80, 80)
    # On the stream thalweg, HAND should be near 0
    assert np.min(bundle.hand_m) == 0.0
    # On high ridge slopes, HAND should be high and hydraulic prior should be very low
    ridge_prior = bundle.hydraulic_prior[:, 0]
    assert np.all(ridge_prior < 0.2), "Ridge pixels far above stage should have low hydraulic prior"


def test_kuro_siwo_dataset_loader():
    """Verify Kuro Siwo patch structure and held-out validation split."""
    loader = KuroSiwoDatasetLoader(patch_size=224)
    # Iterate validation split
    val_patches = list(loader.iterate_split(split="val", patches_per_event=2))
    assert len(val_patches) == 4
    patch = val_patches[0]
    assert patch.features.shape == (10, 224, 224)
    assert patch.labels.shape == (224, 224)
    assert patch.event_id in loader.val_events, "Validation patch must strictly belong to held-out validation events"
