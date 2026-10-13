"""Unit tests for Milestone 3: GeoSAR-UNet inference, EVENT-ADAPT, and Hydraulic Stage Profile."""

import numpy as np

from trishul.model.adapt import EventAdaptCrossTeacher
from trishul.model.infer import GeoSarUnetInferencer
from trishul.terrain.stage import HydraulicStageReconstructor


def test_geosar_unet_tiled_inference():
    """Verify that GeoSarUnetInferencer executes Hann-blended tiling without seams."""
    inferencer = GeoSarUnetInferencer(model_path="models/geosar_unet_v1.onnx", patch_size=224, stride=112)
    # Test on full valley grid
    H, W = 150, 150
    features = np.random.randn(10, H, W).astype(np.float32)
    features[4] = -12.0  # Post VV
    features[6] = -4.0   # Diff VV (flood signal)

    result = inferencer.predict_full_raster(features)
    assert result.prob_flood_water.shape == (H, W)
    assert np.all(result.prob_flood_water >= 0.0)
    assert np.all(result.prob_flood_water <= 1.0)
    # Sum of classes should be close to 1
    total_prob = result.prob_no_water + result.prob_perm_water + result.prob_flood_water
    assert np.allclose(total_prob, 1.0, atol=1e-3)


def test_event_adapt_cross_teaching():
    """Verify that EVENT-ADAPT trains SAR student on clear-sky teacher pixels."""
    adapt_engine = EventAdaptCrossTeacher(spatial_block_size_cells=15)
    H, W = 60, 60

    delta_ndvi = np.zeros((H, W), dtype=np.float32)
    delta_bsi = np.zeros((H, W), dtype=np.float32)
    clear_sky_mask = np.ones((H, W), dtype=bool)
    hand_m = np.full((H, W), 20.0, dtype=np.float32)
    slope_deg = np.full((H, W), 15.0, dtype=np.float32)
    z_vv = np.zeros((H, W), dtype=np.float32)
    z_vh = np.zeros((H, W), dtype=np.float32)
    r_geo = np.ones((H, W), dtype=np.float32)

    # Inject teacher debris pixels in valley floor
    delta_ndvi[10:30, 10:30] = -0.25
    delta_bsi[10:30, 10:30] = 0.18
    z_vv[10:30, 10:30] = 2.5

    # Stable pixels
    delta_ndvi[35:55, 35:55] = 0.01
    hand_m[35:55, 35:55] = 50.0

    res = adapt_engine.adapt_and_infer(
        delta_ndvi, delta_bsi, clear_sky_mask, hand_m, slope_deg, z_vv, z_vh, r_geo
    )
    assert res.adapted is True
    assert res.debris_scour_prob.shape == (H, W)
    # Debris zone should have higher probability than stable mountain zone
    assert np.mean(res.debris_scour_prob[10:30, 10:30]) > np.mean(res.debris_scour_prob[35:55, 35:55])


def test_hydraulic_stage_profile():
    """Verify along-channel flood stage reconstruction."""
    recon = HydraulicStageReconstructor(spacing_m=150.0, resolution_m=30.0)
    H, W = 100, 100

    stream_mask = np.zeros((H, W), dtype=bool)
    stream_mask[:, 50] = True  # North to south stream down center column

    hand_m = np.zeros((H, W), dtype=np.float32)
    for c in range(W):
        hand_m[:, c] = abs(c - 50) * 2.0

    hazard_prob = np.zeros((H, W), dtype=np.float32)
    hazard_prob[:, 45:56] = 0.6  # Boundary pixels around column 45 and 55 (HAND ~ 10m)

    res = recon.reconstruct_profile(stream_mask, hand_m, hazard_prob)
    assert len(res.thalweg_points) > 0
    assert len(res.flood_stage_m) == len(res.thalweg_points)
    assert res.converged is True
    # Flood stage should be in realistic bounds
    assert 5.0 <= np.mean(res.flood_stage_m) <= 30.0
