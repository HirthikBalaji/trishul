"""Robustness Matrix Verification Tests (§8).

Verifies graceful degradation under severe real-world failure modes:
1. No usable Sentinel-2 (monsoon cloud) -> SAR-only mode; blindspot map grows; EVENT-ADAPT skipped.
2. Short pre-stack (< 3) -> fallback to pre1-vs-post ratio with explicit low reliability.
3. Empty/sparse OSM -> still maps hazard; assets report insufficient coverage without crashing.
4. Settlement with no pre-event road -> labeled 'no road access (pre-existing)', NEVER 'cut off'.
5. Adversarial Claim Ledger verifier -> catches injected invented numbers.
"""

from __future__ import annotations

import numpy as np
from rasterio.transform import from_bounds

from trishul.exposure.buildings import BuildingExposureEvaluator
from trishul.exposure.roads import RoadExposureEvaluator
from trishul.network.connectivity import NetworkConnectivityAnalyzer
from trishul.network.graph import RoadGraphBuilder
from trishul.optical.composites import OpticalEvidenceAnalyzer
from trishul.report.verifier import LedgerVerifier
from trishul.sar.backends import SarSceneData
from trishul.sar.change import SarChangeDetector


def test_monsoon_cloud_sar_only_mode():
    """Verify that when Sentinel-2 is 100% cloud-covered, optical reliability is 0 and pipeline continues."""
    opt_analyzer = OpticalEvidenceAnalyzer()
    H, W = 50, 50
    # Create 100% cloud SCL band (class 9 = high probability cloud)
    pre_bands = {
        "B02": np.zeros((H, W), dtype=np.float32),
        "B03": np.zeros((H, W), dtype=np.float32),
        "B04": np.zeros((H, W), dtype=np.float32),
        "B08": np.zeros((H, W), dtype=np.float32),
        "B11": np.zeros((H, W), dtype=np.float32),
        "SCL": np.full((H, W), 9, dtype=np.uint8),
    }
    post_bands = pre_bands.copy()

    evidence = opt_analyzer.analyze_change(pre_bands, post_bands)
    assert np.all(evidence.reliability == 0.0), "100% cloud cover must result in zero optical reliability"
    assert np.all(~evidence.clear_sky_mask)


def test_short_pre_stack_fallback():
    """Verify that short pre-stack (<3 scenes) falls back gracefully with lower r_temporal."""
    detector = SarChangeDetector()
    H, W = 40, 40
    transform = from_bounds(85.25, 28.10, 85.35, 28.20, W, H)
    pre = SarSceneData("pre_01", 19, "DESCENDING", "2026-08-14", np.full((H, W), -12.0), np.full((H, W), -19.0), transform, "EPSG:4326")
    post = SarSceneData("post_01", 19, "DESCENDING", "2026-08-27", np.full((H, W), -22.0), np.full((H, W), -28.0), transform, "EPSG:4326")

    # Only 1 pre scene (no stack)
    res = detector.detect_change(pre, post, pre_stack=None)
    assert res.reliability[0, 0] <= 0.25, "Single pre scene must have low reliability factor"
    assert res.z_vv.shape == (H, W)


def test_empty_osm_handling():
    """Verify that empty OSM vectors do not crash exposure or network analyzers."""
    empty_geojson = {"type": "FeatureCollection", "features": []}
    hazard_prob = np.zeros((50, 50), dtype=np.float32)
    transform = from_bounds(85.25, 28.10, 85.35, 28.20, 50, 50)

    bld_eval = BuildingExposureEvaluator().evaluate(empty_geojson, hazard_prob, transform)
    assert bld_eval.total_buildings == 0
    assert bld_eval.expected_hit_count == 0.0

    road_eval = RoadExposureEvaluator().evaluate(empty_geojson, hazard_prob, transform)
    assert road_eval.total_roads == 0
    assert road_eval.total_length_km == 0.0


def test_settlement_no_pre_event_road_labeling():
    """Rule §6.8: Remote settlement without prior road connection must NEVER be labeled 'cut off'."""
    builder = RoadGraphBuilder()
    graph_bundle = builder.build_graph([], [])  # Empty road graph

    osm_places = {
        "features": [
            {
                "id": "node/999",
                "properties": {"name": "Remote High Himalayan Gompa"},
                "geometry": {"type": "Point", "coordinates": [85.30, 28.15]},
            }
        ]
    }
    osm_facilities = {"features": []}

    analyzer = NetworkConnectivityAnalyzer()
    res = analyzer.analyze(graph_bundle, osm_places, osm_facilities)
    assert res.settlements[0].status == "no road access (pre-existing)"
    assert res.settlements[0].status != "cut off by road"


def test_adversarial_ledger_verifier():
    """Verify that LedgerVerifier rejects ungrounded digits or spelled-out numbers."""
    verifier = LedgerVerifier(whitelist_strings=["2026", "2026-08-26"])

    # Permitted string using placeholders and whitelisted year
    valid_text = "On 2026-08-26, TRISHUL mapped [[F:flood_water_km2]] of flood water."
    ok, violations = verifier.verify_template_compliance(valid_text)
    assert ok is True
    assert len(violations) == 0

    # Adversarial string with ungrounded raw digit
    bad_digit_text = "The flood destroyed 41 bridges in the valley."
    ok_bad, violations_bad = verifier.verify_template_compliance(bad_digit_text)
    assert ok_bad is False
    assert any("Forbidden raw digit" in v for v in violations_bad)

    # Adversarial string with spelled-out number
    bad_spelled_text = "There were two hundred buildings damaged."
    ok_word, violations_word = verifier.verify_template_compliance(bad_spelled_text)
    assert ok_word is False
    assert any("two" in v for v in violations_word)
