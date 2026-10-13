"""Smoke test for CDSE STAC and Auth APIs."""

from trishul.ingest.cdse import CDSEClient


def test_cdse_search_smoke():
    """Verify CDSE candidate search runs cleanly for Himalayan bounding box."""
    client = CDSEClient(allow_mock=True)
    bounds = [85.25, 28.10, 85.40, 28.25]  # Trishuli Valley bbox
    event_date = "2026-08-26"

    pre_stack, post_scenes = client.search_s1_scenes(bounds, event_date)
    assert len(pre_stack) >= 3, "Sentinel-1 pre-event stack must have at least 3 scenes"
    assert len(post_scenes) >= 1, "Must find at least 1 post-event Sentinel-1 scene"
    assert pre_stack[0].relative_orbit == post_scenes[0].relative_orbit, "Must pair identical relative orbit"
    assert pre_stack[0].orbit_direction == post_scenes[0].orbit_direction, "Must pair identical orbit direction"

    pre_s2, post_s2 = client.search_s2_scenes(bounds, event_date)
    assert pre_s2 is not None
    assert post_s2 is not None
    assert pre_s2.sensor == "S2"
