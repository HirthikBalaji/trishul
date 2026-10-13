"""Smoke test for ohsome historical OSM snapshot API."""

from trishul.ingest.ohsome import OhsomeClient


def test_ohsome_smoke():
    """Verify ohsome client extracts pre-event vector assets without breaking."""
    client = OhsomeClient()
    bounds = [85.25, 28.10, 85.40, 28.25]
    event_date = "2026-08-26"

    snapshot = client.get_latest_valid_snapshot(event_date)
    assert snapshot < event_date, f"Snapshot {snapshot} must be strictly before {event_date}"

    roads = client.fetch_elements(bounds, "highway=* and geometry:line", snapshot, asset_type="roads")
    assert "features" in roads
    assert len(roads["features"]) > 0

    buildings = client.fetch_elements(bounds, "building=* and geometry:polygon", snapshot, asset_type="buildings")
    assert "features" in buildings
    assert len(buildings["features"]) > 0
