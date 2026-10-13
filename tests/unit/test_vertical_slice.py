"""Unit and integration test verifying Milestone 1 Vertical Slice end-to-end."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from trishul.api.app import app
from trishul.pipeline import run_pipeline


def test_vertical_slice_pipeline(tmp_path: Path):
    """Verify that a minimal pipeline run produces all required bundle artifacts."""
    test_aoi = [85.30, 28.12, 85.36, 28.18]
    event_date = "2026-08-26"
    out_dir = tmp_path / "slice_run"

    result = run_pipeline(
        aoi=test_aoi,
        event_date=event_date,
        output_dir=out_dir,
        run_id="test_slice_01",
    )
    assert result["metrics"]["total_affected_km2"] >= 0

    # 1. Verify files exist
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "facts.json").exists()
    assert (out_dir / "assets.geojson").exists()
    assert (out_dir / "network.json").exists()
    assert (out_dir / "hazard_prob.tif").exists()
    assert (out_dir / "hazard_class.tif").exists()
    assert (out_dir / "reliability.tif").exists()
    assert (out_dir / "blindspot.tif").exists()
    assert (out_dir / "report_en.pdf").exists()
    assert (out_dir / "report_ne.pdf").exists()

    # 2. Verify manifest compliance
    with open(out_dir / "manifest.json") as f:
        manifest = json.load(f)
    assert manifest["compliance_verdict"] == "PASS"
    assert len(manifest["compliance_reasons"]) == 0
    assert manifest["event_date"] == event_date

    # 3. Verify facts
    with open(out_dir / "facts.json") as f:
        facts = json.load(f)
    fact_ids = {f["id"] for f in facts}
    assert "flood_water_km2" in fact_ids
    assert "debris_deposit_km2" in fact_ids
    assert "buildings_hit_expected" in fact_ids
    assert "road_length_hit_km" in fact_ids


def test_api_endpoints(tmp_path: Path, monkeypatch):
    """Verify FastAPI endpoints serve the latest run bundle and dashboard."""
    import trishul.api.app as api_app

    run_dir = tmp_path / "trishuli_aug2026"
    run_pipeline(
        aoi=[85.30, 28.12, 85.36, 28.18],
        event_date="2026-08-26",
        output_dir=run_dir,
        run_id="test_slice_api",
    )
    monkeypatch.setattr(api_app, "RUNS_BASE_DIR", tmp_path)

    client = TestClient(app)

    # Health check
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

    # Dashboard root
    res_dash = client.get("/")
    assert res_dash.status_code == 200
    assert "TRISHUL" in res_dash.text

    # Latest run endpoints
    res_run = client.get("/api/run/latest")
    assert res_run.status_code == 200
    assert res_run.json()["compliance_verdict"] == "PASS"

    res_facts = client.get("/api/run/latest/facts")
    assert res_facts.status_code == 200
    assert isinstance(res_facts.json(), list)

    res_manifest = client.get("/api/run/latest/manifest")
    assert res_manifest.status_code == 200
    assert res_manifest.json()["compliance_verdict"] == "PASS"

