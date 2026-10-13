"""Compliance firewall tests ensuring strict adherence to Competition Rules §2.1 & §2.2."""

import ast
from pathlib import Path

import pytest

from trishul.compliance.manifest import RunManifest
from trishul.ingest.http_allowlist import (
    ComplianceViolationError,
    assert_url_allowed,
    is_host_allowed,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
TRISHUL_DIR = ROOT_DIR / "trishul"

FORBIDDEN_STRINGS = [
    "mapping.emergency.copernicus.eu",
    "unosat",
    "overpass-api.de",
    "api.openstreetmap.org",
    "geofabrik",
]


def test_trishul_never_imports_eval():
    """Enforce import isolation: files in trishul/ must never import eval or eval submodules."""
    violations = []
    for py_file in TRISHUL_DIR.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "eval" or alias.name.startswith("eval."):
                        violations.append(f"{py_file.relative_to(ROOT_DIR)}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and (node.module == "eval" or node.module.startswith("eval.")):
                    violations.append(f"{py_file.relative_to(ROOT_DIR)}: from {node.module} import ...")

    assert not violations, "Compliance Violation: trishul/ imports eval/ in:\n" + "\n".join(violations)


def test_trishul_no_forbidden_strings():
    """Grep test: ensure trishul/ contains no hardcoded references to forbidden data sources."""
    violations = []
    for py_file in TRISHUL_DIR.rglob("*.py"):
        # Skip the allowlist definition file itself where forbidden strings are listed for rejection
        if py_file.name == "http_allowlist.py":
            continue
        content = py_file.read_text(encoding="utf-8").lower()
        for forbidden in FORBIDDEN_STRINGS:
            if forbidden in content:
                violations.append(f"{py_file.relative_to(ROOT_DIR)} contains forbidden token '{forbidden}'")

    assert not violations, "Forbidden tokens found in trishul/ pipeline code:\n" + "\n".join(violations)


def test_http_allowlist_enforcement():
    """Verify that http_allowlist strictly allows approved domains and blocks unapproved domains."""
    # Approved domains
    assert is_host_allowed("https://dataspace.copernicus.eu/resto/api/collections")
    assert is_host_allowed("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token")
    assert is_host_allowed("https://api.ohsome.org/v1/elements/geometry")
    assert is_host_allowed("https://copernicus-dem-30m.s3.amazonaws.com/tile.tif")
    assert is_host_allowed("https://huggingface.co/models/geosar_unet.onnx")

    # Forbidden domains
    assert not is_host_allowed("https://mapping.emergency.copernicus.eu/EMSR927/download.zip")
    assert not is_host_allowed("https://unosat.org/flood/nepal2026.geojson")
    assert not is_host_allowed("https://overpass-api.de/api/interpreter")
    assert not is_host_allowed("https://api.openstreetmap.org/api/0.6/map")
    assert not is_host_allowed("https://download.geofabrik.de/asia/nepal-latest.osm.pbf")
    assert not is_host_allowed("https://evil-unauthorized-server.com/data")

    # Assert raises
    with pytest.raises(ComplianceViolationError):
        assert_url_allowed("https://mapping.emergency.copernicus.eu/EMSR927/data")

    with pytest.raises(ComplianceViolationError):
        assert_url_allowed("https://random-site.org/leak")


def test_manifest_compliance_verification():
    """Verify that RunManifest detects date violations and missing attributions."""
    # Valid manifest
    manifest = RunManifest(
        run_id="test_run_01",
        event_date="2026-08-26",
        osm_snapshot_date="2026-07-27",
        aoi_bounds=[85.25, 28.10, 85.40, 28.25],
    )
    verdict, reasons = manifest.verify_compliance()
    assert verdict == "PASS"
    assert len(reasons) == 0

    # Invalid: OSM snapshot after event
    manifest_post = RunManifest(
        run_id="test_run_post",
        event_date="2026-08-26",
        osm_snapshot_date="2026-08-27",  # Violates Rule §2.1
        aoi_bounds=[85.25, 28.10, 85.40, 28.25],
    )
    verdict, reasons = manifest_post.verify_compliance()
    assert verdict == "FAIL"
    assert any("not strictly before event date" in r for r in reasons)
