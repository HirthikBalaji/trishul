"""Run compliance manifest generation and verification.

In compliance with Competition Rules §2.2:
Every run writes manifest.json with complete lineage, hashes, acquisition dates,
OSM snapshot date, model checksum, and an automated compliance verdict (PASS/FAIL).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

MANDATORY_ATTRIBUTIONS = [
    "Contains modified Copernicus Sentinel data 2026.",
    "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.",
    "© OpenStreetMap contributors.",
]

DISCLAIMER_TEXT = (
    "Educational prototype, not an operational tool. Developed for Multimodal AI Hackathon 2026."
)


def get_git_commit() -> str:
    """Retrieve current git HEAD commit SHA."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown-commit"


def compute_file_hash(path: Path | str) -> str:
    """Compute SHA-256 hash of a file."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return "nonexistent"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


def compute_dict_hash(d: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of a dictionary."""
    serialized = json.dumps(d, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


class ProductRecord(BaseModel):
    sensor: str  # e.g., "Sentinel-1", "Sentinel-2", "Copernicus-DEM"
    product_id: str
    acquisition_time: str
    relative_orbit: int | None = None
    orbit_direction: str | None = None  # "ASCENDING" or "DESCENDING"
    sha256: str = "mock-sha256"


class RunManifest(BaseModel):
    run_id: str
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    git_commit: str = Field(default_factory=get_git_commit)
    config_hash: str = "default-config"
    event_date: str
    osm_snapshot_date: str
    aoi_bounds: list[float]  # [min_lon, min_lat, max_lon, max_lat]
    products: list[ProductRecord] = Field(default_factory=list)
    model_checksum: str = "geosar-unet-onnx-v1"
    compliance_verdict: str = "PASS"  # "PASS" or "FAIL"
    compliance_reasons: list[str] = Field(default_factory=list)
    attributions: list[str] = Field(default_factory=lambda: list(MANDATORY_ATTRIBUTIONS))
    disclaimer: str = DISCLAIMER_TEXT
    is_mock: bool = False

    def verify_compliance(self) -> tuple[str, list[str]]:
        """Verify compliance rules and return (verdict, reasons)."""
        reasons = []

        # 1. OSM Snapshot date must be strictly before event_date
        try:
            event_dt = datetime.fromisoformat(self.event_date)
            osm_dt = datetime.fromisoformat(self.osm_snapshot_date)
            if osm_dt >= event_dt:
                reasons.append(
                    f"OSM snapshot date ({self.osm_snapshot_date}) is not strictly before event date ({self.event_date})."
                )
        except Exception as e:
            reasons.append(f"Date parsing error during compliance check: {e}")

        # 2. Check for presence of required attributions
        for req in MANDATORY_ATTRIBUTIONS:
            if req not in self.attributions:
                reasons.append(f"Missing mandatory attribution: {req}")

        verdict = "PASS" if not reasons else "FAIL"
        self.compliance_verdict = verdict
        self.compliance_reasons = reasons
        return verdict, reasons

    def save(self, run_dir: Path | str) -> Path:
        """Write manifest.json to run output bundle."""
        self.verify_compliance()
        out_path = Path(run_dir) / "manifest.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        return out_path
