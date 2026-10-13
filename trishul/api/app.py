"""FastAPI application for TRISHUL pipeline and dashboard backend.

Provides RESTful endpoints for Judge Mode execution, partial results streaming,
asset inspection, facts ledger access, static dashboard delivery, and PDF report delivery.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from trishul.api.schemas import RunRequest, RunStatusResponse
from trishul.pipeline import run_pipeline

app = FastAPI(
    title="TRISHUL API",
    description="Physics-Gated Multimodal Flood Damage Mapping Backend",
    version="1.0.0",
)

# Enable CORS for React/Vite dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory run state tracker
RUNS_REGISTRY: dict[str, dict[str, Any]] = {}
RUNS_BASE_DIR = Path("./runs")
WEB_DIR = Path("./web")


def resolve_run_dir(run_id: str) -> Path:
    """Resolve run directory, supporting the 'latest' alias."""
    if run_id == "latest":
        # Look for trishuli_aug2026 or most recently modified run folder
        candidates = [d for d in RUNS_BASE_DIR.glob("*") if d.is_dir()]
        if not candidates:
            case_study_dir = RUNS_BASE_DIR / "trishuli_aug2026"
            case_study_dir.mkdir(parents=True, exist_ok=True)
            from trishul.pipeline import run_pipeline
            run_pipeline(
                aoi=[85.25, 28.08, 85.42, 28.24],
                event_date="2026-08-26",
                output_dir=case_study_dir,
                run_id="trishuli_aug2026",
            )
            return case_study_dir
        # Prefer trishuli_aug2026 if exists
        case_study_dir = RUNS_BASE_DIR / "trishuli_aug2026"
        if case_study_dir.exists():
            return case_study_dir
        # Otherwise sort by mtime
        return max(candidates, key=lambda d: d.stat().st_mtime)

    p = RUNS_BASE_DIR / run_id
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return p


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "trishul-api", "version": "1.0.0"}


@app.post("/api/run", response_model=RunStatusResponse)
def trigger_run(req: RunRequest, background_tasks: BackgroundTasks) -> RunStatusResponse:
    """Trigger a new pipeline run for an AOI and event date."""
    import uuid
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    out_dir = RUNS_BASE_DIR / run_id

    RUNS_REGISTRY[run_id] = {
        "status": "queued",
        "output_dir": str(out_dir),
        "event_date": req.event_date,
        "aoi": req.aoi,
        "metrics": {},
    }

    def _execute():
        RUNS_REGISTRY[run_id]["status"] = "running"
        try:
            res = run_pipeline(
                aoi=req.aoi,
                event_date=req.event_date,
                output_dir=out_dir,
                run_id=run_id,
                lang=req.lang,
            )
            RUNS_REGISTRY[run_id]["status"] = "completed"
            RUNS_REGISTRY[run_id]["metrics"] = res.get("metrics", {})
        except Exception as e:
            RUNS_REGISTRY[run_id]["status"] = "failed"
            RUNS_REGISTRY[run_id]["error"] = str(e)

    background_tasks.add_task(_execute)

    return RunStatusResponse(
        run_id=run_id,
        status="queued",
    )


@app.get("/api/run/{run_id}", response_model=RunStatusResponse)
def get_run_status(run_id: str) -> RunStatusResponse:
    """Retrieve execution status and metadata for a given run ID."""
    run_dir = resolve_run_dir(run_id)
    manifest_file = run_dir / "manifest.json"

    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        facts_file = run_dir / "facts.json"
        metrics = {}
        if facts_file.exists():
            with open(facts_file, "r", encoding="utf-8") as f:
                facts = json.load(f)
                metrics = {fact["id"]: fact["value"] for fact in facts}

        return RunStatusResponse(
            run_id=manifest_data.get("run_id", run_id),
            status="completed",
            compliance_verdict=manifest_data.get("compliance_verdict", "PASS"),
            metrics=metrics,
            manifest_url=f"/api/run/{run_id}/manifest",
            report_en_url=f"/api/run/{run_id}/report/en",
            report_ne_url=f"/api/run/{run_id}/report/ne",
        )

    run_info = RUNS_REGISTRY.get(run_id)
    if not run_info:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    return RunStatusResponse(
        run_id=run_id,
        status=run_info.get("status", "unknown"),
        metrics=run_info.get("metrics", {}),
    )


@app.get("/api/run/{run_id}/manifest")
def get_manifest(run_id: str):
    """Retrieve compliance manifest JSON."""
    run_dir = resolve_run_dir(run_id)
    p = run_dir / "manifest.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Manifest not found")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/run/{run_id}/facts")
def get_facts(run_id: str):
    """Retrieve auditable Claim Ledger facts JSON."""
    run_dir = resolve_run_dir(run_id)
    p = run_dir / "facts.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Facts not found")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/run/{run_id}/assets")
def get_assets(run_id: str):
    """Retrieve evaluated assets GeoJSON."""
    run_dir = resolve_run_dir(run_id)
    p = run_dir / "assets.geojson"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Assets not found")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/run/{run_id}/network")
def get_network(run_id: str):
    """Retrieve road graph and settlement isolation network.json."""
    run_dir = resolve_run_dir(run_id)
    p = run_dir / "network.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Network bundle not found")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/run/{run_id}/report/{lang}")
def download_report_pdf(run_id: str, lang: str = "en"):
    """Serve PDF report file."""
    run_dir = resolve_run_dir(run_id)
    p = run_dir / f"report_{lang}.pdf"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Report PDF for language '{lang}' not found")
    return FileResponse(p, media_type="application/pdf", filename=f"trishul_report_{lang}.pdf")


@app.get("/api/run/{run_id}/restore")
def get_restore_ranking(run_id: str):
    """Serve restore ranking CSV."""
    run_dir = resolve_run_dir(run_id)
    p = run_dir / "restore_ranking.csv"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Restore ranking not found")
    return FileResponse(p, media_type="text/csv", filename="restore_ranking.csv")


@app.get("/api/run/{run_id}/flowpath")
def get_flowpath(run_id: str):
    """Serve flood-path GeoJSON."""
    run_dir = resolve_run_dir(run_id)
    p = run_dir / "flowpath.geojson"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Flowpath GeoJSON not found")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/run/{run_id}/raster/bounds")
def get_raster_bounds(run_id: str):
    """Retrieve bounding box and 4-corner coordinates for MapLibre ImageSource."""
    import rasterio
    run_dir = resolve_run_dir(run_id)
    tif_path = run_dir / "hazard_class.tif"
    if not tif_path.exists():
        tif_path = run_dir / "hazard_prob.tif"
    if not tif_path.exists():
        raise HTTPException(status_code=404, detail="Hazard raster not found")

    with rasterio.open(tif_path) as src:
        b = src.bounds
        coords = [
            [b.left, b.top],
            [b.right, b.top],
            [b.right, b.bottom],
            [b.left, b.bottom],
        ]
        return {
            "coordinates": coords,
            "bbox": [b.left, b.bottom, b.right, b.top],
            "center": [(b.left + b.right) / 2.0, (b.top + b.bottom) / 2.0],
        }


@app.get("/api/run/{run_id}/raster/hazard.png")
def get_hazard_png(run_id: str):
    """Render transparent RGBA PNG overlay from hazard_class.tif."""
    from io import BytesIO

    import numpy as np
    import rasterio
    from fastapi.responses import Response
    from PIL import Image

    run_dir = resolve_run_dir(run_id)
    tif_path = run_dir / "hazard_class.tif"
    if not tif_path.exists():
        raise HTTPException(status_code=404, detail="hazard_class.tif not found")

    with rasterio.open(tif_path) as src:
        arr = src.read(1)
        h, w = arr.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # Class 1: Permanent water -> Cyan
        rgba[arr == 1] = [0, 168, 204, 200]
        # Class 2: Flood water -> Deep Blue
        rgba[arr == 2] = [30, 90, 240, 220]
        # Class 3: Debris flow -> Amber / Terracotta
        rgba[arr == 3] = [220, 110, 20, 210]
        # Class 6 / other: Scour / channel inundation -> Crimson
        rgba[arr == 6] = [200, 40, 40, 220]
        rgba[(arr != 0) & (arr != 1) & (arr != 2) & (arr != 3) & (arr != 6)] = [220, 110, 20, 180]

        img = Image.fromarray(rgba, mode="RGBA")
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/api/run/{run_id}/raster/blindspot.png")
def get_blindspot_png(run_id: str):
    """Render transparent RGBA PNG overlay for radar shadow and layover."""
    from io import BytesIO

    import numpy as np
    import rasterio
    from fastapi.responses import Response
    from PIL import Image

    run_dir = resolve_run_dir(run_id)
    tif_path = run_dir / "blindspot.tif"
    if not tif_path.exists():
        raise HTTPException(status_code=404, detail="blindspot.tif not found")

    with rasterio.open(tif_path) as src:
        arr = src.read(1)
        h, w = arr.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # Class 1: Radar shadow / extreme terrain occlusion -> Hatched yellow/amber
        rgba[arr == 1] = [245, 158, 11, 160]
        rgba[arr == 2] = [234, 88, 12, 170]

        img = Image.fromarray(rgba, mode="RGBA")
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="image/png")



@app.post("/api/run/{run_id}/chat")
def chat_copilot(run_id: str, payload: dict[str, Any]):
    """Grounded Situation Report Copilot answering rescuers' tactical queries."""
    from trishul.report.copilot import GroundedReportCopilot
    from trishul.report.ledger import ClaimLedger

    run_dir = resolve_run_dir(run_id)
    p_facts = run_dir / "facts.json"
    if not p_facts.exists():
        raise HTTPException(status_code=404, detail="Facts ledger not found")

    ledger = ClaimLedger(run_id=run_id)
    with open(p_facts, "r", encoding="utf-8") as f:
        facts_list = json.load(f)
    for fact in facts_list:
        ledger.register(
            fact["id"],
            fact["value"],
            fact.get("unit", ""),
            fact.get("ci_lo"),
            fact.get("ci_hi"),
            fact.get("layer", "general"),
            fact.get("method", "direct"),
        )

    copilot = GroundedReportCopilot(ledger)
    query = payload.get("query", "")
    lang = payload.get("lang", "en")
    return copilot.answer_query(query, lang=lang)


# Serve Web Dashboard at root / and /dashboard
@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    index_file = WEB_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard index.html not found")
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
