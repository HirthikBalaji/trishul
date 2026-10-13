"""FastAPI data contracts and schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    aoi: Any = Field(..., description="Bbox list [min_x, min_y, max_x, max_y] or GeoJSON polygon")
    event_date: str = Field(default="2026-08-26", description="ISO date YYYY-MM-DD")
    lang: str = Field(default="en", description="'en' or 'ne'")


class RunStatusResponse(BaseModel):
    run_id: str
    status: str  # "queued", "running", "completed", "failed"
    compliance_verdict: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    manifest_url: str | None = None
    report_en_url: str | None = None
    report_ne_url: str | None = None
