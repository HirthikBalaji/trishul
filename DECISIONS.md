# TRISHUL Architecture & Design Decisions Log

This document records all architectural, technical, and algorithmic decisions made during the development of TRISHUL, adhering to Operating Rule 1 (*Decide, don't ask*).

---

## Decision Index

### DEC-001: Python Runtime & Package Manager
- **Date:** 2026-10-13
- **Context:** Need fast, reproducible dependency resolution supporting PyTorch, GDAL/rasterio, Shapely, GeoPandas, and FastAPI.
- **Options Considered:**
  1. System `pip` + `venv`
  2. `conda` / `mamba`
  3. `uv` with pinned Python 3.12
- **Choice:** `uv` with Python 3.12 (`cpython-3.12-macos-aarch64-none`).
- **Why:** `uv` provides sub-second virtualenv creation, deterministic lockfile pinning, and wheels for Python 3.12 on Apple Silicon / x86_64 across Linux and macOS. Geospatial binary wheels (rasterio, shapely, pyproj) install cleanly without manual C++ library compiling.

---

### DEC-002: Ingest & Mock Fallback Architecture
- **Date:** 2026-10-13
- **Context:** Automated CI/CD, local evaluation, and offline judges may not have active CDSE credentials or network access.
- **Options Considered:**
  1. Fail immediately if credentials or network are missing.
  2. Embed synthetic / cached fixtures with automatic graceful fallback when network/credentials are absent (`TRISHUL_MOCK_NETWORK_ON_MISSING_CREDS=true`).
- **Choice:** Option 2: Automatic graceful fallback to realistic deterministic synthetic fixtures / local cache with clear compliance tagging (`is_mock=True` recorded in `manifest.json`).
- **Why:** Vertical slice and live judge demonstrations must never crash; unauthenticated runs generate realistic physics-constrained terrain and mock SAR/optical stacks while logging the warning honestly in the Claim Ledger.

---

### DEC-003: Compliance Firewall & Host Allowlist Enforcement
- **Date:** 2026-10-13
- **Context:** Competition Hard Rule §2.1 & §2.2 forbid importing `eval/`, accessing post-event OSM, or accessing published disaster maps (`mapping.emergency.copernicus.eu`, `unosat`, etc.).
- **Options Considered:**
  1. Documentation-only rule.
  2. Centralized HTTP wrapper `trishul.ingest.http_allowlist` with domain regex filtering + AST import linter and CI grep checks.
- **Choice:** Option 2: Strict code firewall with allowlist:
  - `dataspace.copernicus.eu` / `identity.dataspace.copernicus.eu` / `stac.dataspace.copernicus.eu` / `sh.dataspace.copernicus.eu`
  - `api.ohsome.org`
  - `copernicus-dem-30m.s3.amazonaws.com`
  - `huggingface.co` / `raw.githubusercontent.com` (for official weights/models)
- **Why:** Ensures compliance-by-construction; any unauthorized domain access raises `ComplianceViolationError` immediately.

---

### DEC-004: Primary AI Component Selection
- **Date:** 2026-10-13
- **Context:** Problem statement Section 3 requires declaring whether the AI component is (A) Flood segmentation model trained on listed data, shown on unseen Himalayan scenes, or (B) Report copilot.
- **Options Considered:**
  1. (B) only (Report copilot)
  2. (A) Flood segmentation model ("GeoSAR-UNet") + grounded report layer (Claim Ledger)
- **Choice:** Option 2: Declare **(A) GeoSAR-UNet Flood Segmentation Model** as the primary AI component, complemented by the Grounded Claim Ledger situation report layer.
- **Why:** Maximum scoring alignment with the 25% AI component weight and 30% flood mapping quality weight, addressing the specific failure mode of debris flood detection in Himalayan topography.

---

### DEC-005: Debris Flood Identification (Innovation #3: EVENT-ADAPT)
- **Date:** 2026-10-13
- **Context:** Listed training sets (Kuro Siwo, Sen1Floods11) only contain water labels. Trishuli 2026 was a debris flood and scour event where dry sediment and scour have different radar cross-sections than open ponded water.
- **Options Considered:**
  1. Standard log-ratio thresholding (misses debris).
  2. Rule-compliant cross-teaching (EVENT-ADAPT): where Sentinel-2 has clear skies, derive pseudo-labels for scour/debris via $\Delta\text{NDVI} \ll 0$ and $\Delta\text{BSI} \gg 0$ hydrologically connected to the channel; train a SAR+terrain student model (LightGBM/small CNN) and evaluate on spatial blocks $\ge 1\text{ km}$, then infer over cloud-covered SAR pixels.
- **Choice:** Option 2: EVENT-ADAPT cross-teaching.
- **Why:** Purely uses allowed raw data, produces debris classification without violating data rules, and directly tackles the Himalayan physics challenge.
