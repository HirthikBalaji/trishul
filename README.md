# TRISHUL (त्रिशूल)
### Physics-Gated Multimodal Flood Damage Mapping & Isolation Analysis for Himalayan Valleys
**Multimodal AI Hackathon 2026 · Track B: Mapping Flood Damage from Space**

[![CI](https://github.com/trishul-team/trishul/actions/workflows/ci.yml/badge.svg)](https://github.com/trishul-team/trishul/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data: ODbL](https://img.shields.io/badge/Data-ODbL-blue.svg)](LICENSE-DATA.md)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

---

## 1. Overview & The Thesis

Given an **area (polygon/bbox)** and a **flood date**, TRISHUL answers three critical operational questions for disaster responders in steep Himalayan terrain:

1. **Where did the flood hit?** (Prong 1: Hazard) — Calibrated water, debris deposit, and channel scour probabilities from Sentinel-1 SAR and Sentinel-2 optical imagery.
2. **What was damaged?** (Prong 2: Exposure) — Pre-event OpenStreetMap buildings, roads, and bridges with probabilistic damage scores ($p_{\text{hit}}$).
3. **Who is cut off?** (Prong 3: Isolation) — Settlements isolated from hospitals and supply hubs evaluated via spatially correlated Monte-Carlo network percolation, with a greedy restore-first ranking.

### Why Standard Methods Fail in the Himalaya (and Why TRISHUL Wins)
Conventional flood mapping thresholds a single SAR log-ratio and overlays it with OSM. In the Himalaya, this fails because:
- **Radar layover and shadow** produce false water detections. *TRISHUL introduces a ray-marched terrain-geometry gate and multi-track fusion (ascending + descending).*
- **Debris floods, not clean water:** Events like the August 2026 Trishuli flood deposit rocky slurry with high radar backscatter. *TRISHUL introduces EVENT-ADAPT cross-teaching.*
- **Sub-pixel bridges:** 10 m pixels cannot resolve bridges. *TRISHUL evaluates hydraulic flood-stage vs deck clearance and bank approach damage.*
- **Uncertainty loss:** Thresholding discards confidence. *TRISHUL propagates calibrated probabilities from radar pixels to road graphs.*
- **Monsoon cloud cover:** *TRISHUL operates gracefully in SAR-only mode, publishing an honest Blindspot Map.*

---

## 2. System Architecture

```
            ┌────────────── INGEST + COMPLIANCE FIREWALL (§6.0) ───────────────┐
            │ CDSE S1/S2 · Copernicus DEM · ohsome OSM snapshot · host allowlist │
            └───────┬───────────────┬───────────────────┬──────────────┬───────┘
                    ▼               ▼                   ▼              ▼
          SAR GEOMETRY GATE   OPTICAL EVIDENCE     TERRAIN LAYER    OSM ASSETS
          (shadow/layover,    (NDVI/BSI/NDWI,      (HAND, flow      (buildings,
           local incidence,    cloud/snow gates)    paths, slopes)   roads, bridges)
           per-track r_geo)         │                   │               │
                    ▼               │                   │               │
          TEMPORAL-STATS CHANGE     │                   │               │
          (same-track stack,        │                   │               │
           calibrated p-values)     │                   │               │
                    ▼               ▼                   ▼               │
          ┌──────────── AI: Segmentation CNN (Kuro Siwo) ───────────┐    │
          │  + EVENT-ADAPT: optical→SAR cross-teaching for debris   │    │
          └──────────────────────────┬──────────────────────────────┘    │
                                     ▼                                   │
                  EVIDENCE FUSION (calibrated log-odds, reliability-     │
                  weighted, multi-track) ⇄ HYDRAULIC CONSISTENCY LAYER   │
                  (flood-stage profile along thalweg, HAND prior)        │
                                     ▼                                   │
        ┌───────────── PRONG 1: HAZARD  (Where did it hit?) ─────────────┤
        │  hazard_prob (water/debris/scour/slope-failure), blindspot map │
        ├───────────── PRONG 2: EXPOSURE (What was damaged?) ◄───────────┘
        │  per-asset p_hit with CI; bridge model (sub-pixel)
        ├───────────── PRONG 3: ISOLATION (Who is cut off?)
        │  Monte-Carlo road-graph survival → P(cut-off), repair priority
        └──────────────► CLAIM LEDGER → EN/NE report · Dashboard · Q&A
```

---

## 3. Quickstart & Reproducibility

### Prerequisites
- Python 3.12+ and `uv` (recommended) or standard `pip`
- Node.js 18+ (for Web Dashboard)

### Installation
```bash
# Clone the repository
git clone https://github.com/trishul-team/trishul.git
cd trishul

# Set up virtual environment and install dependencies
make setup
```

### Running the Case Study (Trishuli Valley, August 2026)
```bash
# Run end-to-end pipeline on Trishuli case study
make case-study
```

### Running Judge Mode on Any Arbitrary Valley
```bash
# Run for any custom AOI polygon / bbox and event date
python -m trishul.pipeline \
  --aoi '{"type": "Polygon", "coordinates": [[[85.25, 28.10], [85.40, 28.10], [85.40, 28.25], [85.25, 28.25], [85.25, 28.10]]]}' \
  --event-date 2026-08-26 \
  --output runs/judge_run
```

### Launching the Dashboard and API
```bash
make api   # Starts FastAPI backend on http://localhost:8000
make web   # Starts React/MapLibre dashboard on http://localhost:5173
```

---

## 4. Required Attributions

- "Contains modified Copernicus Sentinel data 2026."
- "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved."
- "© OpenStreetMap contributors."
- "European Union, Copernicus Emergency Management Service data" (for EMSR927 evaluation comparison).

### Training Dataset Citations
```bibtex
@inproceedings{bountos2024kurosiwo,
  title={Kuro Siwo: A Multimodal Multi-temporal Benchmark Dataset for Global Flood Mapping},
  author={Bountos, Nikolaos Ioannis and Kondylatos, Spyros and Papoutsis, Ioannis and Michail, Dimitrios and others},
  booktitle={Thirty-eighth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2024}
}
```

---

## 5. Responsible Use & Disclaimer
**Educational prototype, not an operational tool.**
This system is an experimental prototype developed for the Multimodal AI Hackathon 2026. It is not an operational life-safety dispatch system. In respect for communities affected by recent disasters, no imagery of victims is used or displayed anywhere in this project.
