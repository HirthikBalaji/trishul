# TRISHUL — Operational Bounds & Technical Limitations

In compliance with Competition Rules §0 Rule 4, §1, and §11:
Honesty about engineering boundaries and satellite physics constraints is essential. This document transparently articulates what TRISHUL can and cannot do.

---

## 1. Satellite Constellation Revisit & Latency
- **No Minutes-Ahead Warning:** Synthetic Aperture Radar (Sentinel-1C/D) and optical (Sentinel-2A/B) constellations have nominal orbital repeat cycles of 6 to 12 days. TRISHUL is designed for **rapid post-event situational assessment and rescue logistics**, not real-time early warning before an avalanche or glacial lake outburst occurs.
- **Flood Recession and Reworking:** If the first cloud-free optical or radar acquisition occurs 3 to 7 days after the peak hydrograph, ponded water may have infiltrated, and fine sediment slurry may be redistributed by monsoon rains. The pipeline reflects this by scaling the temporal reliability factor $r_{\text{temporal}}$.

---

## 2. Terrain Blindspots (Radar Layover & Shadow)
- **Himalayan Topography Constraints:** In ultra-steep gorges (slopes exceeding 45° to 65°), radar pulses incident at ~38° inevitably experience layover (slopes facing the antenna) or radar shadow (slopes facing away).
- **The Blindspot Map:** Rather than hallucinating or guessing hazard status in unobservable terrain, TRISHUL computes a 3D ray-marched **Blindspot Map** with explicit reason codes (`shadow`, `layover`, `extreme incidence`). Responders are warned that unobservable terrain carries high residual uncertainty.

---

## 3. Spatial Resolution & Sub-Pixel Asset Grading
- **10 m SAR / 30 m DEM Pixel Scale:** A 10-meter pixel typically covers an entire single-story residential dwelling or bridge deck. Consequently:
  - We model assets **probabilistically** ($p_{\text{hit}} \in [0, 1]$) with 90% confidence intervals.
  - We do **not** claim structural engineering damage states (e.g. "collapsed" vs "minor structural cracking").
- **Sub-Pixel Bridges:** 10 m radar cannot resolve bridge spans directly. Bridges are evaluated using a physical noisy-OR model combining hydraulic thalweg overtopping, bank approach scouring, and bridge radar backscatter loss.

---

## 4. Debris Flow Detection (EVENT-ADAPT)
- **Absence of Listed Debris Labels:** Global benchmark datasets (Kuro Siwo, Sen1Floods11) solely annotate open ponded water. The August 2026 Trishuli disaster was primarily a sediment and rock debris flood.
- **Pseudo-label Circularity Acknowledgment:** Innovation #3 (EVENT-ADAPT) trains a SAR+terrain student on optical teacher labels within the same event. While validated using spatial block cross-validation (blocks $\ge 1\text{ km}$), pseudo-labels remain an approximation of ground-truth debris deposition.

---

## 5. OpenStreetMap Baseline Completeness
- **Pre-event Snapshot Bounds:** By Competition Rule §2.1, TRISHUL strictly queries OSM data created prior to the disaster date.
- **Informal Settlements and Remote Footpaths:** If a remote Himalayan hamlet was never mapped in OSM prior to the event, it cannot be modeled. If a settlement lacked road access in pre-event OSM, TRISHUL classifies it as **`no road access (pre-existing)`**, never as "flood cut-off".
- Responders are advised that community isolation refers strictly to vehicular road disconnection.

---

## 6. Regulatory & Operational Disclaimer
**Educational prototype, not an operational tool.**
TRISHUL was built for the Multimodal AI Hackathon 2026. It is not certified for life-safety dispatch, evacuation ordering, or official civil defense operations.
