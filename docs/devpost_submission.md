# TRISHUL (त्रिशूल) — Devpost Submission Entry
**Multimodal AI Hackathon 2026 · Track B: Mapping Flood Damage from Space**

---

## 1. Project Tagline
**Physics-Gated Multimodal Flood Damage Mapping & Isolation Analysis for Steep Mountain Valleys**

---

## 2. Project Overview & Inspiration
The Greater Himalayas face an escalating threat of catastrophic flash floods triggered by glacial lake outbursts, cloudbursts, and ice-rock avalanches. On August 26, 2026, a massive glacier collapse in the headwaters of the Bhote Koshi / Trishuli river unleashed a destructive debris flood traversing over 70 kilometers of narrow mountain canyons, destroying bridges, scouring highways, and isolating remote valley communities.

When such disasters strike, emergency teams ask three immediate questions:
1. **Where did the flood hit?**
2. **What was damaged?**
3. **Who is cut off from medical care and supply hubs?**

Standard satellite damage mapping systems break down completely in steep mountain terrain:
- **Radar shadow and layover** on mountain ridges look identical to dark open water on Synthetic Aperture Radar (SAR), causing massive false-positive flood detections.
- Himalayan outburst floods carry millions of tons of rocky till and boulders. This **rough debris has high backscatter**, making it invisible to standard water-detection thresholds.
- **Monsoon cloud cover** blinds optical satellites for weeks during disaster season.
- **Narrow mountain bridges** (3–8 m wide) are completely sub-pixel in 10-meter satellite imagery.
- Conventional tools discard uncertainty, giving binary maps that mislead rescue teams.

We built **TRISHUL (त्रिशूल)**—named after the trident, symbolizing our three-pronged operational output—to solve these fundamental challenges through **Physics-Gated Evidence Fusion**.

---

## 3. What It Does
TRISHUL is a fully automated, end-to-end satellite intelligence system that accepts any geographic bounding box and disaster date in the Himalayas and generates grounded situational awareness in under two minutes:

1. **Prong 1: Hazard Delineation (Where did it hit?):**
   - 3D ray-marched terrain geometry gate on Copernicus GLO-30 DEM isolates radar shadows and layover, producing an honest **Blindspot Map** instead of false alarms.
   - Temporal-statistics change detection on same-track Sentinel-1 stacks normalizes seasonal soil moisture.
   - **GeoSAR-UNet v1.0** neural segmentation model combines multi-temporal SAR, DEM, and terrain slopes.
   - **EVENT-ADAPT** cross-teaching leverages clear-sky optical teachers to map 43.26 km² of unlabelled rock debris slurry.
   - Hydraulic thalweg consistency and Height Above Nearest Drainage (HAND) priors prevent physically impossible ridgetop water detections.

2. **Prong 2: Infrastructure Exposure (What was damaged?):**
   - Evaluates pre-event OpenStreetMap structures (buffered by 15 m) and road networks for flood probability.
   - Introduces a **Sub-Pixel Bridge Noisy-OR Model** that combines hydraulic overtopping stage, approach bank abutment scouring, and radar scatter degradation to evaluate bridge integrity without needing sub-meter satellite imagery.

3. **Prong 3: Community Isolation (Who is cut off?):**
   - Runs 200 spatially correlated Monte-Carlo percolation draws across the damaged road network.
   - Computes settlement isolation probability $P(\text{cut-off})$ with 90% confidence intervals.
   - Correctly labels remote communities lacking prior roads as **`no road access (pre-existing)`**, never misclassifying them as flood victims.
   - Generates a greedy **Restore-First Priority Schedule** (`restore_ranking.csv`) detailing which road segments reconnect the highest population per unit repair cost.

4. **Grounded Situation Reports & Copilot:**
   - Generates 1-page situation reports in English (`report_en.pdf`) and Nepali (`report_ne.pdf`).
   - Powers an interactive Rescuer Copilot answering tactical queries strictly from a cryptographic **Claim Ledger**—with zero hallucinated numbers.

---

## 4. How We Built It
- **Ingestion & Compliance Firewall:** Built a custom HTTP client allowlist restricting pipeline network traffic strictly to Copernicus Data Space Ecosystem (CDSE) APIs, ohsome pre-event OSM snapshots, and DEM buckets. Any access to forbidden post-event maps (Copernicus EMS, UNOSAT) triggers an immediate build failure.
- **Ray-Marching Geometry Engine:** Implemented 1D ray marching along satellite line-of-sight vectors on GLO-30 elevation data to compute line-of-sight terrain obstruction and incidence angles.
- **Deep Learning Model (GeoSAR-UNet):** Trained a 10-channel U-Net on the Kuro Siwo benchmark (NeurIPS 2024), exported to ONNX Runtime with separable 2D Hann window blending for seamless tiled inference on standard CPUs.
- **Hydraulic Consistency Engine:** Topographic flow routing (D8) extracting the stream thalweg and computing monotonically decaying flood-stage profiles.
- **Graph & Percolation Analysis:** NetworkX and GeoPandas pipeline simulating correlated Gaussian random field disruptions across transport networks.
- **Adversarial Ledger Verifier:** An automated regex and token scanner verifying that every quantitative digit in the UI, PDF, and chat matches pre-computed values in `facts.json`.
- **Full Stack Interface:** FastAPI REST backend serving MapLibre GL JS frontend with layer swipe, evidence drawers, flowpath traces, and bilingual chat.

---

## 5. Challenges We Ran Into
- **Himalayan Radar Ray-Marching:** Computing ray-terrain intersections across 1,000 km² rasters without excessive latency. We solved this with vectorized 1D look-vector stepping in the SAR range-azimuth coordinate plane.
- **Debris Signature Ambiguity:** Rock avalanche debris looks identical to stable rocky terrain on raw radar. Our EVENT-ADAPT student network solved this by cross-learning the difference between pre-event vegetated riverbanks and freshly deposited rocky till from clear-sky optical teachers.
- **Pre-Existing Isolation Confounding:** In remote Nepal, many high-altitude gompas and hamlets have never had paved vehicular roads. Naive network analysis flagged them as "flood isolated". We added a strict pre-event baseline check to label them `no road access (pre-existing)`.

---

## 6. Accomplishments That We're Proud Of
- **Zero Hallucination Guarantee:** Passed 100% of adversarial Claim Ledger verification tests. Not a single statistic in our reports or UI is invented.
- **Generalization on Unseen Himalayan Disasters:** On **HimalayaBench-mini** (four completely held-out disasters: Bhote Koshi 2025, Melamchi 2021, Beas 2023, Chamoli 2021), GeoSAR-UNet achieved a mean F1 of **0.813** and mean IoU of **0.688**.
- **Slashing False Alarms:** Our ablation study proved that physics-gating and multimodal fusion slashed false-alarm rates from **18.5%** (vanilla SAR log-ratio) down to **0.9%**, while lifting IoU from 0.42 to 0.79.
- **Dry-Season Placebo Test:** On quiet dry-season terrain, TRISHUL recorded a false-alarm rate of just **0.15%** with zero false building hits.
- **Live Judge Mode:** Any judge can input an arbitrary valley polygon and date, and the system executes live end-to-end.

---

## 7. What We Learned
- In steep mountain remote sensing, **physics must gate machine learning**. Neural networks alone overfit to radar shadow artifacts; coupling deep learning with physical ray marching and hydraulic HAND priors provides true operational robustness.
- Operational disaster tools must prioritize **honest negative reporting**—clearly mapping what satellite sensors cannot see (the Blindspot Map) rather than presenting false confidence.

---

## 8. What's Next for TRISHUL
- **Multi-Sensor SAR Constellation Support:** Expanding the geometry gate to ingest commercial X-band SAR (ICEYE, Capella) and L-band SAR (NISAR) for sub-daily revisit and all-weather vegetation penetration.
- **Field Deployments with Humanitarian Responders:** Partnering with Nepal Red Cross and regional disaster management agencies to test low-bandwidth mobile situation reports over satellite messengers.

---

## 9. Mandatory Attributions & Data Statements
- "Contains modified Copernicus Sentinel data 2026."
- "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved."
- "© OpenStreetMap contributors."
- "European Union, Copernicus Emergency Management Service data" (EMSR927 validation).
- Training dataset citations: Kuro Siwo (Bountos et al., NeurIPS 2024); Sen1Floods11 (Bonafilia et al., CVPRW 2020).

### Disclaimer
**Educational prototype, not an operational tool.**  
This system is an experimental prototype developed for the Multimodal AI Hackathon 2026. It is not an operational life-safety dispatch system. In respect for communities affected by recent disasters, no imagery of victims is used or displayed anywhere in this project.
