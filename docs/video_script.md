# TRISHUL (त्रिशूल) — Demo Video Script & Storyboard
**Multimodal AI Hackathon 2026 · Track B: Mapping Flood Damage from Space**  
**Target Duration:** 3:30 (Range: 3:00 – 3:45) · **Format:** 1080p60 / English Voiceover + Subtitles

---

## Storyboard Overview & Timing Budget

| Beat | Timestamp | Section Title | Visual Focus | Target Seconds |
|:---:|:---:|:---|:---|:---:|
| **1** | 0:00 – 0:20 | **The Hook** | Trishuli Valley satellite view, monsoon clouds, the crisis | 20s |
| **2** | 0:20 – 0:55 | **Live Judge Mode on Unseen Valley** | Terminal CLI + live browser run on arbitrary AOI polygon | 35s |
| **3** | 0:55 – 1:40 | **Prong 1: Hazard Swipe & Evidence** | MapLibre swipe, SAR shadow gate, thalweg profile, debris | 45s |
| **4** | 1:40 – 2:20 | **Prongs 2 & 3: Exposure & Isolation** | Sub-pixel bridge noisy-OR, Monte-Carlo percolation, restore CSV | 40s |
| **5** | 2:20 – 2:50 | **AI Component: GeoSAR-UNet & Adapt** | Architecture, EVENT-ADAPT cross-teaching, ablation ladder | 30s |
| **6** | 2:50 – 3:15 | **Validation & Negative Control** | EMSR927 honest comparison, dry-season placebo test | 25s |
| **7** | 3:15 – 3:30 | **Grounded Copilot & Mandatory Credits**| Bilingual Q&A, PDF reports, compliance manifest & credits | 15s |

---

## Detailed Beat-by-Beat Script

### Beat 1: The Hook — The Himalayan Dilemma (0:00 – 0:20)
**Visual:**  
- Opening cinematic push-in on 3D Copernicus DEM of the high Himalayas (Trishuli / Bhote Koshi canyon).  
- Split visual: Monsoon cloud bank covering optical sensors on the left; chaotic radar noise on the right.  
- Bold on-screen text: *"26 August 2026: 72 km Himalayan Valley Hit by Catastrophic Debris Flood"*.  

**Voiceover (Speaker):**  
> *"On the morning of August 26, 2026, a massive glacier collapse sent a wall of boulders and mud down Nepal's Trishuli corridor. For rescue coordinators, standard satellite mapping fails immediately: radar shadows on steep cliffs look identical to water, rocky debris defies water thresholds, and clouds obscure optical sensors. Responders need to know: Where did the flood hit? What was damaged? And who is cut off?"*

---

### Beat 2: Live "Judge Mode" on an Unseen Valley (0:20 – 0:55)
**Visual:**  
- Screen recording of terminal terminal showing `make judge` with an arbitrary bounding box in an unseen valley (e.g., Marsyangdi or Melamchi).  
- Live command execution:
  ```bash
  python -m trishul.pipeline --aoi '{"type":"Polygon",...}' --event-date 2026-08-26 --output runs/live_eval
  ```
- Fast-forward progress log showing compliance firewall check: `HostAllowlist: PASS`, `Sentinel-1 same-track pair: DESCENDING Track 19`, `GLO-30 Ray-march: Complete`, `GeoSAR-UNet ONNX: 1.8s`, `Monte-Carlo 200 runs: Complete`.  
- Smooth cut to Web Dashboard at `http://localhost:8000` rendering the freshly computed layers.

**Voiceover (Speaker):**  
> *"TRISHUL is not a pre-rendered static mock. Built for live judging, any coordinate bounding box and date in the Himalayas executes end-to-end in seconds. All satellite data passes through an automated compliance firewall: raw Sentinel-1 radar, pre-event OpenStreetMap locked strictly before the disaster date, and Copernicus DEM elevation. In under two minutes on CPU, TRISHUL downloads, processes, and outputs complete multi-layer situational awareness."*

---

### Beat 3: Prong 1 — Hazard Swipe & The Evidence Drawer (0:55 – 1:40)
**Visual:**  
- Interactive MapLibre GL UI in browser.  
- Cursor drags the interactive swipe bar: Left side shows raw Sentinel-1 log-ratio; Right side shows TRISHUL’s Physics-Gated Hazard map.  
- Click on **Evidence Drawer**:
  - Toggle 1: **Ray-marched Blindspot Map** (highlighting yellow/red steep shadow zones on East-facing canyon walls).  
  - Toggle 2: **Hydraulic Stage Profile** showing water elevation constrained monotonically along the river thalweg.  
  - Toggle 3: **EVENT-ADAPT Debris Layer** (43.26 km² boulder and mud deposits displayed in warm amber).  

**Voiceover (Speaker):**  
> *"Prong 1: Where did it hit? Look at this canyon wall. A naive SAR log-ratio flags steep ridges as flooded because radar shadow produces zero backscatter. TRISHUL's 3D ray-marching gate computes terrain occlusion directly from satellite orbit vectors, creating an honest Blindspot Map instead of false alarms. Furthermore, our hydraulic layer enforces thalweg consistency—water cannot float 50 meters above the canyon floor. And with EVENT-ADAPT, we separate 3.2 square kilometers of ponded water from 43.3 square kilometers of destructive rock debris."*

---

### Beat 4: Prongs 2 & 3 — Exposure & Cut-Off Settlements (1:40 – 2:20)
**Visual:**  
- Map zooms into Syapru Besi bridge crossing and valley road network.  
- Click on bridge icon: Modal popup displays **Sub-Pixel Bridge Model**:  
  - Deck overtopping probability ($P_{\text{overtop}}$)  
  - Left & Right bank approach scouring ($P_{\text{approach}}$)  
  - Overall failure probability ($P_{\text{fail}} = 0.94$).  
- Map transitions to Road Graph: Intact roads in green, washed out segments in red (21.14 km affected).  
- Click on isolated settlement markers: Modal shows *"Syapru Besi — $P(\text{cut-off}) = 0.88$ [90% CI: 0.82–0.94]"*.  
- Hover over remote monastery: Tooltip states: *"Remote Gompa — no road access (pre-existing)"*.  
- Click *"Download Restore Ranking"* (`restore_ranking.csv`).  

**Voiceover (Speaker):**  
> *"Prong 2: What was damaged? At 10-meter satellite resolution, bridges are invisible. TRISHUL evaluates bridges using a physical noisy-OR model: checking deck overtopping from the hydraulic stage, abutment washout on riverbanks, and radar scatter changes.  
> Prong 3: Who is cut off? Instead of naive shortest paths, we run 200 spatially correlated Monte-Carlo percolation draws across the road network. Settlements receive an exact isolation probability with 90% confidence intervals. Pre-existing trail communities are honestly marked 'pre-existing', never falsely blamed on the flood. And for engineering units, a greedy restore ranking identifies which road segments reconnect the most people first."*

---

### Beat 5: The AI Component — GeoSAR-UNet & EVENT-ADAPT (2:20 – 2:50)
**Visual:**  
- Architecture graphic: 10-channel U-Net input tensor (multi-temporal VV/VH, diffs, DEM, slope).  
- Overlay showing **HimalayaBench-mini** benchmark table:
  - Bhote Koshi: F1 0.83, IoU 0.71  
  - Melamchi: F1 0.80, IoU 0.67  
  - Beas River: F1 0.85, IoU 0.74  
  - Chamoli: F1 0.77, IoU 0.63  
  - Mean Himalayan F1: 0.813 | Mean IoU: 0.688.  
- Stepwise Ablation Bar Chart animating from baseline (F1 0.54, FA 18.5%) to Full TRISHUL (F1 0.86, FA 0.9%).  

**Voiceover (Speaker):**  
> *"Our primary AI component is GeoSAR-UNet, a 10-channel multimodal model pre-trained on the Kuro Siwo benchmark. Because global datasets lack labels for Himalayan rock debris, our innovation EVENT-ADAPT uses clear-sky optical teachers to train a SAR-terrain student within the event. Evaluated on four completely unseen Himalayan disasters across Nepal and India, GeoSAR-UNet achieves an average F1 of 0.81. Our ablation ladder proves that physics-gating and AI slash false-alarm rates from 18.5% down to just 0.9%."*

---

### Beat 6: Grounded Validation & Negative Control (2:50 – 3:15)
**Visual:**  
- Split comparison: TRISHUL hazard footprint vs. Copernicus EMS EMSR927 activation vector.  
- Callout stats: *Flood Extent IoU: 0.69 | F1: 0.81 | Building hits: 21 / 24*.  
- Callout on canyon bend: Zoom into area where TRISHUL identified debris scour under cloud cover where CEMS was obscured.  
- Negative control slide: *Dry-Season Placebo Test (2026-01-15) — False Alarm Rate: 0.15%, 0 building hits*.  

**Voiceover (Speaker):**  
> *"We validate against Copernicus EMS activation EMSR927. TRISHUL achieves 0.81 F1 agreement and catches 21 of 24 damaged structures. Where we disagree, TRISHUL correctly maps upstream gorge debris where optical sensors were blinded by clouds. And on a quiet dry-season placebo test, our false-alarm rate is just 0.15% with zero false building hits."*

---

### Beat 7: Grounded Copilot, 1-Page Report & Mandatory Credits (3:15 – 3:30)
**Visual:**  
- UI showing **Rescuer Copilot**: Typing question *"कुन बस्तीहरू सडक सम्पर्कबाट विच्छेद भएका छन्?"* (Which settlements are cut off?). Instant response in Nepali and English with tokens matching `facts.json`.  
- Preview of generated `report_en.pdf` and `report_ne.pdf`.  
- Footer showing cryptographic `manifest.json` badge: `COMPLIANCE: PASS`.  
- Full-screen Final Attribution Card:  
  - *"Contains modified Copernicus Sentinel data 2026."*  
  - *"Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved."*  
  - *"© OpenStreetMap contributors."*  
  - *"European Union, Copernicus Emergency Management Service data"*  
  - *"Educational prototype, not an operational tool. No imagery of victims."*  
  - GitHub Repo: `https://github.com/HirthikBalaji/trishul`  

**Voiceover (Speaker):**  
> *"Finally, every metric in our one-page bilingual situation reports and Rescuer Copilot is mathematically verified against the Claim Ledger—zero invented numbers. TRISHUL delivers transparent, physics-gated satellite intelligence when Himalayan communities need it most. Thank you."*
