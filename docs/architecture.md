# TRISHUL (त्रिशूल) — Technical Architecture Specification

> **Physics-Gated Multimodal Flood Damage Mapping & Isolation Analysis for Steep Mountain Valleys**  
> Multimodal AI Hackathon 2026 · Track B: Mapping Flood Damage from Space

---

## 1. System Overview & The Core Thesis

Disaster response in steep Himalayan valleys (e.g., Trishuli, Melamchi, Chamoli) breaks standard satellite remote sensing methods:
1. **Radar Shadow and Layover:** Mountain topography casts geometric radar shadows that have near-zero backscatter, identically mimicking specular reflection from open water.
2. **Debris Floods vs. Clean Inundation:** Himalayan outburst floods transport millions of tons of boulders, sediment, and glacial debris. Instead of low radar backscatter from open water, debris deposits exhibit rough surfaces with **high backscatter**, defying conventional SAR thresholding.
3. **Monsoon Cloud Cover:** Optical imagery (Sentinel-2) is heavily obscured by clouds during the active monsoon season.
4. **Sub-Pixel Bridges & Critical Infrastructure:** Bridges spanning 20–80 m are narrow linear features undetectable within 10 m satellite resolution pixels.
5. **Deterministic Overconfidence:** Naive thresholding discards uncertainty, yielding binary maps that misinform tactical helicopter or rescue team deployments.

TRISHUL resolves these challenges through **Physics-Gated Evidence Fusion**: a modular pipeline that treats satellite data, hydraulic principles, and topographical models as probabilistic evidence streams unified through a calibrated evidence ledger.

```
                      ┌──────────────────────────────────────────────┐
                      │    Ingest & Compliance Firewall (§2)         │
                      │  CDSE S1/S2 · Copernicus DEM · ohsome OSM    │
                      └──────────────────────┬───────────────────────┘
                                             │
      ┌──────────────────────────────────────┼──────────────────────────────────────┐
      ▼                                      ▼                                      ▼
┌───────────────────────────┐  ┌───────────────────────────┐  ┌───────────────────────────┐
│     SAR Geometry Gate     │  │   Optical Evidence Engine │  │      Terrain Engine       │
│  • 3D Ray-Marching        │  │  • Dilation Masking (SCL) │  │  • D8 Flow Routing        │
│  • Shadow / Layover Mask  │  │  • MNDWI, NDVI, BSI, NDSI │  │  • Thalweg Stage Profile  │
│  • Local Incidence (θ_loc)│  │  • Reliability Map (r_opt)│  │  • HAND Hydrologic Prior  │
└─────────────┬─────────────┘  └─────────────┬─────────────┘  └─────────────┬─────────────┘
              │                              │                              │
              └──────────────────────┐       │       ┌──────────────────────┘
                                     ▼       ▼       ▼
                               ┌───────────────────────────┐
                               │   AI: GeoSAR-UNet v1.0    │
                               │  + EVENT-ADAPT Cross-Teach│
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Calibrated Evidence Fusion│
                               │  (Multi-track Log-Odds)   │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
    ┌────────────────────────────────────────┼────────────────────────────────────────┐
    ▼                                        ▼                                        ▼
┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
│     Prong 1: Hazard    │      │    Prong 2: Exposure   │      │   Prong 3: Isolation   │
│  • P(water), P(debris) │      │  • Building p_hit (CI) │      │  • Road-Graph MC Perc. │
│  • Blindspot Map       │      │  • Sub-pixel Bridge    │      │  • P(cut-off) [90% CI] │
│  • Thalweg Profile     │      │    Noisy-OR Model      │      │  • Restore Ranking CSV │
└────────────────────────┘      └────────────────────────┘      └────────────────────────┘
```

---

## 2. Ingestion & Compliance Firewall

To prevent any data contamination and guarantee evaluation integrity under strict competition rules:

1. **Host Allowlist Client (`trishul/ingest/http_allowlist.py`):**
   - Outbound HTTP requests from the pipeline are restricted to:
     - Copernicus Data Space Ecosystem (CDSE) STAC, OData, and S3 APIs
     - HeiGIT ohsome API (`api.ohsome.org`)
     - Copernicus DEM AWS Open Data bucket
     - Model weights host
   - Any attempt to reach unapproved domains (e.g., Copernicus EMS, UNOSAT, Overpass API, or Geofabrik latest extracts) raises an immediate `ComplianceViolationError`.

2. **Temporal Quarantine:**
   - Pre-event OpenStreetMap vectors are locked to snapshots timestamped **strictly before the event date** (`osm_snapshot_date < event_date`).
   - S1 and S2 scenes are checked against the event date to prevent post-event ground-truth contamination.

3. **Compliance Manifest (`manifest.json`):**
   - Every execution compiles a cryptographically signed manifest containing:
     - Dataset IDs, acquisition timestamps, and SHA-256 hashes
     - OSM snapshot timestamps
     - Model weight checksums
     - Software git commit hash
     - Compliance verdict (`PASS` or `FAIL` with verifiable audit log)

---

## 3. Mathematical Formulation

### 3.1 SAR Geometry Gate & Ray-Marching

Synthetic Aperture Radar side-looking geometry creates severe distortions over steep slopes. Let $\mathbf{s}(x, y, z)$ be the digital elevation model (GLO-30). Given satellite azimuth $\alpha_{\text{orbit}}$ and look angle $\theta_{\text{look}}$:

1. **Ray-Marching for Shadow Detection:**
   For an illumination vector directed towards the satellite antenna:
   $$\Delta x = \Delta s \cdot \sin(\phi_{\text{orbit}}), \quad \Delta y = \Delta s \cdot \cos(\phi_{\text{orbit}})$$
   $$\Delta z_{\text{ray}} = \Delta s \cdot \tan(90^\circ - \theta_{\text{look}})$$
   A pixel $p$ at elevation $z_p$ is in **radar shadow** if any terrain point along the line of sight casts an elevation higher than the ray:
   $$\text{Shadow}(p) = \mathbb{I}\left[\exists k > 0 : z(p + k\Delta\mathbf{r}) > z_p + k \Delta z_{\text{ray}}\right]$$

2. **Layover Condition:**
   Layover occurs when the terrain slope facing the radar exceeds the look angle:
   $$\theta_{\text{slope}} \cdot \cos(\psi_{\text{aspect}} - \phi_{\text{look}}) > \theta_{\text{look}}$$

3. **Geometric Reliability Weight:**
   $$r_{\text{geo}}(p) = \begin{cases} 
   0.0, & \text{if } \text{Shadow}(p) \lor \text{Layover}(p) \\
   \cos(\theta_{\text{loc}}(p)), & \text{otherwise}
   \end{cases}$$
   Where $\theta_{\text{loc}}$ is the local angle of incidence. Pixels where $r_{\text{geo}} = 0$ are diverted to the **Blindspot Map** with explicit reason codes (`shadow`, `layover`, `extreme_incidence`).

---

### 3.2 SAR Temporal-Statistics Change Detection

Single-pair log-ratio $\ln(\sigma^0_{\text{post}} / \sigma^0_{\text{pre}})$ suffers from speckle noise and seasonal soil moisture fluctuations. TRISHUL constructs a **multi-temporal pre-stack** of $N \ge 3$ same-track images:
$$\mu_{\text{pre}}(p) = \frac{1}{N} \sum_{k=1}^N \sigma^0_{k}(p), \quad s^2_{\text{pre}}(p) = \frac{1}{N-1}\sum_{k=1}^N \left(\sigma^0_{k}(p) - \mu_{\text{pre}}(p)\right)^2$$
The calibrated test statistic is:
$$z(p) = \frac{\sigma^0_{\text{post}}(p) - \mu_{\text{pre}}(p)}{\sqrt{s^2_{\text{pre}}(p) + \epsilon}}$$
A large negative $z_{VV}$ indicates specular water surface formation. A large positive $z_{VH}$ indicates rough debris accumulation.

---

### 3.3 AI Model: GeoSAR-UNet & EVENT-ADAPT

#### GeoSAR-UNet Architecture
GeoSAR-UNet is a 10-channel U-Net operating on $224 \times 224$ patches:
- **Channels 1–6:** Multi-temporal SAR backscatters ($\text{pre}_2, \text{pre}_1, \text{post}$ in VV & VH).
- **Channels 7–8:** Difference features ($\Delta\sigma^0_{VV}, \Delta\sigma^0_{VH}$).
- **Channels 9–10:** Normalized elevation and slope.
- **Tiled Inference:** 50% spatial overlap with 2D separable Hann window weighting:
  $$w(x, y) = \sin^2\left(\frac{\pi x}{W}\right) \cdot \sin^2\left(\frac{\pi y}{H}\right)$$
  eliminating boundary discontinuities.

#### EVENT-ADAPT: Optical-to-SAR Cross-Teaching
Because global benchmark datasets lack ground-truth labels for fresh rock debris, EVENT-ADAPT employs teacher-student cross-modal adaptation:
1. **Teacher:** High-confidence post-event optical indices (MNDWI, BSI) extract confident water and debris patches in clear sky regions.
2. **Student:** GeoSAR-UNet is fine-tuned to predict debris and water strictly from SAR backscatter differences and terrain slope.
3. **Cross-Validation:** Evaluated using spatial block cross-validation with minimum 1 km buffer separation to prevent spatial leakage.

---

### 3.4 Hydraulic Consistency & HAND Prior

Flood inundation in steep valleys must obey gravity and mass conservation:
1. **Thalweg Extraction:** Topographic valley thalweg (drainage axis) derived via D8 flow accumulation.
2. **Flood-Stage Profile ($h_{\text{flood}}(s)$):** For distance $s$ along the thalweg, maximum observed stage is extracted from cross-sections and smoothed monotonically.
3. **HAND Prior:** Height Above Nearest Drainage:
   $$P_{\text{hyd}}(p) = \frac{1}{1 + \exp\left(\frac{\text{HAND}(p) - h_{\text{flood}}(s_p)}{\sigma_{\text{hyd}}}\right)}$$
   Pixels located 50 meters above the riverbed cannot be flooded unless a physical damming event occurred.

---

### 3.5 Calibrated Log-Odds Evidence Fusion

Evidence from SAR change, optical change, GeoSAR-UNet predictions, and hydraulic priors are stacked in log-odds space:
$$L(p) = L_0 + r_{\text{sar}}(p) \cdot \ln\frac{P_{\text{sar}}(p)}{1 - P_{\text{sar}}(p)} + r_{\text{opt}}(p) \cdot \ln\frac{P_{\text{opt}}(p)}{1 - P_{\text{opt}}(p)} + r_{\text{ai}}(p) \cdot \ln\frac{P_{\text{ai}}(p)}{1 - P_{\text{ai}}(p)} + \ln\frac{P_{\text{hyd}}(p)}{1 - P_{\text{hyd}}(p)}$$
Final calibrated hazard probability:
$$P_{\text{hazard}}(p) = \frac{1}{1 + \exp(-L(p))}$$

---

### 3.6 Exposure Modeling & Sub-Pixel Bridge Noisy-OR Model

#### Building & Road Exposure
- **Buildings:** Polygon buffered by 15 m to account for GPS and geo-referencing errors. Mean and 90th percentile hazard probabilities define $p_{\text{hit}}$.
- **Road Segments:** Road geometries divided into 50 m segments; $p_{\text{hit}}$ computed per segment.

#### Sub-Pixel Bridge Noisy-OR Formulation
Bridges are narrow linear structures unresolvable in 10 m SAR. A bridge fails if it is overtopped, if its approach abutments are scoured, or if radar structural backscatter collapses:
$$P(\text{bridge failure}) = 1 - \prod_{m \in \{\text{overtop}, \text{approach}, \text{radar}\}} (1 - P_m)$$
Where:
- $P_{\text{overtop}} = \sigma\left(\frac{h_{\text{flood}} - h_{\text{deck}}}{\sigma_{\text{clearance}}}\right)$ (hydraulic deck overtopping)
- $P_{\text{approach}} = \max\left(P_{\text{hazard}}(\text{bank}_{\text{left}}), P_{\text{hazard}}(\text{bank}_{\text{right}})\right)$ (approach road washouts)
- $P_{\text{radar}} = \sigma\left(\frac{-z_{\text{bridge}}}{\sigma_{\text{sar}}}\right)$ (loss of double-bounce scatter)

---

### 3.7 Network Percolation & Isolation Analysis

To determine settlement isolation under uncertainty:
1. **Correlated Monte-Carlo Simulation:**
   A spatially correlated Gaussian Random Field $\eta(x)$ ($\ell = 500\text{ m}$) models correlated damage across nearby infrastructure.
   For each realization $k \in \{1, \dots, K\}$ ($K=200$):
   $$\text{Edge } e \text{ destroyed} \iff P_{\text{hit}}(e) + \eta(e) > \tau_k \sim \mathcal{U}(0, 1)$$
2. **Graph Reachability:**
   Using pre-event road topologies, reachability from each settlement $s$ to key hubs (hospitals, supply depots) is checked via Dijkstra search:
   $$P_{\text{cut-off}}(s) = \frac{1}{K} \sum_{k=1}^K \mathbb{I}\left[\text{Hospital is unreachable from } s \text{ in realization } k\right]$$
   We report $P(\text{cut-off})$ alongside 90% confidence intervals.
3. **Pre-Existing Disconnection Guarantee:**
   Settlements without pre-event road infrastructure are classified as **`no road access (pre-existing)`**, never as "flood cut-off".

4. **Greedy Restore-First Ranking:**
   Damaged road edges and bridges are prioritized based on marginal population reconnection:
   $$\text{Score}(e) = \frac{\sum_{s} \Delta P_{\text{connected}}(s \mid \text{restore } e)}{\text{Length}(e) \cdot \text{CostWeight}(e)}$$

---

## 4. Grounded Situation Report & Q&A Copilot

1. **Claim Ledger (`facts.json`):**
   Every quantitative metric produced across hazard, exposure, and network modules is assigned a unique identifier (e.g., `[[F:flood_water_km2]]`, `[[F:road_length_hit_km]]`).
2. **Template Synthesis & Verification:**
   Situation reports (English and Nepali) and Q&A answers are compiled strictly from ledger tokens.
3. **Adversarial Ledger Verifier (`trishul/report/verifier.py`):**
   An automated scanner parses generated text, flags any ungrounded numeric digits or spelled-out numbers, and blocks report export if a discrepancy is detected.

---

## 5. Software Architecture & Interfaces

- **Backend:** FastAPI REST API (`trishul/api/app.py`) providing endpoints:
  - `POST /api/run`: Initiate new end-to-end valley analysis
  - `GET /api/run/{id}/status`: Pipeline progress tracking
  - `GET /api/run/{id}/facts`: Grounded factual ledger
  - `GET /api/run/{id}/restore`: Ranked road restoration queue
  - `GET /api/run/{id}/flowpath`: Upstream flood trajectory
  - `POST /api/run/{id}/chat`: Grounded Q&A copilot
- **Frontend:** Interactive MapLibre GL dashboard (`web/index.html`) featuring:
  - Swipe view comparing optical and SAR layers
  - Blindspot map toggles
  - Color-coded road damage and bridge failure states
  - Isolated settlement markers with population stats
  - Interactive Rescuer Copilot with bilingual prompt chips
