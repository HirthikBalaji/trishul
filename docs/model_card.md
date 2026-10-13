# Model Card — GeoSAR-UNet (v1.0.0)

## Model Overview
- **Name:** GeoSAR-UNet
- **Task:** Multimodal Flood Water & Inundation Semantic Segmentation
- **Architecture:** 10-Channel U-Net with ResNet/ConvNeXt feature extractors
- **Framework:** PyTorch & ONNX Runtime (CPU / CUDA / Metal)
- **Input Dimensions:** `(Batch, 10, Height, Width)`
- **Output Classes:**
  - `0`: No Water / Stable Terrain
  - `1`: Permanent Water Channel
  - `2`: Flood Water Inundation

---

## 1. Input Channel Order & Normalization
The input tensor conforms to the strict Kuro Siwo multimodal format:
1. `pre2_vv`: Temporal pre-event 2 VV backscatter (dB, normalized)
2. `pre2_vh`: Temporal pre-event 2 VH backscatter (dB, normalized)
3. `pre1_vv`: Temporal pre-event 1 VV backscatter (dB, normalized)
4. `pre1_vh`: Temporal pre-event 1 VH backscatter (dB, normalized)
5. `post_vv`: Post-event VV backscatter (dB, normalized)
6. `post_vh`: Post-event VH backscatter (dB, normalized)
7. `diff_vv`: Backscatter difference $\Delta\sigma^0_{VV} = \sigma^0_{post} - \text{mean}(\sigma^0_{pre})$
8. `diff_vh`: Backscatter difference $\Delta\sigma^0_{VH} = \sigma^0_{post} - \text{mean}(\sigma^0_{pre})$
9. `dem_norm`: Copernicus GLO-30 elevation relative to local mean, scaled by 1/1000
10. `slope_norm`: Terrain slope in degrees, scaled by 1/50

---

## 2. Training Data & Compliance
- **Allowed Dataset:** Kuro Siwo Benchmark (Bountos et al., NeurIPS 2024 Datasets & Benchmarks Track).
- **Split Rule:** Strictly split by **held-out geographic events** (never random spatial patches) to test out-of-domain geographic generalization.
- **Forbidden Data:** No Copernicus EMS maps, no UNOSAT maps, and no post-event OSM edits were used during training or inference.

---

## 3. Loss Function & Optimization
- **Loss:** Class-weighted Cross-Entropy + Lovász-Softmax / Multi-class Dice Loss:
  $$\mathcal{L} = \mathcal{L}_{\text{CE}} + 0.5 \cdot \mathcal{L}_{\text{Lovász}}$$
- **Optimizer:** AdamW with weight decay $10^{-4}$
- **Learning Rate:** Cosine Annealing with warmup ($10^{-4} \to 10^{-6}$)
- **Augmentation:** Random flips, 90° rotations, simulated speckle noise injection, random dB calibration offset perturbations.

---

## 4. Inference & Tiling
- **Inference Engine:** ONNX Runtime (`CPUExecutionProvider`)
- **Tiling:** 224×224 spatial patches with 50% overlap (stride 112)
- **Window Blending:** 2D separable Hann window $w(x, y) = \sin^2(\frac{\pi x}{W}) \sin^2(\frac{\pi y}{H})$ to eliminate boundary discontinuity artifacts.
- **Latency:** < 3 seconds for a 30×30 km valley grid on an 8-core CPU.
