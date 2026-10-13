| Step | Configuration | Himalayan IoU | Himalayan F1 | False-Alarm % | Marginal ΔF1 |
|:---:|:---|:---:|:---:|:---:|:---:|
| 1 | **Otsu threshold on single SAR log-ratio** | 0.42 | 0.54 | 18.5% | Baseline |
| 2 | **+ Temporal-statistics pre-stack (F-test)** | 0.51 | 0.63 | 11.2% | +0.09 |
| 3 | **+ Ray-marched SAR geometry gate (Blindspot map)** | 0.58 | 0.70 | 4.3% | +0.07 |
| 4 | **+ Multi-track ascending/descending fusion** | 0.63 | 0.74 | 3.1% | +0.04 |
| 5 | **+ GeoSAR-UNet (Kuro Siwo pre-trained)** | 0.68 | 0.78 | 2.5% | +0.04 |
| 6 | **+ Hydraulic consistency layer (HAND thalweg stage)** | 0.73 | 0.82 | 1.4% | +0.04 |
| 7 | **+ EVENT-ADAPT optical-to-SAR student (Full System)** | 0.79 | 0.86 | 0.9% | +0.04 |
