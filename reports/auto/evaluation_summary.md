# TRISHUL — Automated Evaluation & Benchmark Report

## 1. HimalayaBench-mini (Held-out Generalization)
| Event | Valley | Event Date | Precision | Recall | F1 Score | IoU |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Bhote Koshi July 2025** | Trishuli / Bhote Koshi | 2025-07-08 | 0.84 | 0.82 | 0.83 | 0.71 |
| **Melamchi June 2021** | Helambu / Melamchi | 2021-06-15 | 0.81 | 0.79 | 0.80 | 0.67 |
| **Beas Flash Flood 2023** | Kullu / Beas River | 2023-07-10 | 0.86 | 0.84 | 0.85 | 0.74 |
| **Chamoli Avalanche 2021** | Rishi Ganga / Dhauliganga | 2021-02-07 | 0.79 | 0.76 | 0.77 | 0.63 |

## 2. Frozen EMSR927 Comparison
- **Dataset:** European Union, Copernicus Emergency Management Service data (EMSR927)
- **Flood Extent IoU:** 0.69 (±1-pixel tolerance)
- **Flood Extent F1:** 0.81
- **Building Hit Detection:** 21 / 24 structures

## 3. Placebo Test (False-Alarm Calibration)
- **Date:** 2026-01-15 (Quiet dry-season baseline)
- **False-Alarm Flood Area:** 0.08 km²
- **False-Alarm Building Hits:** 0
- **False-Alarm Rate:** 0.15% (Within nominal < 1.0% tolerance)
