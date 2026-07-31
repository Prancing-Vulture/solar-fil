# ☀️ Solar Filament Instance Segmentation — Kaggle 2026

An end-to-end deep learning framework, real-time metrics displayer, and Kaggle submission generator built for the **MAGFiLO 1.0 Solar Filament Instance Segmentation Competition**.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch GPU](https://img.shields.io/badge/PyTorch-CUDA%2012-orange.svg)](https://pytorch.org/)
[![Pixi Package Manager](https://img.shields.io/badge/pixi-managed-green.svg)](https://pixi.sh/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

---

## 📌 Executive Overview

Solar filaments are dense, cool magnetic plasma structures suspended in the hot solar corona, best observed in **H-Alpha GONG** (Global Oscillations Network Group) solar disk observations.

Segmenting solar filaments presents unique computer vision challenges:
- **Thin, Sparse Structures**: Filaments occupy less than **0.4%** of the solar disk area.
- **Complex Solar Background**: Prominences, sunspots, active solar region background noise, and limb darkening.
- **Multi-Annotator Instance Segmentation**: Overlapping filament annotations requiring instance-level isolation and lossless encoding.

This repository provides a pretrained transfer learning architecture with U-Net, Contrast-Limited Adaptive Histogram Equalization (CLAHE) preprocessing, data-driven post-processing filtering, live web dashboard monitoring, and strict Kaggle RLE submission generation.

---

## 🧠 Models, Approaches & Technical Innovations

### 1. Neural Network Architecture: Pretrained U-Net Transfer Learning
- **Model**: `SolarFilamentAttentionUNet` ([`code/model.py`](file:///c:/Users/Harreesh/projects/solarfil/code/model.py))
- **Backbone**: **ResNet-34** encoder pre-trained on ImageNet.
- **Framework**: Wraps `segmentation-models-pytorch` (SMP) to adapt pretrained spatial feature extractors for 1-channel grayscale solar imagery.
- **Benefits**: Fine-tuning a pretrained model converges significantly faster (15 epochs) and generalizes better than training a model from scratch, especially given the small 707-image dataset.

### 2. High-Resolution Spatial Processing & Data Preprocessing
- **Local Contrast Enhancement (CLAHE)**: Applies CLAHE (`clipLimit=2.0`, `tileGridSize=(8, 8)`) to resolve varying global contrast and limb-darkening effects.
- **Training Resolution**: Trained at $512 \times 512$ resolution ($4\times$ spatial area preservation compared to standard $256 \times 256$ baselines), preserving fine filament boundaries without spatial aliasing.
- **Augmentation Pipeline** ([`code/dataset.py`](file:///c:/Users/Harreesh/projects/solarfil/code/dataset.py)):
  - Random Horizontal and Vertical Flips ($p = 0.5$)
  - Random 90° Rotations ($p = 0.5$)
  - **Affine Transforms (ShiftScaleRotate)**: Simulates alignment, orientation, and scale shifts.
  - **Photometric Variation**: Random brightness, contrast scaling, and high-frequency Gaussian noise.

### 3. Loss Formulation for Sparse Foreground Segmentation
- **Loss Function** ([`code/loss_and_metrics.py`](file:///c:/Users/Harreesh/projects/solarfil/code/loss_and_metrics.py)):
  $$\text{Loss}_{\text{total}} = 0.4 \cdot \text{BCEWithLogitsLoss}(\text{pos\_weight} = 5.0) + 0.6 \cdot \text{SoftDiceLoss}$$
- **Pos-Weighting (`pos_weight = 5.0`)**: Gives $5\times$ higher gradient emphasis to foreground filament pixels during BCE calculation, overcoming severe class imbalance.
- **Soft Dice Loss**: Directly optimizes Sørensen–Dice overlap coefficient on sigmoid probability maps.

### 4. Post-Processing & Data-Driven Noise Filtering
- **Connected Components** ([`code/generate_submission.py`](file:///c:/Users/Harreesh/projects/solarfil/code/generate_submission.py)):
  - Employs 8-connectivity component analysis (`cv2.connectedComponentsWithStats`) to separate disconnected filament instances per solar image.
- **Data-Driven Filtering**: Suppresses small noise artifacts below `min_area_pixels = 300` pixels at full $2048 \times 2048$ resolution. Annotation analysis shows the 5th percentile of real filament area is **316.9 pixels**, making `300` the ideal threshold to filter false-positive background dots without dropping true filaments.
- **Lossless COCO RLE Encoding**: Encodes binary instance masks into column-major (Fortran-contiguous) COCO ASCII RLE strings via `pycocotools.mask.encode`.

### 5. Real-Time Live Displayer & Web Dashboard
- **Web Dashboard** ([`code/displayer.py`](file:///c:/Users/Harreesh/projects/solarfil/code/displayer.py)): Spawns a background Flask web server at `http://localhost:5000` auto-refreshing every 3 seconds.
- **Live Visual Plot**: Updates `live_display/live_epoch_display.png` every epoch displaying:
  - Input Solar Image
  - Ground Truth Mask
  - **Live Green Filament Segmentation Overlay**
  - Loss Curves (Train Loss vs Val Loss & BCE Loss)
  - Metric Curves (Train Dice vs Val Dice & IoU)
  - Real-time Terminal Log Box & Metrics Table

---

## 📈 Model Performance & Validation Progression

| Epoch | Train Combined Loss | Val Combined Loss | **Val BCE Logits Loss** | Train Dice Score | **Val Dice Score** | **Val IoU (Jaccard)** | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **01** | 0.5360 | 0.3082 | 0.0415 | 0.3212 | 0.5651 | 0.3973 | 💾 Saved Checkpoint |
| **03** | 0.3055 | 0.2457 | 0.0408 | 0.5469 | 0.6300 | 0.4645 | 💾 Saved Checkpoint |
| **05** | 0.2854 | 0.2313 | 0.0395 | 0.5758 | 0.6516 | 0.4878 | 💾 Saved Checkpoint |
| **08** | 0.2824 | 0.2314 | 0.0448 | 0.5784 | 0.6523 | 0.4884 | 💾 Saved Checkpoint |
| **10** | 0.2730 | 0.2301 | 0.0423 | 0.5928 | 0.6533 | 0.4900 | 💾 Saved Checkpoint |
| **13** | **0.2636** | **0.2215** | **0.0386** | **0.6067** | **0.6654** | **0.5032** | 🏆 **PRETRAINED BEST** |
| **15** | 0.2631 | 0.2219 | 0.0390 | 0.6071 | 0.6647 | 0.5025 | Logged |

- **Best Validation Dice Score**: **`0.6654`** (on CLAHE contrast-enhanced images)
- **Best Validation IoU Index**: **`0.5032`**
- **Kaggle Public Leaderboard Progression**:
  - Baseline (Scratch U-Net): **`0.59`**
  - **Optimized (Pretrained + CLAHE + Area Filter)**: **`0.62`**
- **Total Test Predictions Generated**: **1,345** filament instances across 180 test images (average of **7.47 per image**, mirroring the true training annotation rate of **7.10 per image**)

---

## 📁 Repository Layout

```text
solarfil/
├── code/
│   ├── analysis.py              # Competition dataset path resolution
│   ├── probe_dataset.py         # Probing utility for COCO JSON & images
│   ├── dataset.py               # SolarFilamentDataset with CLAHE & Albumentations
│   ├── model.py                 # Pretrained ResNet-34 U-Net (SMP wrapper)
│   ├── loss_and_metrics.py      # Weighted BCE + Soft Dice loss & metrics
│   ├── displayer.py             # Live Flask web dashboard & real-time plotter
│   ├── train.py                 # GPU training loop with Cosine Annealing (lr=2e-4)
│   ├── infer.py                 # Test set inference & visualization renderer
│   ├── test_smp_setup.py        # Verification utility for environment imports
│   └── generate_submission.py   # Kaggle 2-column RLE submission generator
├── checkpoints/
│   └── best_model.pth           # Trained model weights at peak validation epoch
├── live_display/
│   └── live_epoch_display.png   # Real-time epoch visualization plot
├── logs/
│   ├── training_metrics.csv     # Timestamped CSV metrics history
│   ├── training.log             # Text log file with milestone events
│   └── training_metrics.json    # JSON metric history dictionary
├── submission_output/
│   ├── submission.csv           # Kaggle-compliant CSV submission file
│   └── submission.zip           # Upload-ready zip archive
├── submission.csv               # Root submission file for convenience
├── pixi.toml                    # Environment dependencies & lock file
└── README.md                    # Project documentation
```

---

## 🚀 Commands Reference

### 1. Probe Dataset Structure
```bash
pixi run python code/probe_dataset.py
```

### 2. Train Pretrained U-Net with Live Displayer
```bash
pixi run python code/train.py
```

### 3. Run Test Set Inference
```bash
pixi run python code/infer.py
```

### 4. Generate Kaggle Submission Package
```bash
pixi run python code/generate_submission.py
```

---

## 📋 Kaggle Submission File Specification

The generated submission file [`submission.csv`](file:///c:/Users/Harreesh/projects/solarfil/submission.csv) and [`submission_output/submission.zip`](file:///c:/Users/Harreesh/projects/solarfil/submission_output/submission.zip) strictly comply with the Kaggle evaluation schema:

- **Columns**: `['filament_id', 'segmentation_rle']`
- **Naming Rule**: `filament_id` formatted as `{image_name}_{count}` (e.g., `20110120105534Ch_1`).
- **Encoding**: COCO ASCII RLE counts string without quotes.
- **Fallback**: Images without detected filaments receive an empty background mask encoding (`{image_name}_1`).
- **Null Count**: `0` null values across all 1,345 prediction rows.

---

## 🛡️ License

Dataset annotations and codebase are distributed under the **CC BY-NC 4.0** license.
