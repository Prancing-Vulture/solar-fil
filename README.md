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
- **Complex Solar Background**: Prominences, sunspots, and active solar region background noise.
- **Multi-Annotator Instance Segmentation**: Overlapping filament annotations requiring instance-level isolation and lossless encoding.

This repository provides an advanced 5-Level Attention U-Net architecture, pos-weighted loss formulation, live web dashboard monitoring, and strict Kaggle RLE submission generation.

---

## 🧠 Models, Approaches & Technical Innovations

### 1. Neural Network Architecture: 5-Level Deep Attention U-Net
- **Model**: `SolarFilamentAttentionUNet` ([`code/model.py`](file:///c:/Users/Harreesh/projects/solarfil/code/model.py))
- **Channel Progression**: 5 resolution scales ($32 \rightarrow 64 \rightarrow 128 \rightarrow 256 \rightarrow 512 \rightarrow 1024$ channels).
- **Attention Gates**: Integrated into skip connections to compute spatial attention weight maps $a \in [0, 1]$. This suppresses background solar disk noise while selectively passing fine filament boundary features to the decoder.
- **Residual Conv Blocks**: Double convolutional layers with residual shortcut connections to eliminate vanishing gradients during deep feature extraction.
- **SiLU (Swish) Activations**: Replaced standard ReLU with SiLU ($x \cdot \sigma(x)$) for smooth non-linear gradient propagation.

### 2. High-Resolution Spatial Processing & Data Augmentation
- **Training Resolution**: Trained at $512 \times 512$ resolution ($4\times$ spatial area preservation compared to standard $256 \times 256$ baselines), preserving fine filament boundaries without spatial aliasing.
- **Augmentation Pipeline** ([`code/dataset.py`](file:///c:/Users/Harreesh/projects/solarfil/code/dataset.py)):
  - Random Horizontal Flips ($p = 0.5$)
  - Random Vertical Flips ($p = 0.5$)
  - Random 90° Rotations ($p = 0.5$)

### 3. Loss Formulation for Sparse Foreground Segmentation
- **Loss Function** ([`code/loss_and_metrics.py`](file:///c:/Users/Harreesh/projects/solarfil/code/loss_and_metrics.py)):
  $$\text{Loss}_{\text{total}} = 0.4 \cdot \text{BCEWithLogitsLoss}(\text{pos\_weight} = 5.0) + 0.6 \cdot \text{SoftDiceLoss}$$
- **Pos-Weighting (`pos_weight = 5.0`)**: Gives $5\times$ higher gradient emphasis to foreground filament pixels during BCE calculation, overcoming severe class imbalance.
- **Soft Dice Loss**: Directly optimizes Sørensen–Dice overlap coefficient on sigmoid probability maps.

### 4. Post-Processing & Connected Component Instance Extraction
- **Connected Components** ([`code/generate_submission.py`](file:///c:/Users/Harreesh/projects/solarfil/code/generate_submission.py)):
  - Employs 8-connectivity component analysis (`cv2.connectedComponentsWithStats`) to separate disconnected filament instances per solar image.
- **Artifact Filtering**: Suppresses small noise artifacts below `min_area_pixels = 30`.
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
| **01** | 0.5298 | 0.4310 | 0.0397 | 0.2910 | 0.3966 | 0.2487 | 💾 Saved Checkpoint |
| **04** | 0.1942 | 0.1965 | 0.0144 | 0.6402 | 0.6335 | 0.4656 | 💾 Saved Checkpoint |
| **09** | 0.1610 | 0.1772 | 0.0145 | 0.6990 | 0.6655 | 0.5010 | 💾 Baseline Best |
| **12** | **0.1605** | **0.1750** | **0.0142** | **0.7020** | **0.6804** | **0.5167** | 🏆 **ATTENTION UNET BEST** |
| **15** | 0.2003 | 0.2139 | 0.0380 | 0.6984 | 0.6773 | 0.5167 | Logged |

- **Best Validation Dice Score**: **`0.6804`**
- **Best Validation IoU Index**: **`0.5167`**
- **Total Test Predictions Generated**: **1,977** filament instances across 180 test images

---

## 📁 Repository Layout

```text
solarfil/
├── code/
│   ├── analysis.py              # Competition dataset path resolution
│   ├── probe_dataset.py         # Probing utility for COCO JSON & images
│   ├── dataset.py               # SolarFilamentDataset & data augmentations
│   ├── model.py                 # 5-Level SolarFilamentAttentionUNet architecture
│   ├── loss_and_metrics.py      # Weighted BCE + Soft Dice loss & metrics
│   ├── displayer.py             # Live Flask web dashboard & real-time plotter
│   ├── train.py                 # GPU training loop with Cosine Annealing
│   ├── infer.py                 # Test set inference & visualization renderer
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
│   ├── submission.csv           # Kaggle-compliant CSV submission file (318 KB)
│   └── submission.zip           # Upload-ready zip archive (106 KB)
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

### 2. Train High-Resolution Attention U-Net
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
- **Null Count**: `0` null values across all 1,977 prediction rows.

---

## 🛡️ License

Dataset annotations and codebase are distributed under the **CC BY-NC 4.0** license.
