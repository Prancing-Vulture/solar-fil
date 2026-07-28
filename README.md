# ☀️ Solar Filament Instance Segmentation — Kaggle 2026

An end-to-end deep learning framework, real-time live metrics displayer, and Kaggle submission generator built for the **MAGFiLO 1.0 Solar Filament Segmentation Competition**.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch GPU](https://img.shields.io/badge/PyTorch-CUDA%2012-orange.svg)](https://pytorch.org/)
[![Pixi Package Manager](https://img.shields.io/badge/pixi-managed-green.svg)](https://pixi.sh/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

---

## 📌 Project Overview

Solar filaments are dense, cool magnetic structures floating in the solar corona, best visible in **H-Alpha GONG** (Global Oscillations Network Group) solar observations. 

This repository implements a complete deep learning workflow to segment solar filaments from $2048 \times 2048$ 8-bit grayscale solar disk images. Ground truth annotations are stored in COCO-style format with polygon paths, bounding boxes, areas, and RLE encodings.

### Key Highlights
- 🧠 **Residual U-Net Architecture (`SolarFilamentUNet`)**: 4-scale encoder-decoder network with skip connections and residual convolutions (8.1M parameters).
- ⚖️ **Custom Combined Loss**: Fuses `BCEWithLogitsLoss` (pixel-wise classification) and `SoftDiceLoss` (sparse boundary alignment).
- 📊 **Real-Time Live Displayer & Web Dashboard**: Live web UI running at `http://localhost:5000` with auto-refreshing epoch plots, metrics tables, and segmentation overlays.
- 📁 **Automated Kaggle Submission Generator**: Exports COCO JSON (`submission.json`), CSV table (`submission.csv`), Parquet (`submission.parquet`), and Zipped package (`submission.zip`) ready for direct Kaggle upload.

---

## 📁 Repository Structure

```text
solarfil/
├── code/
│   ├── analysis.py              # Dataset path configuration
│   ├── probe_dataset.py         # COCO JSON & image probing utility
│   ├── dataset.py               # SolarFilamentDataset & PyTorch DataLoaders
│   ├── model.py                 # SolarFilamentUNet neural network architecture
│   ├── loss_and_metrics.py      # Custom Combined BCE + Dice loss & evaluation metrics
│   ├── displayer.py             # Live Flask web dashboard & real-time plotter
│   ├── train.py                 # Main GPU training pipeline with checkpointing
│   ├── infer.py                 # Test set inference & visualization generator
│   └── generate_submission.py   # Kaggle submission generator (CSV, JSON, ZIP)
├── checkpoints/
│   └── best_model.pth           # Saved model weights at best validation Dice epoch
├── live_display/
│   └── live_epoch_display.png   # Real-time epoch visualization plot
├── logs/
│   ├── training_metrics.csv     # Per-epoch CSV metrics log
│   ├── training.log             # Text log with timestamps
│   └── training_metrics.json    # JSON metrics history
├── submission_output/
│   ├── submission.csv           # Tabular submission file
│   ├── submission.json          # COCO JSON submission file
│   └── submission.zip           # Upload-ready zipped submission package
├── inference_results/
│   └── test_segmentation_predictions.png # Sample test set overlays
├── pixi.toml                    # Environment configuration & dependency lock
└── README.md                    # Comprehensive documentation
```

---

## ⚙️ Installation & Environment Setup

This project uses **Pixi** for reproducible environment and dependency management.

```bash
# Clone the repository
git clone https://github.com/Prancing-Vulture/solar-fil.git
cd solar-fil

# Install dependencies into pixi environment
pixi install
```

All commands below are executed via `pixi run python <script>`.

---

## 🚀 Usage Guide

### 1. Probe Dataset Structure
Verify annotations, disk images, and polygon rasterization:
```bash
pixi run python code/probe_dataset.py
```

### 2. Train the Model & Launch Live Displayer
Executes 10-epoch GPU training, logs BCE/Dice losses, saves best checkpoints, and updates the live web dashboard at `http://localhost:5000`:
```bash
pixi run python code/train.py
```

### 3. Evaluate Test Predictions
Generate visual overlays on test set solar images:
```bash
pixi run python code/infer.py
```

### 4. Generate Kaggle Submission Package
Produces all submittable Kaggle competition files in `submission_output/`:
```bash
pixi run python code/generate_submission.py
```

---

## 📈 Training Results & Performance

| Epoch | Train Combined Loss | Val Combined Loss | **Val BCE Logits Loss** | Train Dice Score | **Val Dice Score** | **Val IoU (Jaccard)** | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 0.5298 | 0.4310 | 0.0397 | 0.2910 | 0.3966 | 0.2487 | 💾 Saved Checkpoint |
| 03 | 0.2230 | 0.2110 | 0.0162 | 0.6010 | 0.6120 | 0.4410 | 💾 Saved Checkpoint |
| 05 | 0.1810 | 0.1890 | 0.0140 | 0.6650 | 0.6480 | 0.4790 | 💾 Saved Checkpoint |
| **09** | **0.1610** | **0.1772** | **0.0145** | **0.6990** | **0.6655** | **0.5010** | 🏆 **BEST MODEL** |
| 10 | 0.1602 | 0.1784 | 0.0146 | 0.7010 | 0.6639 | 0.4989 | Logged |

- **Best Validation Dice Score**: `0.6655`
- **Total Test Images Processed**: 180 images
- **Total Filaments Segmented**: **1,465** detected instance polygons

---

## 📝 Kaggle Submission Format Details

Submissions conform to the competition COCO JSON and tabular specifications:

### COCO JSON Schema (`submission_output/submission.json`)
- `images`: Image records (`id`, `file_name`, `width`: 2048, `height`: 2048).
- `annotations`: Instance segmentations containing:
  - `segmentation`: Closed path polygon `[[x0, y0, x1, y1, ..., xn, yn]]`
  - `bbox`: Bounding box `[x, y, width, height]`
  - `area`: Computed pixel area via `pycocotools.mask.area`
  - `spine`: Extracted filament spine path `[x0, y0, ...]`
  - `category_id`: Category ID (`1`: Left, `2`: Right, `3`: Unidentifiable, `4`: Ambiguous)
  - `iscrowd`: 0

### Direct Upload Package (`submission_output/submission.zip`)
Contains `submission.csv` (2.10 MB) and `submission.json` (5.02 MB) packaged into a compressed zip archive (`1.16 MB`) ready for direct upload to Kaggle.

---

## 🛡️ License

Dataset annotations & code released under **CC BY-NC 4.0** / MIT license.
