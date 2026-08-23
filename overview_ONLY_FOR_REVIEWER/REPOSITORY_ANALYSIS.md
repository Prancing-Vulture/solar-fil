# Solar Filament Segmentation Challenge 2026

## Executive Summary

The repository implements a semantic segmentation pipeline for GONG H-Alpha solar-filament images and converts the resulting binary prediction into instance-like connected components for Kaggle submission.

The authoritative implementation is **not an attention U-Net**. `code/model.py` wraps `segmentation_models_pytorch.Unet` with an ImageNet-pretrained **ResNet-34 encoder**, one grayscale input channel, and one binary output channel.

The latest persistent training run reached:

- Best validation Dice: **0.6654** at epoch 13
- Best validation IoU: **0.5032** at epoch 13
- Best validation loss: **0.2215** at epoch 13
- Reported public leaderboard progression: **0.59 to 0.62**

Panoptic Quality is not implemented, logged, or reproducibly evaluated in this repository. Consequently, the reported validation scores are semantic pixel-overlap scores and should not be interpreted as competition-level PQ.

## 1. Repository Map

### Core execution path

`code/analysis.py` -> `code/dataset.py` -> `code/model.py` -> `code/loss_and_metrics.py` -> `code/train.py` -> `checkpoints/best_model.pth` -> `code/generate_submission.py`

### Supporting components

- `code/data_access.py`: KaggleHub download helper; not imported by training or inference.
- `code/data_auth.py`: KaggleHub login helper.
- `code/probe_dataset.py`: one-off COCO/image inspection utility.
- `code/probe_test_set.py`: one-off test-directory inspection utility.
- `code/infer.py`: five-image qualitative inference visualization.
- `code/displayer.py`: Flask dashboard and persistent metric logging.
- `code/test_smp_setup.py`: dependency and model smoke test.
- `code/test_submission_pipeline.py`: synthetic RLE and contour smoke test.

## 2. Data Ingestion and Preprocessing

### Source data

The expected dataset layout is:

```text
<hard-coded KaggleHub path>/MAGFiLO_1.0_Kaggle_2026/
  train/
    train_images/
    MAGFiLO_1.0_Annotations_kaggle2026_train.json
  test/
    test_images/
```

The data is described as 2048x2048 grayscale JPEG GONG H-Alpha observations. Training annotations are COCO-style polygon annotations.

### Dataset construction

`SolarFilamentDataset` performs the following operations:

1. Loads COCO JSON and filters image records against files present on disk.
2. Repairs `.jpeg` to `.jpg` filename mismatches.
3. Applies a NumPy seed of 42 and reserves 15% of valid images for validation.
4. Groups annotations by `image_id`.
5. Opens each observation as grayscale (`PIL.Image.convert('L')`).
6. Rasterizes every valid polygon into a single binary mask using `PIL.ImageDraw`.
7. Resizes the image to 512x512 with bilinear interpolation.
8. Resizes the mask to 512x512 with nearest-neighbor interpolation.
9. Applies CLAHE with `clipLimit=2.0` and `tileGridSize=(8, 8)`.
10. Scales image values to `[0, 1]` and returns tensors shaped approximately `[1, 512, 512]`.

The labels are therefore **semantic unions**. Annotation category, bounding box, area, spine, and `iscrowd` fields are ignored. Multiple filament instances touching in the rasterized union cannot be distinguished during training.

### Augmentation

Training-only Albumentations transforms are:

- Horizontal flip, probability 0.5
- Vertical flip, probability 0.5
- Random 90-degree rotation, probability 0.5
- Shift/scale/rotation with shift limit 0.05, scale limit 0.1, and rotation limit 15 degrees, probability 0.5
- Brightness/contrast variation with limits 0.15, probability 0.5
- Gaussian noise, probability 0.2

Validation uses preprocessing without random augmentation.

### Data risks

- The dataset path is hard-coded to one Windows user profile in `code/analysis.py`.
- Only the split permutation is seeded; PyTorch, CUDA, DataLoader workers, and augmentation randomness are not fully controlled.
- COCO records and annotations are not checked for duplicate annotators, invalid coordinates, or image-size consistency.
- The documented image count is inconsistent with historical logs.

## 3. Model Architecture

### Implemented architecture

`SolarFilamentAttentionUNet` is a naming mismatch. Its implementation is:

```text
Input: [B, 1, 512, 512]
  |
  v
SMP U-Net
  - Encoder: ResNet-34
  - Encoder weights: ImageNet
  - Input channels: 1
  - Decoder: standard SMP U-Net decoder with skip connections
  - Segmentation classes: 1
  |
  v
Output logits: [B, 1, 512, 512]
```

The forward pass is a direct delegation to `self.model(x)`. There are no attention gates, custom residual blocks, SiLU blocks, or manually implemented feature levels in the current source. The `base_features` argument is accepted but unused. `SolarFilamentUNet` is an alias to the same class.

### Why this model was selected

Compared with a scratch vanilla U-Net, the pretrained ResNet-34 encoder provides:

- Stronger multiscale features at the beginning of training.
- Skip connections that preserve local spatial detail.
- A practical transfer-learning baseline for a small, highly imbalanced dataset.
- Faster convergence within the 15-epoch training budget.

It is still a conventional semantic segmentation model. It does not explicitly model filament instances, topology, skeletons, or long-range continuity.

## 4. Training and Evaluation

### Optimization

`code/train.py` uses:

- Device: CUDA when available, otherwise CPU.
- Batch size: 4.
- Training resolution: 512x512.
- 15 epochs.
- AdamW optimizer with weight decay `1e-4`.
- Script entry point learning rate: `5e-4`.
- `CosineAnnealingLR` with `eta_min=1e-6`.
- Best checkpoint selected by validation Dice.

The checkpoint stores epoch, model state, optimizer state, validation Dice, and validation loss. It does not store scheduler state, configuration, split indices, random seeds, or preprocessing parameters.

### Loss

The implemented loss is:

```text
0.4 * BCEWithLogitsLoss(pos_weight=5.0) + 0.6 * SoftDiceLoss
```

The BCE positive weight compensates for sparse filament pixels. Soft Dice improves global foreground overlap compared with BCE alone.

The Dice loss flattens the entire batch before computing one score. This is not a per-image Dice average and can allow large or easy images to dominate the reported value.

### Metrics

`calculate_metrics` applies sigmoid, thresholds at 0.4, and computes globally flattened:

- Dice
- IoU/Jaccard
- Precision
- Recall

The training loop records only Dice for training and Dice/IoU for validation. It does not calculate:

- Panoptic Quality
- Instance matching
- One-to-many or many-to-one errors
- Boundary quality
- Component fragmentation or merge rate
- Per-image score distributions

### Persistent results

The latest coherent run is recorded in `logs/training_metrics.json` and `logs/training.log`:

| Epoch | Train loss | Validation loss | Train Dice | Validation Dice | Validation IoU |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.5360 | 0.3083 | 0.3213 | 0.5651 | 0.3973 |
| 5 | 0.2855 | 0.2314 | 0.5759 | 0.6516 | 0.4878 |
| 10 | 0.2730 | 0.2301 | 0.5929 | 0.6534 | 0.4901 |
| 13 | 0.2637 | 0.2215 | 0.6068 | **0.6654** | **0.5032** |
| 15 | 0.2631 | 0.2219 | 0.6072 | 0.6647 | 0.5025 |

An earlier July 28 log records a separate run with validation Dice 0.6804 and IoU 0.5200. Because the CSV and JSON contain multiple sessions and the run configuration is not versioned, this result should be treated as historical rather than as the reproducible current benchmark.

## 5. Inference and Submission Generation

The submission path in `code/generate_submission.py` is:

1. Load `checkpoints/best_model.pth`.
2. Enumerate and sort all test images.
3. Convert each image to grayscale.
4. Resize to 512x512.
5. Apply the same CLAHE preprocessing used during training.
6. Run the U-Net and apply sigmoid.
7. Resize the probability map to 2048x2048 using bilinear interpolation.
8. Threshold at probability 0.4.
9. Run OpenCV 8-connected-component analysis.
10. Discard components smaller than 300 full-resolution pixels.
11. Encode each remaining component as Fortran-order compressed COCO RLE.
12. Emit `filament_id` values in the form `<image_stem>_<component_number>`.
13. Emit an empty RLE fallback when no component survives filtering.
14. Write root `submission.csv`, `submission_output/submission.csv`, and a ZIP containing the latter CSV.

The current checked-in CSV contains 1,345 prediction rows over 180 test images. The stale `submission_output/submission.json` and historical logs contain different counts and formats.

Thresholds are inconsistent across the codebase: validation and submission use 0.4, while qualitative inference and dashboard overlays use 0.5.

## 6. Baseline Comparison

### Documented baseline

The README reports:

| Approach | Public leaderboard score |
|---|---:|
| Scratch U-Net baseline | 0.59 |
| ResNet-34 U-Net + CLAHE + area filtering | **0.62** |

This is the only persistent baseline comparison. There is no controlled local experiment isolating the contributions of pretraining, CLAHE, augmentation, loss weighting, or component filtering.

### Advantages over vanilla scratch segmentation

- **Fine-scale structures:** U-Net skip connections preserve higher-resolution spatial features, while 512x512 training retains more detail than a 256x256 baseline. CLAHE can improve the visibility of low-contrast filament boundaries.
- **Sparse foreground:** Dice loss directly rewards overlap on rare foreground pixels, and positive-weighted BCE increases the penalty for missing filament pixels.
- **Observatory noise:** CLAHE normalizes local contrast, augmentation exposes the model to photometric variation and noise, and the 300-pixel component filter suppresses small isolated detections.
- **Practical convergence:** ImageNet initialization is a reasonable choice for a small dataset and explains the reported improvement over scratch training.

### Remaining limitations against the competition objective

- **Barbs and thin branches:** Downsampling to 512x512 can erase narrow structures before prediction. Bilinear upsampling cannot recover details that were lost.
- **Structural continuity:** Pixelwise BCE/Dice has no explicit connectivity or skeleton objective. Thin filaments can fragment into islands.
- **Instance identity:** Training unions all polygons into one semantic mask, then infers instances from connected components. Touching filaments may merge, and one filament may split into several predictions.
- **One-to-many penalties:** The 300-pixel filter removes small components but does not repair fragmented predictions or merge components belonging to one physical filament.
- **Background artifacts:** A fixed area threshold is a coarse prior. Large artifacts can survive, while small true filaments can be removed.
- **Evaluation mismatch:** The local validation metric does not model the competition's instance-level or panoptic behavior.

## 7. Prioritized Improvement Roadmap

### Priority 0: Make validation competition-aligned

Before changing the model, implement a local evaluator that reports:

- Per-image Dice and IoU distributions.
- Instance/component matching using the competition's matching rule.
- One-to-many and many-to-one counts.
- Panoptic Quality, Segmentation Quality, and Recognition Quality if those are part of the official evaluator.
- Fragmentation, merge, false-positive, and false-negative rates.

Tune thresholds and post-processing against this evaluator, not semantic batch Dice alone.

### Priority 1: Improve the objective for thin connected structures

Run controlled ablations with the following candidates:

1. **Dice-Focal loss:** use focal BCE or focal Tversky for hard, sparse filament pixels plus soft Dice.
2. **Boundary loss:** add a distance-transform or signed-distance boundary term to improve narrow edges.
3. **Topology/connectivity loss:** add clDice or skeleton overlap loss to preserve filament centerlines and branches.
4. **Per-image loss reduction:** compute Dice per sample and average across the batch.
5. **Deep supervision:** expose intermediate decoder heads, especially at higher resolutions, and supervise them with Dice plus boundary loss.

A reasonable first experiment is:

```text
0.35 * BCE-Focal + 0.35 * SoftDice + 0.20 * BoundaryLoss + 0.10 * clDice
```

The weights should be selected using the competition-aligned validation evaluator.

### Priority 2: Preserve resolution and context

- Train with 768x768 or tiled 1024x1024 crops while retaining full-disk context.
- Use overlapping tiled inference and blend probability maps.
- Compare U-Net++ or FPN decoders for denser skip connectivity.
- Test a stronger encoder such as EfficientNet-B3/B4 or a modern ConvNeXt/Swin encoder if GPU memory permits.
- Consider a two-scale model: full-disk context branch plus high-resolution filament crop branch.

### Priority 3: Repair masks after prediction

Use validation-tuned, conservative post-processing:

- Morphological closing with a small elliptical kernel to bridge one- or two-pixel gaps.
- Skeleton-guided gap bridging for nearby endpoints aligned along a filament.
- Remove isolated noise using area, elongation, intensity support, and confidence statistics rather than area alone.
- Split large merged components using distance-transform watershed when multiple ridges are evident.
- Merge nearby components when their endpoints, orientation, and probability bridge support indicate one continuous filament.
- Keep an unmodified probability map for RLE generation where possible; morphology should be evaluated for both Dice and instance matching.

### Priority 4: TTA and ensembling

- Apply horizontal flip, vertical flip, 90-degree rotations, and inverse-transform averaging.
- Average logits or probabilities before thresholding; do not average binary masks.
- Ensemble checkpoints from the best validation epochs rather than using only one checkpoint.
- Ensemble complementary architectures such as ResNet-34 U-Net, U-Net++, and an FPN model.
- Tune threshold and component filtering jointly after TTA.

### Priority 5: Reproducibility and submission reliability

- Replace the hard-coded dataset path with CLI arguments or environment variables.
- Save split indices, seed, model configuration, preprocessing, threshold, and post-processing parameters in the checkpoint.
- Save scheduler state and optimizer state consistently.
- Add assertions for image coverage, RLE dimensions, unique IDs, expected columns, and empty-mask handling.
- Add end-to-end tests using a tiny synthetic dataset.
- Remove or rotate the plaintext Kaggle API token in `kaggleapi.txt`; it must be treated as compromised.

## 8. Architecture Diagram

The following Eraser diagram describes the implemented best-performing pipeline. It intentionally shows a standard ResNet-34 U-Net, not the unsupported historical attention-U-Net description.

```eraser
direction right

Training Images [shape: cylinder, label: "GONG H-Alpha JPEGs\n2048 x 2048 grayscale"]
COCO Annotations [shape: cylinder, label: "COCO JSON polygons"]

Load Dataset [label: "Load COCO records\nmatch files on disk"]
Rasterize Polygons [label: "PIL ImageDraw\nunion all polygons\nsemantic binary mask"]
Resize Train [label: "Image: bilinear\nMask: nearest\n512 x 512"]
CLAHE [label: "CLAHE\nclipLimit 2.0\ntileGrid 8 x 8"]
Augment [label: "Train only:\nflips, rotations, shift/scale,\nbrightness/contrast, Gaussian noise"]
Tensorize [label: "Normalize to [0,1]\nTensor [B,1,512,512]"]

ResNet34 [label: "ImageNet ResNet-34 Encoder\nmultiscale feature extraction"]
SkipFeatures [label: "Encoder skip features\nspatial detail preservation"]
UNetDecoder [label: "SMP U-Net Decoder\nupsampling + skip concatenation"]
SegHead [label: "1-channel segmentation head\nraw filament logits"]
Sigmoid [label: "Sigmoid probability map"]
Loss [label: "0.4 weighted BCE\n+ 0.6 soft Dice\npos_weight 5.0"]
Metrics [label: "Validation Dice / IoU\nthreshold 0.4"]
Checkpoint [shape: cylinder, label: "best_model.pth\nbest validation Dice"]

TestImages [shape: cylinder, label: "Test JPEGs\n2048 x 2048"]
TestPreprocess [label: "Grayscale -> resize 512\nCLAHE -> [0,1]"]
Inference [label: "ResNet-34 U-Net\nforward pass"]
Upsample [label: "Probability resize\n512 -> 2048"]
Threshold [label: "Binary threshold\nprobability > 0.4"]
Components [label: "OpenCV connected components\n8-connectivity"]
AreaFilter [label: "Discard components\narea < 300 pixels"]
RLE [label: "Fortran-order mask\ncompressed COCO RLE"]
CSV [shape: cylinder, label: "submission.csv\nfilament_id, segmentation_rle"]
ZIP [shape: cylinder, label: "submission.zip"]

Training Images -> Load Dataset
COCO Annotations -> Load Dataset
Load Dataset -> Rasterize Polygons
Rasterize Polygons -> Resize Train
Resize Train -> CLAHE
CLAHE -> Augment
Augment -> Tensorize
Tensorize -> ResNet34
ResNet34 -> SkipFeatures
SkipFeatures -> UNetDecoder
UNetDecoder -> SegHead
SegHead -> Sigmoid
Sigmoid -> Loss
Tensorize -> Loss
Loss -> Metrics
Metrics -> Checkpoint

TestImages -> TestPreprocess
TestPreprocess -> Inference
Checkpoint -> Inference
Inference -> Upsample
Upsample -> Threshold
Threshold -> Components
Components -> AreaFilter
AreaFilter -> RLE
RLE -> CSV
CSV -> ZIP
```

## 9. Evidence and Caveats

The following artifacts were treated as primary evidence:

- `code/dataset.py`
- `code/model.py`
- `code/loss_and_metrics.py`
- `code/train.py`
- `code/generate_submission.py`
- `logs/training_metrics.json`
- `logs/training.log`
- `README.md`

Historical `anti_gravity_logs` contain useful context but also conflicting generated narratives. They report alternate architectures, prediction counts, thresholds, and scores that cannot all correspond to the current source. No local artifact provides a verified Panoptic Quality result.
