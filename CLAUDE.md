# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dental panoramic X-ray instance segmentation for 32 FDI tooth positions. Compares three methods as an ECE3070 course project:

1. **Active Contour** — unsupervised baseline using `skimage.segmentation.active_contour`
2. **U-Net** — standard U-Net trained on X-ray images only
3. **OralBBNet** — U-Net variant with YOLOv8 bounding-box priors concatenated as additional input channels

## Repository Structure

- `teeth_three_methods.ipynb` — **main entry point**. All three methods, data pipeline, evaluation, and visualization in one notebook. Contains the canonical implementations of `build_unet_model()`, `build_oralbbnet_model()`, evaluation functions, and threshold search logic.
- `notebooks/yolov8/` — YOLOv8 training (`yolov8_train.ipynb`) and inference (`yolo_test.ipynb`) on the Roboflow dataset
- `notebooks/Unet/` — Standalone U-Net training (`unet_training.ipynb`) and cross-validation (`unet+cv.ipynb`)
- `notebooks/yolov8+unet/` — OralBBNet training (`yolov8+unet_training.ipynb`) and cross-validation (`yolov8+unet+cv.ipynb`)
- `notebooks/Data_gen/2ddatagen.ipynb` — 2D data generation from OME TIFF masks
- `Dataset/yolo_train_dataset/` & `Dataset/yolo_test_dataset/` — YOLO-format data from Roboflow (1022 images, 32 classes, 640x640, histogram-equalized + augmented)
- `Dataset/bb_u_net_dataset/` — U-Net/OralBBNet dataset with OME TIFF label masks per category
- `Dataset/plots/` — Standalone analysis scripts (tooth position heatmaps, metadata visualizations, PCA, size correlation)
- `results/` — Output metrics (`metrics.csv`), training curves (`unet_history.csv`, `oralbbnet_history.csv`), qualitative visualizations, and model checkpoints
- `cache/` (gitignored) — Preprocessed data cache

## Evaluation Metrics

All metrics reported per image: Binary Dice, Binary IoU, Mean Channel Dice (averaged across 32 channels), and grouped Dice by tooth type (incisors/canines/premolars/molars). The notebook performs automatic threshold search on validation set to find the optimal binarization threshold for each method.

## Dataset

- UFBA-425 dataset (figshare.com/articles/dataset/UFBA-425/29827475)
- 32 FDI classes: 11-18, 21-28, 31-38, 41-48
- YOLO labels from Roboflow (`teeth-segmentation` project v15)
- OME TIFF mask files in `bb_u_net_dataset/labels/` for U-Net/OralBBNet training

## Dependencies

```bash
pip install numpy pandas matplotlib pillow tifffile scikit-image scikit-learn opencv-python tensorflow keras ultralytics
```

Python 3.10+, TensorFlow GPU recommended. U-Net and OralBBNet use TensorFlow/Keras; YOLOv8 uses `ultralytics`.

## Common Tasks

- **Run experiments**: open `teeth_three_methods.ipynb` and execute cells sequentially
- **Train YOLOv8 separately**: open `notebooks/yolov8/yolov8_train.ipynb`
- **Run dataset analysis scripts**: `python Dataset/plots/tooth_position_heatmaps.py` (runs from within `plots/` directory, expects `metadata.csv`)
- **Add new run** by editing config cells (Cell 4) in the main notebook for paths, hyperparameters, seed

## Key Architecture Details

- U-Net: depth=4, base filters=32, kernel=3, batch norm, dropout=0.12 (configurable via Cell 4 globals)
- OralBBNet: same U-Net encoder-decoder with additional input channels for YOLO bounding-box heatmaps (32 channels, one per FDI class)
- Active Contour: `skimage.segmentation.active_contour` with `alpha=0.015, beta=10, gamma=0.001`, initialised on YOLO bounding boxes
- Hyperparameters (optimal): Adam lr=0.0003, momentum=0.99, batch_size=2, dropout=0.12, regularization lambda=0.1, 60 epochs, halve lr on 5-epoch val loss plateau. YOLOv8 inference: confidence=0.5, IoU=0.5.

## Notes

- Model checkpoints (`*.weights.h5`, `*.pth`, `*.pt`, `*.ckpt`) are gitignored
- The `cache/` directory is gitignored; delete it to force re-caching of preprocessed data
- YOLO datasets are from Roboflow export — `data.yaml` files define class mappings