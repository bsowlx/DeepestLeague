# Minimap Detection Toolkit

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](environment.yml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

<p align="center">
  <img src="assets/videos/preview.gif" alt="Minimap pipeline preview" width="240" />
</p>

End-to-end League of Legends minimap detection automation: generate synthetic datasets, train YOLO/RT-DETR models, and run a viewport-aware inference pipeline on replays or images.

## ✨ Key Features
- Synthetic minimap dataset generator with controllable noise, overlap, and viewport simulation (@scripts/data/synthetic_data_generator.py).
- Training scripts for YOLO and RT-DETR with reproducible run metadata (@scripts/train/yolo_train.py, @scripts/train/rtdetr_train.py).
- Video/image inference pipeline with viewport detection, Kalman smoothing, and champion locking (@scripts/pipeline/run_minimap_pipeline.py).
- Evaluation utilities for YOLO predictions and viewport detection (@scripts/eval/yolo_eval.py, @scripts/eval/eval_viewport_detection.py).
- Docker + Conda paths for quick setup on CPU or GPU.

## 📂 Project Structure
```text
MinimapDetection/
├── assets/                # champion icons, maps, pings/effects
├── configs/               # YOLO & RT-DETR training/finetune 
├── data/                  # datasets
├── results/               # experiment outputs (checkpoints, viz)
├── scripts/
│   ├── data/              # synthetic_data_generator.py (+ docs)
│   ├── train/             # yolo_train, yolo_finetune, 
│   ├── eval/              # yolo_eval, rtdetr_eval, 
│   ├── pipeline/          # run_minimap_pipeline (+ docs)
│   └── viewport/          # viewport helpers
├── src/                   # shared utilities (git state, 
├── environment.yml        # Conda env (PyTorch + ultralytics)
├── Dockerfile
├── compose.yml
└── README.md
```


## 🚀 Quick Start
Run the minimap pipeline on your own replay or image directory from the repo root:

```bash
conda env create -f environment.yml
conda activate minimap-detection

python -m scripts.pipeline.run_minimap_pipeline \
  --input replay.mp4 \
  --output_dir results/pipeline_run \
  --weights yolo11l-minimap.pt \
  --visualize
```

Image/directory mode works too (no Kalman/locking):

```bash
python -m scripts.pipeline.run_minimap_pipeline \
  --input path/to/frames_dir \
  --weights yolo11l-minimap.pt \
  --output_dir results/pipeline_images
```

More flags are documented in [scripts/pipeline/RUN_MINIMAP_PIPELINE.md](scripts/pipeline/RUN_MINIMAP_PIPELINE.md).

## 🛠️ Installation

**Conda**
```bash
conda env create -f environment.yml
conda activate minimap-detection
```

**Docker**
```bash
docker compose up -d --build
docker compose exec minimap bash
```

*Dependency notes:* `environment.yml` targets Python 3.12 with PyTorch 2.5.1/torchvision 0.20.1/torchaudio 2.5.1, plus ultralytics 8.3.115, OpenCV, Albumentations, and W&B. CUDA wheels can be installed manually if your platform needs GPU acceleration.

## � Hugging Face Assets
- Synthetic minimap sample: https://huggingface.co/datasets/boboyes/leagueoflegends-synthetic-dataset
- Replay dataset: https://huggingface.co/datasets/lusung33/AAAI26_LoL_MinimapDetection_Dataset
- Best Models: https://huggingface.co/boboyes/leagueoflegends-minimap-detection

## �📊 Model Results

**YOLOv11 Synthetic (replay test)**

| Model | P | R | mAP50 | mAP50-95 | Inf (ms) |
|---|---:|---:|---:|---:|---:|
| yolo11n | 0.900 | 0.674 | 0.740 | 0.582 | **1.8** |
| yolo11s | 0.915 | 0.727 | 0.786 | 0.621 | 2.2 |
| yolo11m | 0.930 | 0.767 | 0.824 | 0.644 | 3.0 |
| yolo11l | **0.936** | 0.770 | 0.827 | 0.646 | 3.8 |
| yolo11x | **0.936** | **0.787** | **0.839** | **0.649** | 5.0 |

**YOLOv11 Finetune (replay)**

| Model | P | R | mAP50 | mAP50-95 | Inf (ms) |
|---|---:|---:|---:|---:|---:|
| yolo11n | 0.920 | 0.701 | 0.769 | 0.635 | **1.7** |
| yolo11s | 0.929 | 0.756 | 0.810 | 0.678 | 1.8 |
| yolo11m | 0.931 | 0.804 | 0.850 | 0.726 | 3.0 |
| yolo11l | **0.935** | 0.808 | 0.853 | **0.729** | 3.3 |
| yolo11x | **0.935** | **0.816** | **0.857** | 0.726 | 5.1 |

**YOLOv26 (replay test)**

| Model | P | R | mAP50 | mAP50-95 | Inf (ms) |
|---|---:|---:|---:|---:|---:|
| yolo26n | 0.769 | 0.548 | 0.621 | 0.476 | 1.9 |
| yolo26s | 0.844 | 0.628 | 0.698 | 0.549 | 2.2 |
| yolo26m | 0.902 | 0.699 | 0.765 | 0.603 | 3.8 |
| yolo26l | 0.914 | 0.745 | 0.808 | 0.639 | 3.8 |
| yolo26x | **0.920** | **0.779** | **0.831** | **0.657** | 5.1 |

**YOLOv26 Finetune (replay)**

| Model | P | R | mAP50 | mAP50-95 | Inf (ms) |
|---|---:|---:|---:|---:|---:|
| yolo26n | 0.843 | 0.625 | 0.706 | 0.570 | 1.8 |
| yolo26s | 0.893 | 0.682 | 0.762 | 0.626 | 1.7 |
| yolo26m | 0.922 | 0.742 | 0.807 | 0.680 | 3.2 |
| yolo26l | 0.928 | 0.786 | 0.836 | 0.720 | 3.8 |
| yolo26x | **0.935** | **0.797** | **0.851** | **0.729** | 5.0 |

## 📖 Workflow / Usage

### 1) Generate synthetic data
Docs: [scripts/data/SYNTHETIC_DATA_GENERATOR.md](scripts/data/SYNTHETIC_DATA_GENERATOR.md)
```bash
python -m scripts.data.synthetic_data_generator \
  --n-train 100000 --n-val 10000 --n-test 10000 \
  --imgsz 256 \
  --output-dir data/synthetics \
  --dataset-name synthetic_256_100k
```

### 2) Training
YOLO: @scripts/train/yolo_train.py
```bash
python -m scripts.train.yolo_train \
  --model yolo11l.pt \
  --dataset synthetic_256_100k \
  --config configs/yolo/train_default.yaml \
  --run_id yolo11l_seed1 \
  --device 0
```

RT-DETR: @scripts/train/rtdetr_train.py
```bash
python -m scripts.train.rtdetr_train \
  --model configs/rtdetr/rtdetr-x.yaml \
  --dataset synthetic_256_100k \
  --run_id rtdetr_x_seed1 \
  --device 0
```

Fine-tune YOLO (transfer):
```bash
python -m scripts.train.yolo_finetune \
  --weights results/experiments/<run_id>/weights/best.pt \
  --dataset replay_256 \
  --config configs/yolo/finetune_default.yaml \
  --run_id finetune_<run_id> \
  --device 0
```

### 3) Evaluate
YOLO predictions: @scripts/eval/yolo_eval.py
```bash
python -m scripts.eval.yolo_eval \
  --weights yolo11l-minimap.pt \
  --dataset replay_256 \
  --eval_id eval_replay \
  --split test \
  --imgsz 256 \
  --device 0
```

Viewport detection eval: @scripts/eval/eval_viewport_detection.py
```bash
python -m scripts.eval.eval_viewport_detection \
  --dataset_dir replay_256 \
  --split test \
  --iou 0.85
```

### 4) Run the inference pipeline
Docs: [scripts/pipeline/RUN_MINIMAP_PIPELINE.md](scripts/pipeline/RUN_MINIMAP_PIPELINE.md)
```bash
python -m scripts.pipeline.run_minimap_pipeline \
  --input replay.mp4 \
  --weights yolo11l-minimap.pt \
  --output_dir results/pipeline \
  --visualize \
  --dataset replay_256 \
  --game_id NA1_1234567890 
```

### 5) Helpful notes
- Run commands from the repo root with `python -m` to keep imports deterministic.
- Use explicit `--run_id` / `--eval_id` so outputs land under `results/experiments/<run_id>/` and `runs/detect/<eval_id>/` with metadata captured by `src/run_utils.write_metadata`.
- Prefer dataset names that match generated configs (e.g., `data/synthetics/<dataset>/config.yaml`).


## 📜 License
The MIT License (MIT)
