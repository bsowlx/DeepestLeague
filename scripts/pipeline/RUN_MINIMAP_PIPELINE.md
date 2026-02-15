# Minimap Detection Pipeline

This guide covers `scripts/pipeline/run_minimap_pipeline.py`.

## Purpose
Process League of Legends gameplay video (or images) to detect champion icons on the minimap and classify each as inside or outside the camera viewport.

Pipeline (high level):
1. Crop the minimap from the bottom-right corner of each frame.
2. Detect the observer viewport rectangle.
3. Run YOLO inference to locate champion icons.
4. Smooth detections with per-champion Kalman tracking (video only).
5. Apply temporal filtering and champion locking.
6. Classify each champion as inside or outside the viewport.
7. Write per-frame JSON records and optional overlay video.

## Current Defaults
- Crop size: 256×256 px
- YOLO confidence: `--conf 0.25`
- NMS IoU: `--iou 0.7`
- Kalman tracking: **enabled by default**
- Viewport stabilisation: **enabled by default**
- Champion locking: **enabled by default**
- Temporal min-hits: `--min_hits 30`
- Viewport margin: 6 px inset (configurable via `--viewport_margin_px`)

## CLI Flags

### I/O
- `--input`: video, image, or directory path (required).
- `--weights`: YOLO weights file (.pt).
- `--output_dir`: output directory.
- `--device`: inference device (default `"0"`).

### Detection
- `--imgsz`: YOLO input size (default 256).
- `--conf`: confidence threshold.
- `--iou`: NMS IoU threshold.

### Dataset / Filtering
- `--dataset`: dataset name (resolves class names from config YAML).
- `--data_config`: explicit dataset config YAML path.
- `--game_id`: Riot match ID to restrict detections to the 10 in-game champions.

### Video
- `--visualize`: write overlay video / images.
- `--stride`: process every Nth frame.
- `--max_frames`: stop after N frames.

### Viewport Stabilisation
- `--no_stabilize`: disable viewport stabilisation.

### Kalman Tracking
- `--no_kalman`: disable per-champion Kalman smoothing.
- `--kalman_base_missing`: frames to hold a missing champion before dropping (default 5, ~83ms @60fps).
- `--kalman_overlap_extra`: extra hold frames when occluded by another champion (default 300, ~5s @60fps).
- `--kalman_overlap_dist`: centre distance (px) to consider overlap.
- `--min_hits`: consecutive detections to confirm a champion (default 30, ~0.5s @60fps).

### Viewport Classification
- `--viewport_margin_px`: inset viewport by N px before counting a champion as inside (default half icon size).

### Champion Locking
- `--no_lock_champs`: disable champion locking.
- `--max_champs`: max champions to lock (default 10).
- `--lock_delay`: frames before locking begins (default 60, ~1s @60fps).
- `--lock_min_obs`: min measured frames before a champion can lock (default 45).
- `--lock_min_density`: min fraction of frames since first seen to lock (default 0.30).

## Champion Locking

Once enabled (default), the pipeline locks the 10 in-game champion class IDs to filter out false positives. A class must meet **all three** criteria to lock:

1. **Min observations** — measured (not predicted) in at least `lock_min_obs` frames.
2. **Mean confidence** — average detection confidence ≥ 0.40.
3. **Frame density** — detected in at least `lock_min_density` fraction of frames since first seen.

The density check prevents slow-accumulating false positives (e.g. a map object sporadically misclassified over thousands of frames) from stealing a lock slot. Locking is permanent: once all 10 slots fill, only locked classes pass through.

If `--game_id` is provided, upstream filtering already restricts to the 10 match champions, making the locker redundant but harmless.

## Output Layout
- `predictions.jsonl` — one JSON record per frame with viewport, detections, and inside/outside sets.
- `overlay.mp4` — annotated minimap video (if `--visualize`).
- `metadata.json` — run configuration and git state.

## Recommended Commands

Basic video run:
```bash
python -m scripts.pipeline.run_minimap_pipeline \
  --input replay.mp4 \
  --output_dir results/pipeline_run
```

With overlay and match filtering:
```bash
python -m scripts.pipeline.run_minimap_pipeline \
  --input replay.mp4 \
  --visualize \
  --dataset lol_minimap_512_100k_v4 \
  --game_id NA1_1234567890 \
  --output_dir results/pipeline_run
```

Fast debug (no tracking, no stabilisation):
```bash
python -m scripts.pipeline.run_minimap_pipeline \
  --input replay.mp4 \
  --no_kalman --no_stabilize --no_lock_champs \
  --max_frames 500 \
  --output_dir results/debug_run
```

## Notes
- Run with `python -m scripts.pipeline.run_minimap_pipeline` from repo root.
- Image mode (single image or directory) skips Kalman, temporal filtering, and champion locking.
- Viewport detection uses `scripts/minimap_viewport.detect_viewport_robust`.
- Kalman tracker uses a constant-velocity model with velocity decay (dt=0.6, decay=0.6).
