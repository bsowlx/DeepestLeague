# Synthetic Data Generator

This guide covers `scripts/data/synthetic_data_generator.py`.

## Purpose
Generate synthetic minimap datasets for champion detection. Viewport-label training data is only emitted when the `--viewport-sim` flag is enabled.

Pipeline (high level):
1. Load assets (map variants, champion icons, map objects, ping/recall/tp effects).
2. Build randomized minimap background (fog + objects + vision).
3. Place champion icons and write YOLO labels.
4. Draw observer viewport rectangle (and save viewport labels only when `--viewport-sim` is enabled).
5. Apply ping/recall/tp effects and optional augmentation/noise.
6. Save split outputs and optional dataset YAML.

## Current Defaults
- Split sizes: `--n-train 1000`, `--n-val 100`, `--n-test 100`
- Image size: `--imgsz 256`
- JPEG compression noise: **enabled by default**
- Icon overlap: **enabled by default**
- Background augment: enabled by default
- Icon augment: disabled by default
- Recall/TP effects: enabled by default
- Viewport simulation: disabled by default

## CLI Flags
- `--use-hsv-augmentation`: enable HSV augmentation for icon colors.
- `--use-noise`: explicit enable for JPEG noise (already default-on).
- `--no-noise`: disable JPEG compression noise.
- `--allow-icon-overlap`: explicit enable for overlap (already default-on).
- `--no-icon-overlap`: disable icon overlap.
- `--no-bg-augment`: disable background augment.
- `--icon-augment`: enable icon blur/distortion/brightness augment.
- `--no-recall-tp`: disable recall/teleport overlays.
- `--viewport-sim` / `--viewport_sim`: randomize viewport size **and enable viewport label output**.
- `--workers`: process count.

## Output Layout
Given `--output-dir` and `--dataset-name`, output is:
- `<dataset_name>/<split>/images/*.png`
- `<dataset_name>/<split>/labels/*.txt` (YOLO labels)
- `<dataset_name>/<split>/viewport_labels/*.txt` (`x y w h` in output-image pixels, written only when `--viewport-sim` is used)
- `<dataset_name>/config.yaml` (if YAML output is enabled)

## Recommended Commands

Baseline generation:
```bash
python -m scripts.data.synthetic_data_generator \
  --n-train 100000 --n-val 10000 --n-test 10000 \
  --imgsz 512 \
  --output-dir data/synthetics \
  --dataset-name lol_minimap_512_100k_v4
```

Cleaner sample (disable noise + overlap):
```bash
python -m scripts.data.synthetic_data_generator \
  --n-train 2000 --n-val 200 --n-test 200 \
  --no-noise --no-icon-overlap \
  --dataset-name debug_no_noise_no_overlap
```

Fast smoke test:
```bash
python -m scripts.data.synthetic_data_generator \
  --n-train 50 --n-val 10 --n-test 10 \
  --workers 4 \
  --dataset-name smoke_test
```

## Notes
- Run with `python -m scripts.data.synthetic_data_generator` from repo root.
- Noise path is JPEG-only in current implementation.
