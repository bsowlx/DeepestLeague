# Minimap Viewport Detection

This guide covers `scripts/viewport/minimap_viewport.py`.

## Purpose
Detect the observer (camera) viewport rectangle on a League of Legends minimap crop using classical computer vision (no ML). The viewport is the bright, semi-transparent rectangle that shows where the player's camera is pointing on the map.

## Algorithm

The detector uses a three-strategy cascade, returning the first successful result:

### 1. Corner Junction (highest confidence)
- Find where a horizontal and vertical Hough line meet within 12 px (a viewport corner).
- Estimate box dimensions from line lengths; if one edge is clipped at the image border, infer its length from the other edge using the 16:9 aspect ratio.
- Score candidates by total line length + aspect-ratio bonus. Best score wins.

### 2. Partial-Edge Inference (medium confidence)
- Triggered only when Strategy 1 finds nothing.
- Look for isolated horizontal or vertical lines that touch an image border (partial viewport).
- Infer the full box from the visible edge length and 16:9 aspect ratio.
- Pick the longest qualifying line.

### 3. Longest-Line Fallback (lowest confidence)
- Triggered only when Strategies 1 and 2 both fail.
- Take the single longest detected line (must exceed 20% of image width).
- Infer box dimensions from aspect ratio and position from which half of the image the line sits in.

## Preprocessing Pipeline
1. Convert to HSV.
2. Adaptive brightness threshold: `clamp(percentile_98(V) - 10, 160, 230)`.
3. Binary mask: high-V, low-S pixels (the viewport border is bright and desaturated).
4. Morphological close (3×3 rect kernel).
5. Canny edge detection (50, 150).
6. Hough line detection with progressive relaxation (3 passes with decreasing thresholds).

## Constants
| Name | Value | Description |
|---|---|---|
| `ASPECT_RATIO` | 16/9 | Expected viewport aspect ratio |
| `MIN_W` | 15% of image width | Minimum viewport width to accept |
| `MIN_H` | `MIN_W / ASPECT_RATIO` | Minimum viewport height to accept |
| `BORDER_MARGIN` | 5 px | Distance to image edge to consider a line "clipped" |

## API

### `detect_viewport_robust(img, visualize=False)`
- **Input**: BGR minimap crop (`np.ndarray`), optional debug visualization.
- **Output**: `(x, y, w, h)` tuple in pixel coordinates, or `None` if no viewport found.
- **Note**: The returned box may extend beyond image bounds (partially visible viewport).

## CLI (Batch Runner)

```bash
python -m scripts.viewport.minimap_viewport \
  --input-dir path/to/minimap_crops \
  --output-dir path/to/results
```

### Flags
- `--input-dir`: directory of minimap crop images (required).
- `--output-dir`: directory for annotated output images (required).
- `--visualize`: show interactive debug windows (mask, edges, lines, result).
- `--process-size`: resize input images to this square size before detection (default 256).

### Output Layout
- `<output-dir>/<image>.png` — annotated images with detected viewport (blue box + green overlay).
- `<output-dir>/failures/<image>.png` — images where detection failed.
- Console prints per-image SUCCESS/FAILED and final success rate.

## Integration
- Imported by `scripts/pipeline/run_minimap_pipeline.py` for per-frame viewport detection.
- Imported by `scripts/eval/eval_viewport_detection.py` for evaluation against ground-truth labels.
- The pipeline applies `ViewportStabilizer` (from `src/viewport_utils`) on top of raw detections for temporal smoothing in video mode.

## Notes
- Run with `python -m scripts.viewport.minimap_viewport` from repo root.
- Expects a 256×256 minimap crop (the standard `CROP_SIZE` used throughout the project).
- No ML model required — purely Hough-line + geometric reasoning.
- The `visualize` flag uses `cv2.imshow` and blocks on `cv2.waitKey`; not suitable for headless environments.
