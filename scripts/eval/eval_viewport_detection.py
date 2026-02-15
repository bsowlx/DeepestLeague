import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from scripts.viewport.minimap_viewport import detect_viewport_robust


def _xywh_to_xyxy(box: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
    x, y, w, h = box
    return float(x), float(y), float(x + w), float(y + h)


def _clip_xyxy(
    box: Tuple[float, float, float, float],
    *,
    w: int,
    h: int,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(w), x1))
    x2 = max(0.0, min(float(w), x2))
    y1 = max(0.0, min(float(h), y1))
    y2 = max(0.0, min(float(h), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _read_viewport_label(path: Path) -> Optional[Tuple[int, int, int, int]]:
    if not path.exists():
        return None
    txt = path.read_text().strip()
    if not txt:
        return None
    parts = txt.split()
    if len(parts) < 4:
        return None
    try:
        x, y, w, h = map(int, parts[:4])
    except ValueError:
        return None
    return x, y, w, h


def _iter_images(images_dir: Path):
    yield from sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def main():
    p = argparse.ArgumentParser(
        description="Evaluate viewport detection against synthetic viewport_labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset_dir", required=True, help="Synthetic dataset folder (contains train/val/test)")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--iou", type=float, default=0.85, help="IoU threshold for pass")
    p.add_argument("--max", type=int, default=None, help="Max images to evaluate")
    p.add_argument("--visualize_dir", default=None, help="If set, write failure visualizations here")
    p.add_argument("--vis_max", type=int, default=200, help="Max failure images to write")
    p.add_argument("--process_size", type=int, default=None, help="Optional resize before detection (square)")
    args = p.parse_args()

    dataset_dir = Path(args.dataset_dir)
    images_dir = dataset_dir / args.split / "images"
    vp_dir = dataset_dir / args.split / "viewport_labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")
    if not vp_dir.exists():
        raise FileNotFoundError(f"Missing viewport_labels dir: {vp_dir}")

    vis_dir = Path(args.visualize_dir) if args.visualize_dir else None
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)

    ious: list[float] = []
    detected_ious: list[float] = []
    passed = 0
    detected = 0
    total = 0
    wrote = 0

    for img_path in _iter_images(images_dir):
        gt = _read_viewport_label(vp_dir / f"{img_path.stem}.txt")
        if gt is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        if args.process_size is not None:
            img = cv2.resize(img, (args.process_size, args.process_size), interpolation=cv2.INTER_AREA)

        pred = detect_viewport_robust(img, visualize=False)

        h, w = img.shape[:2]
        gt_xyxy = _clip_xyxy(_xywh_to_xyxy(gt), w=w, h=h)
        if pred is None:
            iou = 0.0
        else:
            detected += 1
            pred_xyxy = _clip_xyxy(_xywh_to_xyxy(pred), w=w, h=h)
            iou = _iou_xyxy(gt_xyxy, pred_xyxy)
            detected_ious.append(iou)

        ious.append(iou)
        if iou >= args.iou:
            passed += 1

        total += 1
        if vis_dir is not None and pred is not None and iou < args.iou and wrote < args.vis_max:
            vis = img.copy()
            gx1, gy1, gx2, gy2 = map(int, gt_xyxy)
            cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
            px1, py1, px2, py2 = map(int, _clip_xyxy(_xywh_to_xyxy(pred), w=w, h=h))
            cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 0, 255), 2)
            cv2.putText(vis, f"IoU {iou:.3f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imwrite(str(vis_dir / img_path.name), vis)
            wrote += 1

        if args.max is not None and total >= args.max:
            break

    if total == 0:
        print("No samples evaluated (missing labels/images?)")
        return

    ious_np = np.array(ious, dtype=np.float32)
    det_np = np.array(detected_ious, dtype=np.float32) if detected_ious else np.array([], dtype=np.float32)

    summary = {
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "iou_threshold": float(args.iou),
        "total": int(total),
        "detected": int(detected),
        "passed": int(passed),
        "pass_rate": float(passed / total),
        "detect_rate": float(detected / total),
        "mean_iou": float(ious_np.mean()),
        "median_iou": float(np.median(ious_np)),
        "mean_iou_detected_only": float(det_np.mean()) if det_np.size else None,
        "median_iou_detected_only": float(np.median(det_np)) if det_np.size else None,
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
