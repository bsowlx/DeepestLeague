"""
Minimap detection pipeline: crop → viewport → YOLO → Kalman → in/out classification.

Processes League of Legends gameplay video (or images) to detect champion icons
on the minimap and classify each as inside or outside the camera viewport.

Usage:
    python -m scripts.pipeline.run_minimap_pipeline --input video.mp4 --output_dir results/out
    python -m scripts.pipeline.run_minimap_pipeline --input video.mp4 --visualize
    python -m scripts.pipeline.run_minimap_pipeline --input frames/ --no_kalman --no_stabilize
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from scripts.viewport.minimap_viewport import detect_viewport_robust
from src.run_utils import get_git_state, write_metadata
from src.viewport_utils import ViewportStabilizer

# ────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
CROP_SIZE = 256
ICON_SIZE_PX = int(np.round(CROP_SIZE / 512 * 43))
DEFAULT_OVERLAP_DIST = ICON_SIZE_PX / 3.0
DEFAULT_VIEWPORT_MARGIN_PX = 6
DEFAULT_WEIGHTS = (
    ROOT / "results" / "experiments"
    / "finetune_v5_yolo11l_default_b962_epoch402" / "weights" / "best.pt"
)

# ────────────────────────────────────────────────────────────────────
# Dataset / class-name helpers
# ────────────────────────────────────────────────────────────────────

def _resolve_data_config(
    dataset: Optional[str], data_config: Optional[str],
) -> Optional[Path]:
    """Locate dataset YAML from an explicit path or a dataset name."""
    if data_config:
        p = Path(data_config)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        return p if p.exists() else None
    if not dataset:
        return None
    for candidate in (
        ROOT / "results" / "configs" / f"{dataset}.yaml",
        ROOT / "data" / "synthetics" / dataset / "config.yaml",
        ROOT / "data" / "replays" / dataset / "config.yaml",
    ):
        if candidate.exists():
            return candidate
    return None


def _load_class_names(config_path: Optional[Path]) -> Optional[list[str]]:
    if not config_path or not config_path.exists():
        return None
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names") if isinstance(cfg, dict) else None
    return list(names) if names else None


def _model_class_names(model: YOLO) -> Optional[list[str]]:
    """Extract class-name list from YOLO weights metadata (if available)."""
    raw = getattr(model, "names", None)
    if raw is None:
        raw = getattr(getattr(model, "model", None), "names", None)
    if raw is None:
        return None

    if isinstance(raw, list):
        return list(raw)

    if isinstance(raw, dict):
        normalized: dict[int, str] = {}
        for key, value in raw.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            normalized[idx] = str(value)
        if not normalized:
            return None
        max_idx = max(normalized)
        return [normalized.get(i, str(i)) for i in range(max_idx + 1)]

    return None


def _get_allowed_class_ids(
    game_id: Optional[str], config_path: Optional[Path],
) -> Optional[set[int]]:
    """Fetch the 10 champion class-IDs for a specific match."""
    if not game_id or not config_path:
        return None
    try:
        from src.visualize_utils import (
            get_game_info_by_match_id, get_champion_name, get_class_indices,
        )
        match_data = get_game_info_by_match_id(game_id)
        champions = get_champion_name(match_data)
        indices = get_class_indices(champions, str(config_path))
    except Exception as exc:
        print(f"[warn] Could not resolve match classes: {exc}")
        return None
    filtered = {i for i in indices if i is not None and i >= 0}
    return filtered or None


# ────────────────────────────────────────────────────────────────────
# Minimap crop
# ────────────────────────────────────────────────────────────────────

def crop_minimap(frame: np.ndarray) -> np.ndarray:
    """Extract the bottom-right CROP_SIZE×CROP_SIZE region (minimap)."""
    h, w = frame.shape[:2]
    return frame[max(0, h - CROP_SIZE):h, max(0, w - CROP_SIZE):w]


# ────────────────────────────────────────────────────────────────────
# Per-champion Kalman tracker
# ────────────────────────────────────────────────────────────────────

class _KalmanTrack:
    """Constant-velocity Kalman filter for one champion icon."""

    def __init__(self, cx: float, cy: float, w: float, h: float, conf: float):
        kf = cv2.KalmanFilter(4, 2)
        dt, decay = 0.6, 0.6
        kf.transitionMatrix = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, decay, 0], [0, 0, 0, decay]],
            dtype=np.float32,
        )
        kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 5e-3
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2e-1
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
        self.kf = kf
        self.missing = 0
        self.overlap_boost = 0
        self.last_wh = (w, h)
        self.last_conf = conf
        self.last_center = (cx, cy)

    def predict(self) -> tuple[float, float]:
        p = self.kf.predict()
        self.last_center = (float(p[0, 0]), float(p[1, 0]))
        return self.last_center

    def correct(self, cx: float, cy: float, w: float, h: float, conf: float) -> tuple[float, float]:
        self.kf.correct(np.array([[cx], [cy]], dtype=np.float32))
        s = self.kf.statePost
        self.last_wh, self.last_conf = (w, h), conf
        self.last_center = (float(s[0, 0]), float(s[1, 0]))
        return self.last_center


# ────────────────────────────────────────────────────────────────────
# Detection helpers
# ────────────────────────────────────────────────────────────────────

def _clamp_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(round(v)))))


def _viewport_safe_zone(
    vp: Iterable[int],
    *,
    img_w: int,
    img_h: int,
    margin_px: int,
) -> Optional[tuple[int, int, int, int]]:
    """Return a clamped, inset viewport box.

    The margin intentionally ignores ambiguous edge regions so minor viewport
    jitter does not flip inside/outside for champions near the boundary.
    """
    vx, vy, vw, vh = [int(v) for v in vp]

    # Clamp to image bounds first (viewport detection can extend past crop).
    x1 = _clamp_int(vx, 0, img_w - 1)
    y1 = _clamp_int(vy, 0, img_h - 1)
    x2 = _clamp_int(vx + vw, 0, img_w)
    y2 = _clamp_int(vy + vh, 0, img_h)
    if x2 <= x1 or y2 <= y1:
        return None

    # Inset (safe-zone). If it collapses, treat as unusable.
    m = int(max(0, margin_px))
    x1 += m
    y1 += m
    x2 -= m
    y2 -= m
    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2 - x1, y2 - y1


def _inside_viewport(
    bbox_xyxy: list[float],
    vp: Optional[Iterable[int]],
    *,
    img_w: int,
    img_h: int,
    safe_margin_px: int = 0,
) -> bool:
    if vp is None:
        return False

    safe = _viewport_safe_zone(vp, img_w=img_w, img_h=img_h, margin_px=safe_margin_px)
    if safe is None:
        return False

    cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
    cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
    vx, vy, vw, vh = safe
    return vx <= cx <= vx + vw and vy <= cy <= vy + vh


def _bbox_center(b: list[float]) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _bbox_from_center(
    cx: float, cy: float, w: float, h: float, iw: int, ih: int,
) -> list[float]:
    return [
        max(0.0, min(iw - 1.0, cx - w / 2)),
        max(0.0, min(ih - 1.0, cy - h / 2)),
        max(0.0, min(iw - 1.0, cx + w / 2)),
        max(0.0, min(ih - 1.0, cy + h / 2)),
    ]


def _predict_detections(
    model: YOLO, crop: np.ndarray,
    conf: float, iou: float, imgsz: int, device: str,
    names: Optional[list[str]], allowed_ids: Optional[set[int]],
    viewport_box,
    viewport_margin_px: int,
) -> list[dict]:
    """Run YOLO inference on a single crop and return structured detections."""
    result = model.predict(crop, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)[0]
    detections = []
    img_h, img_w = crop.shape[:2]
    for box, cid, score in zip(
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.cls.cpu().numpy(),
        result.boxes.conf.cpu().numpy(),
    ):
        cid = int(cid)
        if allowed_ids is not None and cid not in allowed_ids:
            continue
        bbox_xyxy = [float(v) for v in box]
        detections.append({
            "class_id": cid,
            "class_name": names[cid] if names and cid < len(names) else None,
            "conf": float(score),
            "bbox_xyxy": bbox_xyxy,
            "inside_viewport": _inside_viewport(
                bbox_xyxy,
                viewport_box,
                img_w=img_w,
                img_h=img_h,
                safe_margin_px=viewport_margin_px,
            ),
        })
    return detections


def _summarize_sets(detections: list[dict], names: Optional[list[str]]) -> dict:
    inner = sorted({d["class_id"] for d in detections if d["inside_viewport"]})
    outer = sorted({d["class_id"] for d in detections if not d["inside_viewport"]})
    return {
        "inner_champion_ids": inner,
        "outer_champion_ids": outer,
        "inner_champion_names": [names[i] for i in inner] if names else None,
        "outer_champion_names": [names[i] for i in outer] if names else None,
    }


def _assign_class_names(detections: list[dict], names: Optional[list[str]]) -> None:
    if not names:
        return
    for det in detections:
        cid = det.get("class_id")
        if cid is None or cid < 0 or cid >= len(names):
            continue
        det["class_name"] = names[cid]


# ────────────────────────────────────────────────────────────────────
# Kalman smoothing + temporal filtering (video only)
# ────────────────────────────────────────────────────────────────────

def _find_overlap_classes(detections: list[dict], dist_thresh: float) -> set[int]:
    """Return class IDs whose centres are within *dist_thresh* of another."""
    centers = [(d["class_id"], _bbox_center(d["bbox_xyxy"])) for d in detections]
    overlap: set[int] = set()
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            ci, (x1, y1) = centers[i]
            cj, (x2, y2) = centers[j]
            if np.hypot(x2 - x1, y2 - y1) <= dist_thresh:
                overlap |= {ci, cj}
    return overlap


def _apply_kalman(
    detections: list[dict],
    tracks: dict[int, _KalmanTrack],
    img_w: int, img_h: int,
    base_missing: int, overlap_extra: int, overlap_dist: float,
) -> list[dict]:
    """Smooth measured detections and predict missing champions."""
    overlap_classes = _find_overlap_classes(detections, overlap_dist)

    # Best measurement per class
    best: dict[int, dict] = {}
    for d in detections:
        cid = d["class_id"]
        if cid not in best or d["conf"] > best[cid]["conf"]:
            best[cid] = d
    measured_centers = {c: _bbox_center(d["bbox_xyxy"]) for c, d in best.items()}

    updated: list[dict] = []

    # Update measured tracks
    for cid, det in best.items():
        cx, cy = _bbox_center(det["bbox_xyxy"])
        w = det["bbox_xyxy"][2] - det["bbox_xyxy"][0]
        h = det["bbox_xyxy"][3] - det["bbox_xyxy"][1]
        if cid not in tracks:
            tracks[cid] = _KalmanTrack(cx, cy, w, h, det["conf"])
        trk = tracks[cid]
        trk.predict()
        fx, fy = trk.correct(cx, cy, w, h, det["conf"])
        if cid in overlap_classes:
            trk.overlap_boost = overlap_extra
        trk.missing = 0
        det["bbox_xyxy"] = _bbox_from_center(fx, fy, w, h, img_w, img_h)
        det["predicted"] = False
        det["overlap_hold"] = cid in overlap_classes
        updated.append(det)

    # Predict missing tracks
    for cid, trk in list(tracks.items()):
        if cid in best:
            continue
        trk.missing += 1
        if trk.overlap_boost > 0:
            trk.overlap_boost -= 1
        else:
            for mx, my in measured_centers.values():
                if np.hypot(mx - trk.last_center[0], my - trk.last_center[1]) <= overlap_dist:
                    trk.overlap_boost = overlap_extra
                    break
        limit = base_missing + (overlap_extra if trk.overlap_boost > 0 else 0)
        if trk.missing > limit:
            del tracks[cid]
            continue
        px, py = trk.predict()
        w, h = trk.last_wh
        updated.append({
            "class_id": cid, "class_name": None,
            "conf": float(trk.last_conf),
            "bbox_xyxy": _bbox_from_center(px, py, w, h, img_w, img_h),
            "inside_viewport": False,
            "predicted": True,
            "overlap_hold": trk.overlap_boost > 0,
        })
    return updated


def _apply_temporal_filter(
    detections: list[dict],
    measured_ids: set[int],
    state: dict[int, dict],
    min_hits: int,
) -> list[dict]:
    """Require *min_hits* consecutive measurements before showing a class."""
    if min_hits <= 1:
        return detections

    for cid in measured_ids:
        st = state.setdefault(cid, {"hits": 0, "misses": 0, "confirmed": False})
        st["hits"] += 1
        st["misses"] = 0
        if st["hits"] >= min_hits:
            st["confirmed"] = True

    overlap_holds = {d["class_id"] for d in detections if d.get("overlap_hold")}
    for cid, st in list(state.items()):
        if cid in measured_ids or cid in overlap_holds:
            continue
        st["misses"] += 1
        st["hits"] = 0
        if st["misses"] >= min_hits:
            st["confirmed"] = False

    return [
        d for d in detections
        if d.get("overlap_hold") or state.get(d["class_id"], {}).get("confirmed")
    ]


def _dedupe_by_class(detections: list[dict]) -> list[dict]:
    """Keep only the best detection per class (prefer measured over predicted)."""
    best: dict[int, dict] = {}
    for d in detections:
        cid = d["class_id"]
        prev = best.get(cid)
        if prev is None:
            best[cid] = d
            continue
        prev_pred = bool(prev.get("predicted"))
        d_pred = bool(d.get("predicted"))
        if (prev_pred and not d_pred) or (prev_pred == d_pred and d["conf"] > prev["conf"]):
            best[cid] = d
    return list(best.values())


class ChampionLocker:
    """Lock the 10 in-game champion IDs, robust to false positives.

    A class must be *measured* (not predicted) in at least ``min_obs`` separate
    frames, have a mean confidence >= ``min_conf``, **and** appear in at least
    ``min_density`` fraction of frames since it was first seen before it is
    permanently locked.  Locking only begins after ``delay`` frames have elapsed.

    The density check prevents slow-accumulating false positives (e.g. a map
    object sporadically misclassified over thousands of frames) from stealing
    a lock slot.
    """

    def __init__(
        self,
        max_champs: int = 10,
        delay: int = 30,
        min_obs: int = 20,
        min_conf: float = 0.40,
        min_density: float = 0.30,
    ):
        self.max_champs = max_champs
        self.delay = delay
        self.min_obs = min_obs
        self.min_conf = min_conf
        self.min_density = min_density
        self.locked: list[int] = []
        self._locked_set: set[int] = set()
        # running stats: class_id -> {"count": int, "conf_sum": float, "first_seen": int}
        self._stats: dict[int, dict[str, float]] = {}

    @property
    def is_full(self) -> bool:
        return len(self.locked) >= self.max_champs

    def update(self, detections: list[dict], frame_idx: int) -> None:
        """Accumulate stats and promote candidates that meet the bar."""
        if self.is_full or frame_idx < self.delay:
            return
        for d in detections:
            if d.get("predicted"):
                continue
            cid = d["class_id"]
            if cid in self._locked_set:
                continue
            st = self._stats.setdefault(
                cid, {"count": 0, "conf_sum": 0.0, "first_seen": frame_idx},
            )
            st["count"] += 1
            st["conf_sum"] += d["conf"]
            elapsed = frame_idx - st["first_seen"] + 1
            density = st["count"] / elapsed if elapsed > 0 else 0.0
            if (
                st["count"] >= self.min_obs
                and st["conf_sum"] / st["count"] >= self.min_conf
                and density >= self.min_density
            ):
                self.locked.append(cid)
                self._locked_set.add(cid)
                if self.is_full:
                    return

    def filter(self, detections: list[dict]) -> list[dict]:
        """Keep only locked classes once all slots are filled."""
        if not self.is_full:
            return detections
        return [d for d in detections if d["class_id"] in self._locked_set]


# ────────────────────────────────────────────────────────────────────
# Overlay drawing
# ────────────────────────────────────────────────────────────────────

def _draw_overlay(img: np.ndarray, viewport_box, detections: list[dict]) -> np.ndarray:
    vis = img.copy()
    if viewport_box is not None:
        vx, vy, vw, vh = map(int, viewport_box)
        cv2.rectangle(vis, (vx, vy), (vx + vw, vy + vh), (0, 255, 0), 2)
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox_xyxy"])
        color = (0, 0, 255) if det["inside_viewport"] else (255, 0, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = det.get("class_name") or str(det["class_id"])
        cv2.putText(vis, label, (x1, max(10, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return vis


# ────────────────────────────────────────────────────────────────────
# Image processing (single / directory)
# ────────────────────────────────────────────────────────────────────

def _iter_images(path: Path) -> Iterable[Path]:
    if path.is_dir():
        yield from sorted(
            p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
    else:
        yield path


def process_images(
    model: YOLO, input_path: Path, output_dir: Path,
    names: Optional[list[str]], allowed_ids: Optional[set[int]],
    conf: float, iou: float, device: str, imgsz: int, visualize: bool,
    viewport_margin_px: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.jsonl").open("w") as fp:
        for img_path in _iter_images(input_path):
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            crop = crop_minimap(image)
            if crop.shape[:2] != (CROP_SIZE, CROP_SIZE):
                crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)

            vp = detect_viewport_robust(crop)
            dets = _predict_detections(
                model, crop, conf, iou, imgsz, device, names, allowed_ids, vp,
                viewport_margin_px,
            )
            _assign_class_names(dets, names)
            record = {
                "image": img_path.name,
                "viewport": [int(v) for v in vp] if vp else None,
                "detections": dets,
                **_summarize_sets(dets, names),
            }
            fp.write(json.dumps(record) + "\n")
            if visualize:
                cv2.imwrite(str(output_dir / img_path.name), _draw_overlay(crop, vp, dets))


# ────────────────────────────────────────────────────────────────────
# Video processing
# ────────────────────────────────────────────────────────────────────

def process_video(
    model: YOLO, input_path: Path, output_dir: Path,
    names: Optional[list[str]], allowed_ids: Optional[set[int]],
    *,
    conf: float = 0.25,
    iou: float = 0.7,
    device: str = "0",
    imgsz: int = CROP_SIZE,
    visualize: bool = False,
    stabilize: bool = True,
    stride: int = 1,
    max_frames: Optional[int] = None,
    kalman: bool = True,
    kalman_base_missing: int = 5,
    kalman_overlap_extra: int = 300,
    kalman_overlap_dist: float = DEFAULT_OVERLAP_DIST,
    min_hits: int = 15,
    lock_champs: bool = True,
    max_champs: int = 10,
    lock_delay: int = 60,
    lock_min_obs: int = 45,
    lock_min_density: float = 0.30,
    viewport_margin_px: int = DEFAULT_VIEWPORT_MARGIN_PX,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"[error] Cannot open video: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = None
    if visualize:
        writer = cv2.VideoWriter(
            str(output_dir / "overlay.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), fps, (CROP_SIZE, CROP_SIZE),
        )

    stabilizer = ViewportStabilizer(img_size=CROP_SIZE) if stabilize else None
    tracks: dict[int, _KalmanTrack] = {}
    temporal: dict[int, dict] = {}
    locker = ChampionLocker(
        max_champs=max_champs, delay=lock_delay, min_obs=lock_min_obs,
        min_conf=0.40, min_density=lock_min_density,
    ) if lock_champs else None

    with (output_dir / "predictions.jsonl").open("w") as fp:
        frame_idx = 0
        processed = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if stride > 1 and frame_idx % stride != 0:
                frame_idx += 1
                continue

            crop = crop_minimap(frame)
            if crop.shape[:2] != (CROP_SIZE, CROP_SIZE):
                crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)

            raw_vp = detect_viewport_robust(crop)
            vp = stabilizer.update(raw_vp) if stabilizer else raw_vp

            dets = _predict_detections(
                model, crop, conf, iou, imgsz, device, names, allowed_ids, vp,
                viewport_margin_px,
            )
            measured = {d["class_id"] for d in dets}

            if kalman:
                dets = _apply_kalman(
                    dets, tracks, crop.shape[1], crop.shape[0],
                    kalman_base_missing, kalman_overlap_extra, kalman_overlap_dist,
                )
                for d in dets:
                    d["inside_viewport"] = _inside_viewport(
                        d["bbox_xyxy"],
                        vp,
                        img_w=crop.shape[1],
                        img_h=crop.shape[0],
                        safe_margin_px=viewport_margin_px,
                    )

            dets = _apply_temporal_filter(dets, measured, temporal, min_hits)
            dets = _dedupe_by_class(dets)

            if locker is not None:
                locker.update(dets, frame_idx)
                dets = locker.filter(dets)

            _assign_class_names(dets, names)

            record = {
                "frame_idx": frame_idx,
                "timestamp_ms": float(cap.get(cv2.CAP_PROP_POS_MSEC)),
                "viewport": [int(v) for v in vp] if vp else None,
                "detections": dets,
                "locked_champion_ids": list(locker.locked) if locker else None,
                **_summarize_sets(dets, names),
            }
            fp.write(json.dumps(record) + "\n")

            if writer is not None:
                writer.write(_draw_overlay(crop, vp, dets))

            frame_idx += 1
            processed += 1
            if max_frames is not None and processed >= max_frames:
                break

    cap.release()
    if writer is not None:
        writer.release()


# ────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Minimap pipeline: crop → viewport → YOLO → in/out classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    p.add_argument("--input", required=True, help="Video, image, or directory path")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="YOLO weights (.pt)")
    p.add_argument("--output_dir", default=str(ROOT / "results" / "pipeline_run"),
                   help="Output directory")
    p.add_argument("--device", default="0")

    # Detection
    p.add_argument("--imgsz", type=int, default=CROP_SIZE, help="YOLO input size")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")

    # Dataset / filtering
    p.add_argument("--dataset", default=None, help="Dataset name (resolves class names)")
    p.add_argument("--data_config", default=None, help="Explicit dataset config YAML")
    p.add_argument("--game_id", default=None, help="Riot match ID for 10-champ filtering")

    # Video
    p.add_argument("--visualize", action="store_true", help="Write overlay video / images")
    p.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    p.add_argument("--max_frames", type=int, default=None, help="Stop after N frames")

    # Viewport stabilisation (on by default for video)
    p.add_argument("--no_stabilize", action="store_true",
                   help="Disable viewport stabilisation")

    # Kalman tracking (on by default for video)
    p.add_argument("--no_kalman", action="store_true",
                   help="Disable per-champion Kalman smoothing")
    p.add_argument("--kalman_base_missing", type=int, default=5,
                   help="Frames to hold a missing champion before dropping (~83ms @60fps)")
    p.add_argument("--kalman_overlap_extra", type=int, default=300,
                   help="Extra hold frames when occluded by another champion (~5s @60fps)")
    p.add_argument("--kalman_overlap_dist", type=float, default=DEFAULT_OVERLAP_DIST,
                   help="Centre distance to consider overlap (px)")
    p.add_argument("--min_hits", type=int, default=30,
                   help="Consecutive detections to confirm a champion (~0.5s @60fps)")

    # Inside/outside classification
    p.add_argument(
        "--viewport_margin_px",
        type=int,
        default=DEFAULT_VIEWPORT_MARGIN_PX,
        help="Inset viewport by N px before counting a champ as inside (ignores edge region)",
    )

    # Champion locking (on by default)
    p.add_argument("--no_lock_champs", action="store_true",
                   help="Disable champion locking")
    p.add_argument("--max_champs", type=int, default=10, help="Max champions to lock")
    p.add_argument("--lock_delay", type=int, default=60, help="Frames before locking begins (~1s @60fps)")
    p.add_argument("--lock_min_obs", type=int, default=45,
                   help="Min measured frames before a champion can lock")
    p.add_argument("--lock_min_density", type=float, default=0.30,
                   help="Min fraction of frames since first seen to lock (0-1)")

    return p


def main():
    args = _build_parser().parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    config_path = _resolve_data_config(args.dataset, args.data_config)
    model = YOLO(args.weights)
    names = _load_class_names(config_path)
    if not names:
        names = _model_class_names(model)
    allowed_ids = _get_allowed_class_ids(args.game_id, config_path)

    write_metadata(output_dir, {
        "mode": "pipeline",
        "input": str(input_path),
        "weights": args.weights,
        "device": args.device,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "data_config": str(config_path) if config_path else None,
        "game_id": args.game_id,
        "git": get_git_state(ROOT),
    })

    # Image mode
    if input_path.is_dir() or input_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        process_images(
            model, input_path, output_dir, names, allowed_ids,
            args.conf, args.iou, args.device, args.imgsz, args.visualize,
            max(0, int(args.viewport_margin_px)),
        )
        return

    # Video mode
    if input_path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        process_video(
            model, input_path, output_dir, names, allowed_ids,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            imgsz=args.imgsz,
            visualize=args.visualize,
            stabilize=not args.no_stabilize,
            stride=max(1, args.stride),
            max_frames=args.max_frames,
            kalman=not args.no_kalman,
            kalman_base_missing=max(0, args.kalman_base_missing),
            kalman_overlap_extra=max(0, args.kalman_overlap_extra),
            kalman_overlap_dist=max(0.0, args.kalman_overlap_dist),
            min_hits=max(1, args.min_hits),
            lock_champs=not args.no_lock_champs,
            max_champs=max(1, args.max_champs),
            lock_delay=max(0, args.lock_delay),
            lock_min_obs=max(1, args.lock_min_obs),
            lock_min_density=max(0.0, min(1.0, args.lock_min_density)),
            viewport_margin_px=max(0, int(args.viewport_margin_px)),
        )
        return

    print("[error] Unsupported input type. Provide a video, image, or directory.")
    sys.exit(1)


if __name__ == "__main__":
    main()
