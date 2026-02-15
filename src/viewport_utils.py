"""Shared viewport utilities used by minimap scripts."""

from __future__ import annotations

from typing import Optional

import numpy as np


class ViewportStabilizer:
    """Smooth viewport detections across frames.

    - EMA on position and size to remove jitter.
    - Confirmation gate for large jumps.
    - Aspect-ratio / size outlier rejection.
    - Optional edge-aware behavior for partially clipped boxes.
    """

    EXPECTED_ASPECT = 16.0 / 9.0
    ASPECT_TOL = 0.35

    def __init__(
        self,
        *,
        img_size: int,
        deadzone: float = 3.0,
        confirmation_frames: int = 12,
        pos_alpha: float = 0.25,
        size_alpha: float = 0.08,
        size_reject_ratio: float = 0.35,
        min_viewport_w: float = 50.0,
        large_jump_px: float = 20.0,
        pending_merge_dist_px: float = 15.0,
        clear_after_missing: int = 60,
        edge_aware: bool = True,
    ):
        self.img_size = img_size
        self.deadzone = deadzone
        self.confirm_limit = confirmation_frames
        self.pos_alpha = pos_alpha
        self.size_alpha = size_alpha
        self.size_reject_ratio = size_reject_ratio
        self.min_viewport_w = min_viewport_w
        self.large_jump_px = large_jump_px
        self.pending_merge_dist_px = pending_merge_dist_px
        self.clear_after_missing = clear_after_missing
        self.edge_aware = edge_aware

        self.stable_box: Optional[list[float]] = None
        self.stable_size: Optional[list[float]] = None
        self.pending_box: Optional[list[float]] = None
        self.pending_counter = 0
        self.missing_streak = 0

    def _valid_aspect(self, w: float, h: float) -> bool:
        if h < 1:
            return False
        return abs(w / h - self.EXPECTED_ASPECT) / self.EXPECTED_ASPECT < self.ASPECT_TOL

    def _size_ok(self, w: float, h: float) -> bool:
        if self.stable_size is None:
            return True
        sw, sh = self.stable_size
        if sw < 1 or sh < 1:
            return True
        return (
            abs(w - sw) / sw < self.size_reject_ratio
            and abs(h - sh) / sh < self.size_reject_ratio
        )

    def _smooth_size(self, w: float, h: float) -> tuple[int, int]:
        if self.stable_size is None:
            self.stable_size = [float(w), float(h)]
        else:
            a = self.size_alpha
            self.stable_size[0] += a * (w - self.stable_size[0])
            self.stable_size[1] += a * (h - self.stable_size[1])
        return int(round(self.stable_size[0])), int(round(self.stable_size[1]))

    def _at_edge(self, x: float, y: float, w: float, h: float) -> bool:
        m = 3
        return x < -m or y < -m or x + w > self.img_size + m or y + h > self.img_size + m

    def _use_stable_size(self) -> tuple[int, int]:
        return int(round(self.stable_size[0])), int(round(self.stable_size[1]))

    def update(self, raw_box) -> Optional[list[int]]:
        if raw_box is None:
            self.missing_streak += 1
            self.pending_counter = 0
            self.pending_box = None
            if self.missing_streak > self.clear_after_missing:
                self.stable_box = None
                self.stable_size = None
            return self.stable_box

        self.missing_streak = 0
        x, y, w, h = raw_box
        edge = self.edge_aware and self._at_edge(x, y, w, h)

        if w < self.min_viewport_w or not self._valid_aspect(w, h) or not self._size_ok(w, h):
            if edge and self.stable_size is not None:
                w, h = self.stable_size
            else:
                return self.stable_box

        if self.stable_box is None:
            sw, sh = self._smooth_size(w, h)
            self.stable_box = [int(x), int(y), sw, sh]
            return self.stable_box

        sw, sh = (
            self._use_stable_size()
            if (edge and self.stable_size)
            else self._smooth_size(w, h)
        )

        dist = np.hypot(x - self.stable_box[0], y - self.stable_box[1])
        if dist < self.large_jump_px:
            self.pending_counter = 0
            self.pending_box = None
            if dist >= self.deadzone:
                a = self.pos_alpha
                self.stable_box[0] = int(round(self.stable_box[0] * (1 - a) + x * a))
                self.stable_box[1] = int(round(self.stable_box[1] * (1 - a) + y * a))
            self.stable_box[2], self.stable_box[3] = sw, sh
            return self.stable_box

        if (
            self.pending_box is not None
            and np.hypot(x - self.pending_box[0], y - self.pending_box[1]) < self.pending_merge_dist_px
        ):
            self.pending_counter += 1
            a = 0.5
            self.pending_box[0] = int(round(self.pending_box[0] * (1 - a) + x * a))
            self.pending_box[1] = int(round(self.pending_box[1] * (1 - a) + y * a))
            self.pending_box[2], self.pending_box[3] = int(w), int(h)
        else:
            self.pending_box = [int(x), int(y), int(w), int(h)]
            self.pending_counter = 1

        if self.pending_counter >= self.confirm_limit:
            sw2, sh2 = (
                self._use_stable_size()
                if (edge and self.stable_size)
                else self._smooth_size(self.pending_box[2], self.pending_box[3])
            )
            self.stable_box = [self.pending_box[0], self.pending_box[1], sw2, sh2]
            self.pending_counter = 0
            self.pending_box = None

        return self.stable_box