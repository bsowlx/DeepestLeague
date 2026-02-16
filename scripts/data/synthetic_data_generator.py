"""
Synthetic minimap data generator.

Pipeline summary:
1) Load map/champion/effect assets from assets/.
2) Build a randomized minimap base (map variant, fog, map objects, vision).
3) Sample 10 champion icons, place them, and write YOLO labels.
4) Draw observer viewport rectangle and optional ping/recall/tp effects.
5) Apply optional augmentation and JPEG quality reduction noise.
6) Save image, labels, viewport labels, and optional dataset YAML.

Key defaults:
- Image size: 256
- CLI split sizes: train 1000 / val 100 / test 100
- Noise path: JPEG compression enabled by default (disable with --no-noise)
"""

import argparse
import os
import datetime
import random
import multiprocessing
from pathlib import Path

import numpy as np
import cv2
import albumentations as A


DEFAULT_WORKERS = max(1, min(8, os.cpu_count() or 1))


# ---------- Paths ----------
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

# ---------- Albumentations transforms ----------
BACKGROUND_AUGMENT = A.Compose([
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 0.5), p=1),
        A.Downscale((0.9, 1.0), p=1),
    ], p=1)
])

ICON_AUGMENT = A.Compose([
    A.OneOf([
        A.MotionBlur(direction_range=[-0.5, 0.5], p=1),
        A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 0.5), p=1),
        A.Downscale((0.9, 1.0), p=1),
    ], p=0.15),
    A.OneOf([
        A.GridDistortion(num_steps=5, distort_limit=0.03, p=1),
        A.OpticalDistortion(distort_limit=0.03, p=1),
    ], p=0.01),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.05)
])

def _worker_entry_point(args):
    """Worker entrypoint used by multiprocessing pool."""
    generator_instance, split_name, index, allow_overlap = args
    seed = (os.getpid() * int(datetime.datetime.now().microsecond)) % 123456789
    np.random.seed(seed)
    random.seed(seed)

    cv2.setNumThreads(0)

    return generator_instance._generate_single_sample(split_name, index, allow_overlap)


class SyntheticDataGenerator:
    """Generates synthetic minimap datasets for champion detection experiments."""

    # Cosmetic / geometry constants
    RED_TEAM_CIRCLE_BGR = (61, 61, 232)
    BLUE_TEAM_CIRCLE_BGR = (220, 150, 0)

    OBSERVER_RECTANGLE_WH = [75, 40]
    CV_THICKNESS_WORK_SIZE = 256 * 3

    RECALL_PROB = 0.12
    TP_PROB = 0.1
    TP_ON_CHAMP_PROB = 0.25

    def __init__(self):
        self.resource_dir = os.path.join(PROJECT_ROOT, "assets")
        self.split_names = ["train", "val", "test"]

        self._load_resources()

        self.output_image_size = 256
        self.apply_fog_of_war = True
        self.draw_map_objects = True
        self.use_background_augment = True
        self.use_icon_augment = False
        self.circle_detection_type = 0
        self.thickness_correction = True

        self.viewport_sim = False
        self.viewport_size_factors = [0.80, 0.90, 1.00, 1.10, 1.25, 1.40]

        self.ping_attach_prob = 0.4
        self.ping_overlay_colors = [(177, 145, 53), (39, 39, 148)]
        self.ping_overlays = self._load_ping_overlays()
        self.recall_tp_overlays = self._load_recall_tp_overlays()

        self.champion_names = list(self.all_champion_names)
        self.num_champions = len(self.champion_names)
        assert self.num_champions >= 10, "Selected champions are fewer than 10"

    # ---------------- Public API ----------------
    def generate_data(
        self,
        n_train: int,
        n_val: int,
        n_test: int,
        use_hsv_augmentation: bool = False,
        use_noise: bool = True,
        use_background_augment: bool = True,
        use_icon_augment: bool = False,
        yaml: bool = True,
        allow_icon_overlap: bool = True,
        use_recall_tp: bool = True,
        viewport_sim: bool = False,
        output_image_size: int | None = None,
        output_dir: str | None = None,
        dataset_name: str | None = None,
        num_workers: int | None = None,
    ):
        """Generate train/val/test splits and save images/labels to disk."""
        self.n_train = int(n_train)
        self.n_val = int(n_val)
        self.n_test = int(n_test)
        self.use_hsv_augmentation = bool(use_hsv_augmentation)
        self.use_noise = bool(use_noise)
        self.use_background_augment = bool(use_background_augment)
        self.use_icon_augment = bool(use_icon_augment)
        self.use_recall_tp = bool(use_recall_tp)
        self.viewport_sim = bool(viewport_sim)
        if output_image_size is not None:
            self.output_image_size = int(output_image_size)
        self.output_dir = output_dir
        self.dataset_name = dataset_name

        workers = DEFAULT_WORKERS if num_workers is None else int(num_workers)
        workers = max(1, workers)

        self._make_output_directories()

        self.champion_icon_templates = self._make_circular_icon_templates(self.champion_names)

        self.circle_radius_pixels = int(np.round(self.icon_size_pixels * 0.5 + 0.5 + 1.0))
        self.yolo_box_size_norm = 2 * (self.circle_radius_pixels + 2) / self.minimap_canvas_size

        tasks = []
        for split_size, split_name in zip([self.n_train, self.n_val, self.n_test], self.split_names):
            for index in range(split_size):
                tasks.append((self, split_name, index, allow_icon_overlap))

        total_tasks = len(tasks)
        print(f"Starting generation of {total_tasks} images using {workers} cores...")

        chunk_size = max(1, total_tasks // (workers * 4))

        with multiprocessing.Pool(processes=workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(_worker_entry_point, tasks, chunksize=chunk_size)):
                if i % 100 == 0:
                    print(f"\rProgress: {i}/{total_tasks} ({(i/total_tasks)*100:.1f}%)", end="")

        print(f"\rProgress: {total_tasks}/{total_tasks} (100.0%)")
        print("Synthetic Data generation completed.")

        if yaml:
            self._write_dataset_yaml()

    def _generate_single_sample(self, split_name, index, allow_icon_overlap):
        """Generate one synthetic sample for a split and persist outputs."""
        base_minimap = self._prepare_base_minimap()
        composed_canvas = base_minimap.copy()

        chosen_champ_indices = np.random.permutation(self.num_champions)[:10]
        chosen_icons = np.array(self.champion_icon_templates)[chosen_champ_indices]
        chosen_icons = np.array([self._augment_icon_bgra(icon) for icon in chosen_icons])

        if self.circle_detection_type == 1:
            class_ids = [0] * 10
        elif self.circle_detection_type == 2:
            class_ids = [0] * 5 + [1] * 5
        else:
            class_ids = list(chosen_champ_indices)

        if self.use_hsv_augmentation:
            hsv_augmented_icons = []
            for icon_rgba in chosen_icons:
                bgr = icon_rgba[:, :, :3]
                alpha = icon_rgba[:, :, 3:]
                bgr_hsv_aug = self._augment_hsv(bgr)
                hsv_augmented_icons.append(np.concatenate((bgr_hsv_aug, alpha), axis=2))
            chosen_icons = np.array(hsv_augmented_icons)

        if allow_icon_overlap:
            top_left_positions, center_norm_positions, center_round_positions = self._sample_overlapping_icon_positions(
                n_icons=10,
                radius=int(np.round(self.icon_size_pixels * 0.5)),
                min_offset_ratio=0.5,
            )
        else:
            top_left_positions, center_norm_positions, center_round_positions = self._sample_nonoverlapping_icon_positions(10)

        active_indices = list(range(10))
        if np.random.uniform() < 0.05:
            k = np.random.randint(1, 10)
            active_indices = sorted(np.random.choice(10, size=k, replace=False).tolist())

        chosen_icons = [chosen_icons[i] for i in active_indices]
        class_ids = [class_ids[i] for i in active_indices]
        top_left_positions = [top_left_positions[i] for i in active_indices]
        center_norm_positions = [center_norm_positions[i] for i in active_indices]
        center_round_positions = [center_round_positions[i] for i in active_indices]

        if self.use_recall_tp:
            composed_canvas = self._apply_recall_tp_effects(
                composed_canvas,
                center_round_positions,
            )

        yolo_labels = []
        for i_idx, icon_rgba in enumerate(chosen_icons):
            icon_top_left = top_left_positions[i_idx]
            center_norm = center_norm_positions[i_idx]
            center_px_round = center_round_positions[i_idx]

            original_slot = active_indices[i_idx]

            composed_canvas = self._overlay_rgba(composed_canvas, icon_rgba, icon_top_left[0], icon_top_left[1])

            composed_canvas = self._draw_team_circle(
                composed_canvas,
                center_px_round,
                self.circle_radius_pixels,
                original_slot
            )

            yolo_labels.append((
                class_ids[i_idx],
                center_norm[0], center_norm[1],
                self.yolo_box_size_norm, self.yolo_box_size_norm
            ))

        resized_canvas = cv2.resize(
            composed_canvas,
            (self.output_image_size, self.output_image_size),
            interpolation=cv2.INTER_AREA
        )

        resized_canvas, viewport_box = self._draw_observer_rectangle(resized_canvas)
        resized_canvas = self._apply_random_pings(resized_canvas)

        if self.use_background_augment:
            resized_canvas = BACKGROUND_AUGMENT(image=resized_canvas)["image"]

        if self.use_noise:
            resized_canvas = self._apply_quality_reduction(resized_canvas)

        image_out_path = os.path.join(self.save_dir, split_name, "images", f"{index}.png")
        label_out_path = os.path.join(self.save_dir, split_name, "labels", f"{index}.txt")
        viewport_out_path = None
        if self.viewport_sim:
            viewport_out_path = os.path.join(self.save_dir, split_name, "viewport_labels", f"{index}.txt")

        cv2.imwrite(image_out_path, resized_canvas)
        with open(label_out_path, "w") as f:
            for label in yolo_labels:
                f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(*label) + "\n")

        if self.viewport_sim and viewport_box is not None and viewport_out_path is not None:
            vx, vy, vw, vh = viewport_box
            with open(viewport_out_path, "w") as f:
                f.write(f"{int(vx)} {int(vy)} {int(vw)} {int(vh)}\n")

    # ---------------- Resource loading ----------------
    def _load_resources(self):
        """
        Loads:
          - champion icons from assets/champs/*.png
          - minimap variants from assets/map/*.png
          - map object overlays from assets/icons/*.png
        """
        boundary_crop = 2
        champion_dir = os.path.join(self.resource_dir, "champs")
        champion_files = os.listdir(champion_dir)

        self.champion_icons_original = {}
        self.all_champion_names = []

        for filename in sorted(champion_files):
            if not filename.lower().endswith(".png"):
                continue
            champion_name = filename[:-4]
            icon_path = os.path.join(champion_dir, f"{champion_name}.png")
            icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
            if icon is None:
                continue
            self.champion_icons_original[champion_name] = icon[boundary_crop:-boundary_crop, boundary_crop:-boundary_crop]
            self.all_champion_names.append(champion_name)

        print(f"Load champion images -- {len(self.all_champion_names)} images")

        self.minimap_variants, self.minimap_variant_weights = self._load_minimap_images()
        self.minimap_bgr = self.minimap_variants[0]
        self.minimap_canvas_size = self.minimap_bgr.shape[0]

        self.icon_size_pixels = int(np.round(self.minimap_canvas_size / 512 * 43))
        self.map_object_overlays = self._load_map_objects()

    def _load_minimap_images(self):
        map_dir = os.path.join(self.resource_dir, "map")
        groups = {
            "usual": {
                "weight": 0.7,
                "files": [
                    "map.png",
                    "map_mountain_baron2.png",
                    "map_ocean_baron2.png",
                    "map_cloud_baron1.png",
                ],
            },
            "hextech": {
                "weight": 0.3,
                "files": [
                    "map_hextech.png",
                    "map_hextech_baron3.png",
                ],
            },
        }

        images = []
        weights = []
        for group in groups.values():
            group_files = []
            for filename in group["files"]:
                path = os.path.join(map_dir, filename)
                if not os.path.exists(path):
                    continue
                img = cv2.imread(path)
                if img is None:
                    continue
                group_files.append(img)

            if len(group_files) == 0:
                continue

            per_file_weight = float(group["weight"]) / float(len(group_files))
            for img in group_files:
                images.append(img)
                weights.append(per_file_weight)

        if len(images) == 0:
            raise FileNotFoundError(f"Minimap not found in: {map_dir}")

        total = float(sum(weights)) or 1.0
        weights = [w / total for w in weights]
        print(f"Load minimap images -- {len(images)} variants")
        return images, weights

    def _choose_minimap_bgr(self):
        if len(self.minimap_variants) == 1:
            return self.minimap_variants[0]
        idx = int(np.random.choice(len(self.minimap_variants), p=self.minimap_variant_weights))
        return self.minimap_variants[idx]

    def _load_map_objects(self):
        overlays = {}
        object_dir = os.path.join(self.resource_dir, "icons")
        filelist = [f for f in os.listdir(object_dir) if f.lower().endswith(".png")]
        for filename in filelist:
            path = os.path.join(object_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            img_scaled = cv2.resize(img, (0, 0), fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC)
            overlays[filename[:-4]] = img_scaled
        print("Load object images -- done")
        return overlays
    
    def _load_ping_overlays(self):
        ping_dir = os.path.join(self.resource_dir, "pings")
        overlays = []
        if not os.path.exists(ping_dir):
            print(f"[WARN] ping_dir not found: {ping_dir}")
            return overlays

        filelist = sorted(os.listdir(ping_dir))
        for fname in filelist:
            if not fname.lower().endswith(".png"):
                continue
            img = cv2.imread(os.path.join(ping_dir, fname), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            overlays.append(img)
        print("Load ping overlays -- done")
        return overlays

    def _load_recall_tp_overlays(self):
        """
        Recall / teleport overlay assets.
        Expected files (assets/icons/):
          - red_recall.png / blue_recall.png
          - red_tp.png / blue_tp.png
        """
        candidates = [
            "red_recall.png",
            "blue_recall.png",
            "red_tp.png",
            "blue_tp.png",
        ]

        overlays = {}
        search_dirs = [
            self.resource_dir,
            os.path.join(self.resource_dir, "effects"),
            os.path.join(self.resource_dir, "icons"),
        ]

        for name in candidates:
            for d in search_dirs:
                path = os.path.join(d, name)
                if os.path.exists(path):
                    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        overlays[name] = img
                    break

        if len(overlays) == 0:
            print("[WARN] recall/tp overlays not found")
        else:
            print(f"Load recall/tp overlays -- {len(overlays)} images")
        return overlays

    # ---------------- Output dirs / yaml ----------------
    def _make_output_directories(self):
        base_dir = self.output_dir or os.path.join(PROJECT_ROOT, "data", "trials")
        if not os.path.isabs(base_dir):
            base_dir = os.path.join(PROJECT_ROOT, base_dir)

        if self.dataset_name:
            folder_name = self.dataset_name
        else:
            folder_name = f"lol_minimap_{self.output_image_size}_{datetime.datetime.now().strftime('%y%m%d_%H%M')}"

        self.save_dir = os.path.join(base_dir, folder_name)

        for split_name in self.split_names:
            os.makedirs(os.path.join(self.save_dir, split_name, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, split_name, "labels"), exist_ok=True)
            if self.viewport_sim:
                os.makedirs(os.path.join(self.save_dir, split_name, "viewport_labels"), exist_ok=True)

    def _write_dataset_yaml(self):
        yaml_path = os.path.join(self.save_dir, "config.yaml")
        with open(yaml_path, "w") as f:
            f.write(f"path: {os.path.basename(self.save_dir)}\n")
            f.write("train: train/images\n")
            f.write("val: val/images\n")
            f.write("test: test/images\n")
            f.write(f"nc: {self.num_champions}\n")
            f.write("names: [{}]\n".format(", ".join([f"'{c}'" for c in self.champion_names])))

    # ---------------- Base canvas preparation ----------------
    def _prepare_base_minimap(self):
        """
        Creates base canvas:
          minimap -> optional fog -> optional map objects
        """
        base_minimap = self._choose_minimap_bgr()
        if self.apply_fog_of_war:
            fog_mask = self._make_fog_of_war_mask()
            fogged = self._blend_fog(base_minimap, fog_mask)
        else:
            fogged = base_minimap.copy()

        if self.draw_map_objects:
            return self._draw_map_objects_random(fogged)
        return fogged

    # ---------------- Icon template creation ----------------
    def _make_circular_icon_templates(self, champion_names):
        """
        Creates circular-masked RGBA icon templates for each champion.
        """
        circle_size = self.icon_size_pixels + 2
        circular_alpha_mask = self._create_circular_alpha_mask(circle_size, circle_size, circle_size / 2)

        templates = []
        for name in champion_names:
            original = self.champion_icons_original[name]

            # Ensure RGBA
            if original.shape[2] == 4:
                icon_bgra = original.copy()
            else:
                icon_bgra = cv2.cvtColor(original, cv2.COLOR_BGR2BGRA)

            # Resize and apply circular alpha mask
            icon_bgra = cv2.resize(icon_bgra, (circle_size, circle_size), interpolation=cv2.INTER_AREA)
            icon_bgra = (icon_bgra * circular_alpha_mask).astype(icon_bgra.dtype)

            templates.append(icon_bgra)

        return templates

    def _create_circular_alpha_mask(self, height, width, radius):
        """
        Returns (H, W, 4) uint8 mask where alpha is 0 outside circle.
        (Kept same brute-force logic style to avoid subtle differences.)
        """
        center_y = height / 2 - 0.5
        center_x = width / 2 - 0.5
        mask = np.ones((height, width, 4), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                x = center_x - i
                y = center_y - j
                if x**2 + y**2 > radius**2:
                    mask[j, i, 3] = 0
        return mask

    # ---------------- Icon placement sampling ----------------
    def _random_top_left_in_map(self, scale=1.0, n=2):
        min_val = self.icon_size_pixels * 1.5 * scale
        max_val = (self.minimap_canvas_size - self.icon_size_pixels * 2.0) * scale
        return np.random.randint(min_val, max_val, n)

    def _random_top_left_for_overlay(self, overlay):
        h, w = overlay.shape[:2]
        max_x = max(0, self.minimap_canvas_size - w)
        max_y = max(0, self.minimap_canvas_size - h)
        x = int(np.random.randint(0, max_x + 1)) if max_x > 0 else 0
        y = int(np.random.randint(0, max_y + 1)) if max_y > 0 else 0
        return x, y

    def _sample_nonoverlapping_icon_positions(self, n_icons=1, radius=None):
        """
        Returns:
          top_left_positions: list[np.array([x,y])]
          center_norm_positions: list[np.array([x_norm,y_norm])]
          center_round_positions: list[tuple(int,int)]
        """
        if n_icons == 1:
            top_left = self._random_top_left_in_map()
            center_xy = top_left + self.icon_size_pixels * 0.5 + 0.5
            center_norm = center_xy / self.minimap_canvas_size
            center_round = tuple(np.round(center_xy).astype(int))
            return top_left, center_norm, center_round

        if radius is None:
            radius = int(np.round(self.icon_size_pixels * 0.5))

        occupancy = np.zeros(self.minimap_bgr.shape, dtype=np.uint8)

        top_left_positions = []
        while len(top_left_positions) < n_icons:
            pos = self._random_top_left_in_map()
            if occupancy[pos[1], pos[0], 0] == 0:
                top_left_positions.append(pos)
                occupancy = cv2.circle(occupancy, (pos[0], pos[1]), radius, (255, 255, 255), -1)

        center_positions = [p + self.icon_size_pixels * 0.5 + 0.5 for p in top_left_positions]
        center_norm_positions = [c / self.minimap_canvas_size for c in center_positions]
        center_round_positions = [tuple(np.round(c).astype(int)) for c in center_positions]

        return top_left_positions, center_norm_positions, center_round_positions
    
    def _sample_overlapping_icon_positions(
        self,
        n_icons: int,
        radius: int | None = None,
        min_offset_ratio: float = 0.5,
        max_tries: int = 5000,
    ):
        if radius is None:
            radius = int(np.round(self.icon_size_pixels * 0.5))

        min_offset = float(min_offset_ratio) * float(radius)
        min_offset_sq = min_offset * min_offset

        top_left_positions: list[np.ndarray] = []
        tries = 0
        while len(top_left_positions) < n_icons:
            tries += 1
            if tries > max_tries:
                min_offset_sq *= 0.8
                tries = 0

            pos = self._random_top_left_in_map()

            ok = True
            for prev in top_left_positions:
                dx = float(pos[0] - prev[0])
                dy = float(pos[1] - prev[1])
                if dx * dx + dy * dy < min_offset_sq:
                    ok = False
                    break

            if ok:
                top_left_positions.append(pos)

        center_positions = [p + self.icon_size_pixels * 0.5 + 0.5 for p in top_left_positions]
        center_norm_positions = [c / self.minimap_canvas_size for c in center_positions]
        center_round_positions = [tuple(np.round(c).astype(int)) for c in center_positions]

        return top_left_positions, center_norm_positions, center_round_positions

    # ---------------- Drawing / compositing ----------------
    def _overlay_rgba(self, background_bgr, overlay_rgba, x, y):
        bg_h, bg_w = background_bgr.shape[:2]
        if x >= bg_w or y >= bg_h:
            return background_bgr

        oh, ow = overlay_rgba.shape[:2]

        # Handle negative offsets by cropping the overlay
        if x < 0:
            overlay_rgba = overlay_rgba[:, -x:]
            ow = overlay_rgba.shape[1]
            x = 0
        if y < 0:
            overlay_rgba = overlay_rgba[-y:, :]
            oh = overlay_rgba.shape[0]
            y = 0

        if x + ow > bg_w:
            ow = bg_w - x
            overlay_rgba = overlay_rgba[:, :ow]
        if y + oh > bg_h:
            oh = bg_h - y
            overlay_rgba = overlay_rgba[:oh]

        # Ensure alpha channel exists
        if overlay_rgba.shape[2] < 4:
            alpha = np.ones((overlay_rgba.shape[0], overlay_rgba.shape[1], 1), dtype=overlay_rgba.dtype) * 255
            overlay_rgba = np.concatenate([overlay_rgba, alpha], axis=2)

        overlay_rgb = overlay_rgba[..., :3]
        alpha = overlay_rgba[..., 3:] / 255.0

        background_bgr[y:y+oh, x:x+ow, :3] = (
            (1.0 - alpha) * background_bgr[y:y+oh, x:x+ow, :3] + alpha * overlay_rgb
        )
        return background_bgr

    def _draw_team_circle(self, canvas_bgr, center_xy, radius, icon_slot_index):
        """Draw team-colored circle around a champion icon center."""
        if self.thickness_correction:
            enlarged = cv2.resize(
                canvas_bgr,
                (self.CV_THICKNESS_WORK_SIZE, self.CV_THICKNESS_WORK_SIZE),
                interpolation=cv2.INTER_CUBIC
            )
            center_xy = tuple(np.round(1.5 * np.array(center_xy)).astype(int))
            radius = int(1.5 * radius)
        else:
            enlarged = np.copy(canvas_bgr)

        color = self.RED_TEAM_CIRCLE_BGR if icon_slot_index < 5 else self.BLUE_TEAM_CIRCLE_BGR
        enlarged = cv2.circle(enlarged, center_xy, radius, color, 2)

        if self.thickness_correction:
            canvas_bgr = cv2.resize(
                enlarged,
                (self.minimap_canvas_size, self.minimap_canvas_size),
                interpolation=cv2.INTER_AREA
            )
            return canvas_bgr

        return enlarged

    def _sample_viewport_wh_px(self) -> tuple[int, int]:
        """Sample viewport (observer rectangle) size in *output-image* pixels."""
        base_imgsz = 256
        scale_factor = self.output_image_size / base_imgsz
        wh = (np.array(self.OBSERVER_RECTANGLE_WH, dtype=np.float32) * scale_factor)
        if self.viewport_sim:
            wh = wh * float(random.choice(self.viewport_size_factors))
        w, h = int(round(wh[0])), int(round(wh[1]))
        return max(2, w), max(2, h)


    def _random_top_left_for_viewport(self, vp_w: int, vp_h: int) -> tuple[int, int]:
        """Sample a top-left so the viewport stays inside the output canvas."""
        max_x = max(0, self.output_image_size - vp_w)
        max_y = max(0, self.output_image_size - vp_h)
        x = int(np.random.randint(0, max_x + 1)) if max_x > 0 else 0
        y = int(np.random.randint(0, max_y + 1)) if max_y > 0 else 0
        return x, y


    def _draw_observer_rectangle(self, canvas_bgr):
        vp_w, vp_h = self._sample_viewport_wh_px()
        x, y = self._random_top_left_for_viewport(vp_w, vp_h)
        top_left = (int(x), int(y))
        bottom_right = (int(x + vp_w), int(y + vp_h))
        out = cv2.rectangle(canvas_bgr, top_left, bottom_right, (255, 255, 255), 2)
        return out, (int(x), int(y), int(vp_w), int(vp_h))

    def _rotate_rgba(self, rgba, angle_deg):
        h, w = rgba.shape[:2]
        center = (w / 2, h / 2)
        mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        return cv2.warpAffine(
            rgba,
            mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

    # ---------------- Fog of war ----------------
    def _make_fog_of_war_mask(self):
        fog = np.zeros(self.minimap_bgr.shape, dtype=np.uint8)
        n_circles = 40

        radius = np.random.randint(
            int(self.icon_size_pixels * 0.5),
            int(self.icon_size_pixels * 1.5),
            n_circles
        )
        x = np.random.randint(
            int(self.icon_size_pixels * 0.5),
            int(self.minimap_canvas_size - self.icon_size_pixels * 0.5),
            n_circles
        )
        y = np.random.randint(
            int(self.icon_size_pixels * 0.5),
            int(self.minimap_canvas_size - self.icon_size_pixels * 0.5),
            n_circles
        )

        for r, xt, yt in zip(radius, x, y):
            fog = cv2.circle(fog, (xt, yt), r, (255, 255, 255), -1)

        return cv2.GaussianBlur(fog, (21, 21), 0)

    def _blend_fog(self, minimap_bgr, fog_mask_bgr):
        alpha_max = 0.65
        gray = cv2.cvtColor(fog_mask_bgr, cv2.COLOR_BGR2GRAY)
        alpha = (1.0 - gray / 255.0) * alpha_max

        black = np.zeros(fog_mask_bgr.shape, dtype=np.uint8)
        out = np.empty_like(minimap_bgr)

        for i in range(3):
            out[:, :, i] = (1 - alpha) * minimap_bgr[:, :, i] + alpha * black[:, :, i]
        return out.astype(np.uint8)

    # ---------------- Map objects ----------------
    def _draw_map_objects_random(self, canvas_bgr):
        """Overlay static map entities and randomized objectives/vision objects."""
        # probabilities (same meaning as your snippet)
        third_tower_skip_prob = 0.2     # if rand > 0.2 => place (80%)
        second_tower_skip_prob = 0.4    # if rand > 0.4 => place (60%) BUT only if stage-3 exists
        first_tower_skip_prob = 0.4     # if rand > 0.4 => place (60%) BUT only if stage-2 and stage-3 exist
        jungle_rate = 0.2               # used as interval thresholds
        shop_skip_prob = 0.0            # always place when available

        ov = self.map_object_overlays
        out = canvas_bgr

        # --- Nexus (always) ---
        out = self._overlay_rgba(out, ov["red_nexus"], 419, 45)
        out = self._overlay_rgba(out, ov["blue_nexus"], 47, 408)

        # --- Shop icons (if available) ---
        if "shop" in ov:
            if np.random.rand() > shop_skip_prob:
                out = self._overlay_rgba(out, ov["shop"], 449, -14)
            if np.random.rand() > shop_skip_prob:
                out = self._overlay_rgba(out, ov["shop"], -19, 445)

        # --- Inhibitors (always) ---
        out = self._overlay_rgba(out, ov["red_exhibitor"], 376, 28)
        out = self._overlay_rgba(out, ov["red_exhibitor"], 391, 92)
        out = self._overlay_rgba(out, ov["red_exhibitor"], 455, 108)
        out = self._overlay_rgba(out, ov["blue_exhibitor"], 26, 377)
        out = self._overlay_rgba(out, ov["blue_exhibitor"], 97, 388)
        out = self._overlay_rgba(out, ov["blue_exhibitor"], 105, 455)

        # --- Tower flags ---
        rt2 = rt3 = rm2 = rm3 = rb2 = rb3 = False
        bt2 = bt3 = bm2 = bm3 = bb2 = bb3 = False

        # --- Normal towers (stage-3 first) ---
        if np.random.rand() > third_tower_skip_prob:
            out = self._overlay_rgba(out, ov["red_tower"], 343, 26); rt3 = True
        if np.random.rand() > third_tower_skip_prob:
            out = self._overlay_rgba(out, ov["red_tower"], 374, 110); rm3 = True
        if np.random.rand() > third_tower_skip_prob:
            out = self._overlay_rgba(out, ov["red_tower"], 460, 132); rb3 = True

        if np.random.rand() > third_tower_skip_prob:
            out = self._overlay_rgba(out, ov["blue_tower"], 29, 350); bt3 = True
        if np.random.rand() > third_tower_skip_prob:
            out = self._overlay_rgba(out, ov["blue_tower"], 114, 368); bm3 = True
        if np.random.rand() > third_tower_skip_prob:
            out = self._overlay_rgba(out, ov["blue_tower"], 137, 451); bb3 = True

        # --- Normal towers (depends on stage-3) ---
        if rt3 and (np.random.rand() > second_tower_skip_prob):
            out = self._overlay_rgba(out, ov["red_tower"], 263, 34); rt2 = True
        if rm3 and (np.random.rand() > second_tower_skip_prob):
            out = self._overlay_rgba(out, ov["red_tower"], 325, 148); rm2 = True
        if rb3 and (np.random.rand() > second_tower_skip_prob):
            out = self._overlay_rgba(out, ov["red_tower"], 449, 212); rb2 = True

        if bt3 and (np.random.rand() > second_tower_skip_prob):
            out = self._overlay_rgba(out, ov["blue_tower"], 40, 264); bt2 = True
        if bm3 and (np.random.rand() > second_tower_skip_prob):
            out = self._overlay_rgba(out, ov["blue_tower"], 163, 329); bm2 = True
        if bb3 and (np.random.rand() > second_tower_skip_prob):
            out = self._overlay_rgba(out, ov["blue_tower"], 228, 443); bb2 = True

        # --- Outer towers (depends on both stage-2 and stage-3) ---
        if rt2 and rt3 and (np.random.rand() > first_tower_skip_prob):
            out = self._overlay_rgba(out, ov["red_tower2"], 138, 18)
        if rm2 and rm3 and (np.random.rand() > first_tower_skip_prob):
            out = self._overlay_rgba(out, ov["red_tower2"], 296, 203)
        if rb2 and rb3 and (np.random.rand() > first_tower_skip_prob):
            out = self._overlay_rgba(out, ov["red_tower2"], 468, 342)

        if bt2 and bt3 and (np.random.rand() > first_tower_skip_prob):
            out = self._overlay_rgba(out, ov["blue_tower2"], 22, 136)
        if bm2 and bm3 and (np.random.rand() > first_tower_skip_prob):
            out = self._overlay_rgba(out, ov["blue_tower2"], 191, 272)
        if bb2 and bb3 and (np.random.rand() > first_tower_skip_prob):
            out = self._overlay_rgba(out, ov["blue_tower2"], 349, 458)

        # --- Buff spots: jungle_buff / hourglass_silver / hourglass_gold / none ---
        out = self._place_random_jungle_object(out, (235, 125), jungle_rate)
        out = self._place_random_jungle_object(out, (374, 264), jungle_rate)
        out = self._place_random_jungle_object(out, (118, 227), jungle_rate)
        out = self._place_random_jungle_object(out, (256, 368), jungle_rate)

        # --- Jungle monsters: if rand > jungle_rate (80%) ---
        if np.random.rand() > jungle_rate: out = self._overlay_rgba(out, ov["jungle_monster"], 216, 81)
        if np.random.rand() > jungle_rate: out = self._overlay_rgba(out, ov["jungle_monster"], 262, 176)
        if np.random.rand() > jungle_rate: out = self._overlay_rgba(out, ov["jungle_monster"], 374, 217)
        if np.random.rand() > jungle_rate: out = self._overlay_rgba(out, ov["jungle_monster"], 430, 283)
        if np.random.rand() > jungle_rate: out = self._overlay_rgba(out, ov["jungle_monster"], 69, 215)
        if np.random.rand() > jungle_rate: out = self._overlay_rgba(out, ov["jungle_monster"], 123, 280)
        if np.random.rand() > jungle_rate: out = self._overlay_rgba(out, ov["jungle_monster"], 236, 321)
        if np.random.rand() > jungle_rate: out = self._overlay_rgba(out, ov["jungle_monster"], 282, 414)

        # --- Baron & Dragon: same threshold style ---
        out = self._place_random_epic_object(out, (156, 141), jungle_rate, key="baron")
        out = self._place_random_epic_object(out, (334, 344), jungle_rate, key="dragon_fire")

        # --- Wards / jammers (vision objects) ---
        out = self._place_random_vision_objects(out)

        return out

    def _place_random_jungle_object(self, canvas_bgr, xy, jungle_rate):
        x, y = xy
        ov = self.map_object_overlays
        r = np.random.rand()
        if r < jungle_rate:
            return self._overlay_rgba(canvas_bgr, ov["jungle_buff"], x, y)
        elif r < jungle_rate * 2:
            return self._overlay_rgba(canvas_bgr, ov["hourglass_silver"], x, y)
        elif r < jungle_rate * 2.5:
            return self._overlay_rgba(canvas_bgr, ov["hourglass_gold"], x, y)
        return canvas_bgr

    def _place_random_epic_object(self, canvas_bgr, xy, jungle_rate, key: str):
        x, y = xy
        ov = self.map_object_overlays
        r = np.random.rand()
        if r < jungle_rate:
            return self._overlay_rgba(canvas_bgr, ov[key], x, y)
        elif r < jungle_rate * 2:
            return self._overlay_rgba(canvas_bgr, ov["hourglass_silver"], x, y)
        elif r < jungle_rate * 2.5:
            return self._overlay_rgba(canvas_bgr, ov["hourglass_gold"], x, y)
        return canvas_bgr

    def _place_random_vision_objects(self, canvas_bgr):
        ov = self.map_object_overlays
        ward_keys = [k for k in ov.keys() if "ward" in k]
        jammer_keys = [k for k in ov.keys() if "jammer" in k]
        if len(ward_keys) == 0 and len(jammer_keys) == 0:
            return canvas_bgr

        out = canvas_bgr
        ward_count = int(np.random.randint(5, 20)) if len(ward_keys) > 0 else 0
        jammer_count = 1 if (len(jammer_keys) > 0 and np.random.rand() < 0.85) else 0

        for _ in range(ward_count):
            key = random.choice(ward_keys)
            overlay = ov[key]
            x, y = self._random_top_left_for_overlay(overlay)
            out = self._overlay_rgba(out, overlay, x, y)

        for _ in range(jammer_count):
            key = random.choice(jammer_keys)
            overlay = ov[key]
            x, y = self._random_top_left_for_overlay(overlay)
            out = self._overlay_rgba(out, overlay, x, y)

        return out
    
    # ---------------- Pings ----------------
    def _draw_ping_wave(self, background_bgr, center, base_radius, color_bgr):
        """Draw a foggy, non-uniform expanding ping wave."""
        h, w = background_bgr.shape[:2]
        cx, cy = center

        ring_count = int(np.random.randint(2, 4))
        step = int(np.random.randint(5, 12))
        thickness = int(np.random.randint(8, 12))
        start = int(base_radius + np.random.randint(-2, 3))
        radii = [start + i * step for i in range(ring_count)]
        max_r = max(radii) + thickness

        x0 = max(cx - max_r, 0)
        y0 = max(cy - max_r, 0)
        x1 = min(cx + max_r + 1, w)
        y1 = min(cy + max_r + 1, h)

        if x0 >= x1 or y0 >= y1:
            return background_bgr

        roi = background_bgr[y0:y1, x0:x1].astype(np.float32)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2

        noise = np.random.rand(roi.shape[0], roi.shape[1]).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (5, 5), 0)
        noise = 0.7 + 0.3 * noise
        noise = noise[..., None]

        color = np.array(color_bgr, dtype=np.float32).reshape(1, 1, 3)
        out = background_bgr.copy()
        roi_out = roi

        base_alpha = np.random.randint(130, 180)
        for i, r in enumerate(radii):
            r_outer = r
            r_inner = max(0, r - thickness)
            mask = ((dist2 >= r_inner * r_inner) & (dist2 <= r_outer * r_outer)).astype(np.float32)
            if mask.sum() == 0:
                continue
            alpha = (base_alpha * (0.75 ** i)) / 255.0
            a = (alpha * mask)[..., None]
            a = a * noise
            roi_out = (1.0 - a) * roi_out + a * color

        out[y0:y1, x0:x1] = np.clip(roi_out, 0, 255).astype(np.uint8)
        return out
    
    def _apply_random_pings(self, canvas):
        """Draw optional ping wave/overlay effects on the current canvas."""
        if len(self.ping_overlays) == 0:
            return canvas

        h, w = canvas.shape[:2]
        out = canvas.copy()

        # Scale ping visuals relative to the current canvas size (base assets assume 512).
        icon_px = int(np.round(min(h, w) / 512 * 43))
        icon_px = max(2, icon_px)
        overlay_base_scale = (min(h, w) / 512) * 0.1

        for overlay in self.ping_overlays:
            if np.random.rand() >= self.ping_attach_prob:
                continue

            oh, ow = overlay.shape[:2]
            scale = overlay_base_scale * np.random.uniform(0.9, 1.1)
            new_w = int(ow * scale)
            new_h = int(oh * scale)
            if new_w <= 0 or new_h <= 0:
                continue

            resized_overlay = cv2.resize(
                overlay, (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )

            # Random placement
            max_x = max(0, w - new_w)
            max_y = max(0, h - new_h)
            x = int(np.random.randint(0, max_x + 1)) if max_x > 0 else 0
            y = int(np.random.randint(0, max_y + 1)) if max_y > 0 else 0

            # Radius for ping wave
            radius = int(icon_px * np.random.uniform(1.2, 1.8))
            center = (x + new_w // 2, y + new_h // 2)

            color_idx = np.random.randint(len(self.ping_overlay_colors))
            color = self.ping_overlay_colors[color_idx]
            out = self._draw_ping_wave(out, center, radius, color)
            out = self._overlay_rgba(out, resized_overlay, x, y)

        return out

    def _apply_recall_tp_effects(self, canvas, center_round_positions):
        """Apply recall and teleport overlays relative to champion positions."""
        if len(self.recall_tp_overlays) == 0:
            return canvas

        out = canvas.copy()

        recall_pool = []
        tp_pool = []
        for key in self.recall_tp_overlays.keys():
            lower = key.lower()
            if "recall" in lower:
                recall_pool.append(key)
            if "tp" in lower:
                tp_pool.append(key)

        for center in center_round_positions:
            if np.random.rand() >= self.RECALL_PROB:
                continue

            if len(recall_pool) == 0:
                break

            overlay = self.recall_tp_overlays[random.choice(recall_pool)]
            overlay = self._rotate_rgba(overlay, angle_deg=random.uniform(0, 360))
            size = int(self.icon_size_pixels * np.random.uniform(1.2, 1.8))
            if size <= 0:
                continue

            resized = cv2.resize(overlay, (size, size), interpolation=cv2.INTER_AREA)
            x = int(center[0] - size / 2)
            y = int(center[1] - size / 2)
            out = self._overlay_rgba(out, resized, x, y)

        for _ in range(np.random.randint(2, 4)):
            if np.random.rand() >= self.TP_PROB:
                continue

            if len(tp_pool) == 0:
                break

            if len(center_round_positions) > 0 and np.random.rand() < 0.5:
                base_center = random.choice(center_round_positions)
                if np.random.rand() < self.TP_ON_CHAMP_PROB:
                    center = (int(base_center[0]), int(base_center[1]))
                else:
                    offset_x = np.random.randint(10, 41) * (1 if np.random.rand() < 0.5 else -1)
                    offset_y = np.random.randint(10, 41) * (1 if np.random.rand() < 0.5 else -1)
                    center = (int(base_center[0] + offset_x), int(base_center[1] + offset_y))
            else:
                rand_xy = self._random_top_left_in_map()
                center = (int(rand_xy[0] + self.icon_size_pixels * 0.5), int(rand_xy[1] + self.icon_size_pixels * 0.5))

            overlay = self.recall_tp_overlays[random.choice(tp_pool)]
            overlay = self._rotate_rgba(overlay, angle_deg=random.uniform(0, 360))
            size = int(self.icon_size_pixels * np.random.uniform(1.3, 2.1))
            if size <= 0:
                continue

            resized = cv2.resize(overlay, (size, size), interpolation=cv2.INTER_AREA)
            x = int(center[0] - size / 2)
            y = int(center[1] - size / 2)
            out = self._overlay_rgba(out, resized, x, y)

        return out

    # ---------------- Augmentations ----------------
    def _augment_icon_bgra(self, icon_bgra):
        if not self.use_icon_augment:
            return icon_bgra
        bgr = icon_bgra[:, :, :3]
        alpha = icon_bgra[:, :, 3:]

        bgr_aug = ICON_AUGMENT(image=bgr)["image"]
        return np.concatenate([bgr_aug, alpha], axis=2)

    def _augment_hsv(self, bgr, hgain=0.01, sgain=0.2, vgain=0.1, hoffset=0, soffset=0, voffset=0):
        if hgain or sgain or vgain:
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
            hue, sat, val = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
            dtype = bgr.dtype

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0] + hoffset) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1] + soffset, 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2] + voffset, 0, 255).astype(dtype)

            hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr

    def _apply_quality_reduction(
        self,
        image_bgr,
        quality_range=(71, 95),
        p_jpeg=1.0,
    ):
        out = image_bgr

        if np.random.rand() < p_jpeg:
            quality = int(np.random.uniform(*quality_range))
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            ok, enc = cv2.imencode(".jpg", out, encode_param)
            if ok:
                decoded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                if decoded is not None:
                    out = decoded

        return out


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("fork")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Generate synthetic minimap dataset")
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--output-dir", default="data/trials")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--use-hsv-augmentation", action="store_true", help="Apply HSV augmentation to champion icons")
    parser.add_argument("--use-noise", dest="use_noise", action="store_true", default=True,
                        help="Apply JPEG compression noise (enabled by default)")
    parser.add_argument("--no-noise", dest="use_noise", action="store_false",
                        help="Disable JPEG compression noise")
    parser.add_argument("--no-bg-augment", action="store_true", help="Disable background blur/downscale")
    parser.add_argument("--icon-augment", action="store_true", help="Enable icon blur/distortion/brightness")
    parser.add_argument("--allow-icon-overlap", dest="allow_icon_overlap", action="store_true", default=True,
                        help="Allow champion icon overlap (enabled by default)")
    parser.add_argument("--no-icon-overlap", dest="allow_icon_overlap", action="store_false",
                        help="Disallow champion icon overlap")
    parser.add_argument("--no-recall-tp", action="store_true", help="Disable recall and teleport overlays")
    parser.add_argument("--viewport-sim", "--viewport_sim", dest="viewport_sim", action="store_true",
                        help="Randomize viewport size to simulate camera zoom")
    parser.add_argument("--imgsz", type=int, default=256, help="Output image size (square)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of CPU cores to use")
    args = parser.parse_args()

    generator = SyntheticDataGenerator()
    generator.generate_data(
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        use_hsv_augmentation=args.use_hsv_augmentation,
        use_noise=args.use_noise,
        use_background_augment=not args.no_bg_augment,
        use_icon_augment=args.icon_augment,
        yaml=True,
        allow_icon_overlap=args.allow_icon_overlap,
        use_recall_tp=not args.no_recall_tp,
        viewport_sim=args.viewport_sim,
        output_image_size=args.imgsz,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        num_workers=args.workers
    )