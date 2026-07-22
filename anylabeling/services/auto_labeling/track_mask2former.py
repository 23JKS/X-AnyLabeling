# -*- coding: utf-8 -*-
"""Mask2Former-based track instance segmentation model for X-AnyLabeling.

Two-stage pipeline:
  1. YOLOv8n plate detection (detect individual detector boards)
  2. Mask2Former per-plate segmentation → mask_to_ribbon → centerline + auto width

Output: linestrip shapes with auto-detected centerline + track_width.
The existing band-toggle mechanism converts linestrip → band polygon on demand.
Per-track width adjustment (click polygon → slider) is preserved.
"""
import os
import gc
import traceback
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from PyQt6 import QtCore
from PyQt6.QtCore import QCoreApplication

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult


# ======================================================================
#  YOLO plate detection (replicated from utils/plates.py)
# ======================================================================

_YOLO_MODEL = None
_YOLO_MODEL_PATH = None


def _snap_to_multiple(value: int, multiple: int = 32) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _expand_axis_to_multiple(start: int, length: int, limit: int, multiple: int = 32):
    target = _snap_to_multiple(length, multiple)
    if target > limit:
        target = limit
    extra = target - length
    new_start = start - extra // 2
    new_end = new_start + target
    if new_start < 0:
        new_end -= new_start
        new_start = 0
    if new_end > limit:
        new_start -= new_end - limit
        new_end = limit
    new_start = max(0, new_start)
    return int(new_start), int(new_end - new_start)


def expand_box_to_multiple(box, image_shape, multiple=32):
    x, y, w, h = box
    h_img, w_img = image_shape[:2]
    ex, ew = _expand_axis_to_multiple(x, w, w_img, multiple)
    ey, eh = _expand_axis_to_multiple(y, h, h_img, multiple)
    return ex, ey, ew, eh


def _split_by_vertical_gaps(box, binary, gray):
    """Split one merged plate box by tall dark vertical gaps inside it."""
    x, y, w, h = box
    if w < 80 or h < 80:
        return [box]

    crop = binary[y:y + h, x:x + w] > 0
    col_fg_ratio = crop.mean(axis=0)
    gray_crop = gray[y:y + h, x:x + w]
    dark_pixel_ratio = (gray_crop < 90).mean(axis=0)

    margin = max(10, int(w * 0.05))
    dark = (col_fg_ratio < 0.85) & (dark_pixel_ratio > 0.15)
    dark[:margin] = False
    dark[w - margin:] = False

    min_gap_width = 6
    min_part_width = max(300, int(w * 0.08))
    split_points = []
    start = None
    for i, is_dark in enumerate(dark):
        if is_dark and start is None:
            start = i
        elif not is_dark and start is not None:
            if i - start >= min_gap_width:
                split_points.append((start + i) // 2)
            start = None
    if start is not None and w - start >= min_gap_width:
        split_points.append((start + w) // 2)

    if len(split_points) >= 2:
        typical_width = int(np.median(np.diff(split_points)))
    elif len(split_points) == 1:
        typical_width = split_points[0]
    else:
        typical_width = 0

    if typical_width >= min_part_width:
        expected = split_points[0] + typical_width if split_points else typical_width
        search_radius = max(20, int(typical_width * 0.12))
        while expected < w - min_part_width:
            if all(abs(expected - p) > search_radius for p in split_points):
                lo = max(margin, expected - search_radius)
                hi = min(w - margin, expected + search_radius)
                if hi > lo:
                    score = col_fg_ratio[lo:hi] - dark_pixel_ratio[lo:hi]
                    split_points.append(lo + int(np.argmin(score)))
            expected += typical_width

    split_points = sorted(set(split_points))

    if len(split_points) >= 2 and typical_width >= min_part_width:
        typical_width2 = int(np.median(np.diff([0] + split_points + [w])))
        if typical_width2 >= min_part_width:
            max_part_width = int(typical_width2 * 1.55)
            extra_points = []
            edges = [0] + split_points + [w]
            for left_edge, right_edge in zip(edges[:-1], edges[1:]):
                part_width = right_edge - left_edge
                while part_width > max_part_width:
                    expected2 = left_edge + typical_width2
                    sr2 = max(20, int(typical_width2 * 0.18))
                    lo2 = max(left_edge + min_part_width, expected2 - sr2)
                    hi2 = min(right_edge - min_part_width, expected2 + sr2)
                    if hi2 <= lo2:
                        break
                    score = col_fg_ratio[lo2:hi2] - dark_pixel_ratio[lo2:hi2]
                    split_x = lo2 + int(np.argmin(score))
                    extra_points.append(split_x)
                    left_edge = split_x
                    part_width = right_edge - left_edge
            split_points = sorted(set(split_points + extra_points))

    parts = []
    left = 0
    for split_x in split_points:
        if split_x - left >= min_part_width:
            parts.append((x + left, y, split_x - left, h))
            left = split_x
    if w - left >= min_part_width:
        parts.append((x + left, y, w - left, h))
    return parts if len(parts) > 1 else [box]


def find_plates_cv(image, min_area_ratio=0.01, close_kernel=25):
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    raw_binary = binary.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    h_img, w_img = gray.shape[:2]
    min_area = h_img * w_img * min_area_ratio

    components = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            components.append((int(x), int(y), int(w), int(h)))

    if not components:
        return [(0, 0, w_img, h_img)]

    plates = []
    for box in components:
        plates.extend(_split_by_vertical_gaps(box, raw_binary, gray))
    return sorted(plates, key=lambda p: p[0])


def _load_yolo_model(model_path: str):
    global _YOLO_MODEL, _YOLO_MODEL_PATH
    model_path = str(model_path)
    if _YOLO_MODEL is not None and _YOLO_MODEL_PATH == model_path:
        return _YOLO_MODEL
    from ultralytics import YOLO
    _YOLO_MODEL = YOLO(model_path)
    _YOLO_MODEL_PATH = model_path
    return _YOLO_MODEL


def find_plates_yolo(image, model_path, conf=0.25, iou=0.5, imgsz=640):
    weights = Path(model_path)
    if not weights.exists():
        raise FileNotFoundError(f"YOLO plate weights not found: {weights}")

    model = _load_yolo_model(str(weights))
    result = model.predict(source=image, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]

    h_img, w_img = image.shape[:2]
    boxes = []
    if result.boxes is None:
        return boxes

    xyxy_raw = result.boxes.xyxy
    cpu_fn = getattr(xyxy_raw, "cpu", None)
    xyxy_data = cpu_fn().numpy() if cpu_fn else np.asarray(xyxy_raw)

    for xyxy in xyxy_data:
        x1, y1, x2, y2 = xyxy.tolist()
        x1 = int(max(0, min(w_img - 1, round(x1))))
        y1 = int(max(0, min(h_img - 1, round(y1))))
        x2 = int(max(0, min(w_img, round(x2))))
        y2 = int(max(0, min(h_img, round(y2))))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((x1, y1, x2 - x1, y2 - y1))
    return sorted(boxes, key=lambda p: p[0])


def find_plates(image, model_path, conf=0.25, iou=0.5, min_area_ratio=0.005):
    """Detect plate bounding boxes. YOLO first, CV fallback if YOLO fails."""
    try:
        boxes = find_plates_yolo(image, model_path=model_path, conf=conf, iou=iou)
        if boxes:
            return boxes
    except Exception:
        pass
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    return find_plates_cv(gray, min_area_ratio=min_area_ratio)


# ======================================================================
#  mask_to_ribbon (inlined from infer1.py)
# ======================================================================

def _skeletonize(mask):
    mask = mask.astype(np.uint8)
    if mask.sum() < 5:
        return np.zeros_like(mask, dtype=np.uint8)
    try:
        import skimage.morphology as skmorph
        skel = skmorph.skeletonize(mask.astype(bool)).astype(np.uint8)
        if skel.sum() >= 3:
            return skel
    except ImportError:
        pass
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    local_max = cv2.dilate(dist, np.ones((3, 3))) == dist
    skel = (local_max & (dist > dist.max() * 0.3)).astype(np.uint8) & mask
    if skel.sum() < 3:
        skel = (dist >= dist.max() * 0.7).astype(np.uint8) & mask
    return skel


def _neighbors_8(y, x, h, w, skel):
    count = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] > 0:
                count += 1
    return count


def _dijkstra_path(skel, start, end):
    import heapq
    h, w = skel.shape
    pq = [(0.0, start, [start])]
    best = {start: 0.0}
    while pq:
        cost, (y, x), path = heapq.heappop(pq)
        if (y, x) == end:
            return path
        if cost > best.get((y, x), float('inf')):
            continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                    step_cost = 1.414 if dy and dx else 1.0
                    nc = cost + step_cost
                    if nc < best.get((ny, nx), float('inf')):
                        best[(ny, nx)] = nc
                        heapq.heappush(pq, (nc, (ny, nx), path + [(ny, nx)]))
    return None


def _smooth_path(pts, sigma=1.5):
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 5:
        return pts
    ks = max(3, min(int(sigma * 6) | 1, len(pts) - 2))
    x = np.arange(ks) - ks // 2
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    kernel /= kernel.sum()
    y_sm = np.convolve(np.pad(pts[:, 0], ks // 2, 'edge'), kernel, 'same')[ks // 2:len(pts) + ks // 2]
    x_sm = np.convolve(np.pad(pts[:, 1], ks // 2, 'edge'), kernel, 'same')[ks // 2:len(pts) + ks // 2]
    return np.stack([y_sm, x_sm], axis=1)


def mask_to_ribbon(mask, keypoint_interval=30.0):
    """mask → centerline + auto half_width → list of (x, y) centerline points + half_width."""
    h, w = mask.shape
    mask = mask.astype(np.uint8)
    if mask.sum() < 10:
        return [], 0.0

    skel = _skeletonize(mask)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    vals = dist[skel > 0]
    half_width = float(np.median(vals)) if len(vals) > 0 else 3.0
    half_width = max(half_width, 1.0)

    pts = np.column_stack(np.where(skel > 0))
    if len(pts) < 2:
        return [], half_width

    endpoints = [(int(y), int(x)) for y, x in pts if _neighbors_8(y, x, h, w, skel) == 1]
    if len(endpoints) < 2:
        endpoints = [tuple(pts[0]), tuple(pts[-1])]

    path = _dijkstra_path(skel, endpoints[0], endpoints[-1])
    if path is None or len(path) < 3:
        order = np.argsort(pts[:, 1]) if h > w else np.argsort(pts[:, 0])
        path = [(int(pts[i, 0]), int(pts[i, 1])) for i in order]

    path_arr = np.array(path, dtype=np.float64)
    dists = np.zeros(len(path_arr))
    for j in range(1, len(path_arr)):
        dists[j] = dists[j - 1] + np.linalg.norm(path_arr[j] - path_arr[j - 1])
    total_len = dists[-1]
    n_samples = max(2, int(total_len / keypoint_interval) + 1)
    targets = np.linspace(0, total_len, n_samples)
    indices = np.searchsorted(dists, targets)
    indices = np.clip(indices, 0, len(path_arr) - 1)
    indices = np.unique(indices)
    if indices[0] != 0:
        indices = np.concatenate([[0], indices])
    if indices[-1] != len(path_arr) - 1:
        indices = np.concatenate([indices, [len(path_arr) - 1]])
    keypoints_yx = path_arr[indices]

    if len(keypoints_yx) >= 5:
        keypoints_yx = _smooth_path(keypoints_yx, sigma=1.5)
    keypoints_yx[0] = path[0]
    keypoints_yx[-1] = path[-1]

    centerline = [(float(x), float(y)) for y, x in keypoints_yx]
    return centerline, half_width


# ======================================================================
#  Mask2Former helpers (inlined from infer1.py)
# ======================================================================

def _adapt_first_conv_to_grayscale(model):
    import torch.nn as nn
    emb = model.model.pixel_level_module.encoder.embeddings.patch_embeddings
    old_conv = emb.projection
    if old_conv.in_channels == 1:
        return
    new_conv = nn.Conv2d(1, old_conv.out_channels,
                         kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride,
                         padding=old_conv.padding,
                         bias=old_conv.bias is not None)
    new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
    if old_conv.bias is not None:
        new_conv.bias.data = old_conv.bias.data
    emb.projection = new_conv
    emb.num_channels = 1


def _imread_unicode(path: str) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def _segment_plate(crop, model, device, conf=0.5):
    """Run Mask2Former on a plate crop, return instances in crop-local coords."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    if crop.size == 0:
        return []

    H_crop, W_crop = crop.shape[:2]
    transform = A.Compose([
        A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
        ToTensorV2(),
    ])
    inp = transform(image=crop)["image"].float().unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=inp)

    from transformers import Mask2FormerImageProcessor
    processor = Mask2FormerImageProcessor.from_pretrained(
        "facebook/mask2former-swin-tiny-coco-instance",
        size={"height": H_crop, "width": W_crop},
        ignore_index=0,
        local_files_only=True,
    )
    results = processor.post_process_instance_segmentation(
        outputs, threshold=conf,
        target_sizes=[(H_crop, W_crop)],
    )[0]

    instances = []
    if "segmentation" in results:
        seg = results["segmentation"].cpu().numpy()
        seg_info = {s["id"]: s for s in results["segments_info"]}
        for seg_id in np.unique(seg):
            if seg_id == 0:
                continue
            info = seg_info.get(seg_id, {})
            label_id = info.get("label_id", info.get("category_id", 0))
            mask = (seg == seg_id).astype(np.uint8)
            mask = cv2.GaussianBlur(mask, (5, 5), 1.0)
            mask = (mask > 0.5).astype(np.uint8)
            instances.append({
                "class_id": label_id,
                "class_name": "track" if label_id == 0 else "source",
                "mask": mask,
                "confidence": info.get("score", 1.0),
            })
    elif "masks" in results:
        for i in range(len(results["masks"])):
            mask = results["masks"][i].cpu().numpy().astype(np.uint8)
            label_id = results["labels"][i].item()
            conf_val = results["scores"][i].item() if "scores" in results else 1.0
            instances.append({
                "class_id": label_id,
                "class_name": "track" if label_id == 0 else "source",
                "mask": mask,
                "confidence": conf_val,
            })
    return instances


# ======================================================================
#  X-AnyLabeling model class
# ======================================================================

class TrackMask2Former(Model):

    class Meta:
        required_config_names = ["type", "name", "display_name", "model_path", "plate_model_path"]
        widgets = [
            "button_run",
            "toggle_preserve_existing_annotations",
            "edit_min_track_length",
            "input_min_track_length",
            "edit_keypoint_interval",
            "input_keypoint_interval",
            "edit_track_width",
            "input_track_width",
            "button_toggle_band",
            "button_energy_spectrum",
        ]
        output_modes = {
            "polygon": QCoreApplication.translate("Model", "Polygon"),
        }
        default_output_mode = "linestrip"

    def __init__(self, model_config, on_message) -> None:
        super().__init__(model_config, on_message)

        self.model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not self.model_abs_path or not os.path.isdir(self.model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model", "Mask2Former checkpoint dir not found: {path}"
                ).format(path=self.model_abs_path)
            )

        self.plate_model_path = self.config.get("plate_model_path", "")
        if not self.plate_model_path or not os.path.isfile(self.plate_model_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model", "YOLO plate model not found: {path}"
                ).format(path=self.plate_model_path)
            )

        self.conf_threshold = self.config.get("conf_threshold", 0.3)
        self._min_track_length = self.config.get("min_track_length", 30)
        self._keypoint_interval = self.config.get("keypoint_interval", 30)
        self._default_track_width = self.config.get("default_track_width", 15)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()
        self.replace = True

    def _load_model(self):
        self.on_message("Loading Mask2Former model...")
        from transformers import (
            Mask2FormerForUniversalSegmentation,
            Mask2FormerConfig,
        )
        config = Mask2FormerConfig.from_pretrained(
            str(self.model_abs_path),
            local_files_only=True,
        )
        self.model = Mask2FormerForUniversalSegmentation(config)
        _adapt_first_conv_to_grayscale(self.model)

        state_path = Path(self.model_abs_path) / "model.safetensors"
        if not state_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {state_path}"
            )
        try:
            from safetensors.torch import load_file as safe_load
            state_dict = safe_load(str(state_path))
        except ImportError:
            state_dict = torch.load(str(state_path), map_location="cpu",
                                    weights_only=True)
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.on_message("Mask2Former model loaded.")

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        self.replace = not state

    def set_auto_labeling_min_track_length(self, value):
        self._min_track_length = int(value)

    def set_auto_labeling_keypoint_interval(self, value):
        self._keypoint_interval = int(value)

    def set_auto_labeling_track_width(self, value):
        """Set default track width for manually added centerlines without auto width."""
        self._default_track_width = int(value)

    def predict_shapes(self, image, image_path=None):
        if self.model is None:
            return AutoLabelingResult([], replace=self.replace)

        try:
            # Match infer1.py: read grayscale directly from file when available
            if image_path and os.path.isfile(image_path):
                data = np.fromfile(image_path, dtype=np.uint8)
                image_gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            else:
                image_rgb = qt_img_to_rgb_cv_img(image)
                image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            H, W = image_gray.shape[:2]
            # YOLO plate detection expects RGB
            image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
        except Exception as e:
            logger.warning(f"Image preprocessing error: {e}\n{traceback.format_exc()}")
            return AutoLabelingResult([], replace=self.replace)

        # Stage 1: detect plates
        plates = find_plates(image_rgb, self.plate_model_path)
        self.on_message(f"Stage 1: {len(plates)} plate(s) detected")

        # Stage 2: segment each plate
        all_tracks = []
        all_sources = []

        for plate_idx, (px, py, pw, ph) in enumerate(plates):
            ex, ey, ew, eh = expand_box_to_multiple((px, py, pw, ph), (H, W), multiple=32)
            crop = image_gray[ey:ey + eh, ex:ex + ew]
            try:
                instances = _segment_plate(crop, self.model, self.device, conf=self.conf_threshold)
            except Exception as e:
                logger.warning(f"Segmentation error for plate ({px},{py}): {e}")
                continue

            plate_area = ew * eh
            plate_bbox = [px, py, pw, ph]
            for inst in instances:
                mask_area = inst["mask"].sum()
                if mask_area > plate_area * 0.5:
                    continue
                global_mask = np.zeros((H, W), dtype=np.uint8)
                global_mask[ey:ey + eh, ex:ex + ew] = inst["mask"]

                if inst["class_name"] == "source":
                    y_coords = np.where(global_mask > 0)[0]
                    center_y = y_coords.mean() if len(y_coords) > 0 else 0
                    all_sources.append({
                        "mask": global_mask,
                        "confidence": inst["confidence"],
                        "center_y": center_y,
                        "plate_id": plate_idx,
                        "plate_bbox": plate_bbox,
                    })
                else:
                    all_tracks.append({
                        "mask": global_mask,
                        "confidence": inst["confidence"],
                        "plate_id": plate_idx,
                        "plate_bbox": plate_bbox,
                    })

        # Outlier filtering for sources
        if all_sources:
            y_coords = np.array([s["center_y"] for s in all_sources])
            median_y = np.median(y_coords)
            mad = np.median(np.abs(y_coords - median_y))
            threshold = 3 * mad if mad > 0 else 50
            all_sources = [s for s in all_sources if abs(s["center_y"] - median_y) <= threshold]

        self.on_message(
            f"Stage 2: {len(all_tracks)} track(s), {len(all_sources)} source(s)"
        )

        shapes = []
        track_id = 0
        source_id = 0

        # Source shapes → polygon (raw contour)
        source_shapes = []
        for inst in all_sources:
            contours, _ = cv2.findContours(inst["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            shape = Shape(label="source", score=float(inst["confidence"]), shape_type="polygon")
            for pt in cnt:
                shape.add_point(QtCore.QPointF(float(pt[0][0]), float(pt[0][1])))
            shape.closed = True
            shape.other_data["plate_id"] = inst["plate_id"]
            shape.other_data["plate_bbox"] = inst["plate_bbox"]
            source_shapes.append(shape)
        # 按中心线最底部（y 最大）的 x 坐标从左往右排序后编号
        def _bottom_x(shape):
            cl = shape.other_data.get("centerline") if shape.other_data else None
            if cl:
                bottom = max(cl, key=lambda p: p[1])  # (x, y)
                return bottom[0]
            pts = shape.points
            if not pts:
                return 0.0
            max_y = max(p.y() for p in pts)
            return min(p.x() for p in pts if p.y() == max_y)
        source_shapes.sort(key=_bottom_x)
        for s in source_shapes:
            source_id += 1
            s.label = f"source_{source_id}"
            shapes.append(s)

        # Track shapes → linestrip with auto centerline + track_width
        track_shapes = []
        for inst in all_tracks:
            centerline, half_width = mask_to_ribbon(inst["mask"], keypoint_interval=self._keypoint_interval)
            if not centerline or len(centerline) < 2:
                continue

            track_len = 0.0
            for i in range(1, len(centerline)):
                dx = centerline[i][0] - centerline[i - 1][0]
                dy = centerline[i][1] - centerline[i - 1][1]
                track_len += (dx * dx + dy * dy) ** 0.5
            if track_len < self._min_track_length:
                continue

            track_width = int(round(half_width * 2))
            shape = Shape(
                label="track",
                score=float(inst["confidence"]),
                shape_type="linestrip",
                flags={},
            )
            shape.other_data["centerline"] = centerline
            shape.other_data["track_width"] = track_width
            shape.other_data["plate_id"] = inst["plate_id"]
            shape.other_data["plate_bbox"] = inst["plate_bbox"]
            for pt in centerline:
                shape.add_point(QtCore.QPointF(pt[0], pt[1]))
            track_shapes.append(shape)
        # 按带状多边形最底部（y 最大）的 x 坐标从左往右排序后编号
        track_shapes.sort(key=_bottom_x)
        for s in track_shapes:
            track_id += 1
            s.label = f"track_{track_id}"
            shapes.append(s)

        self.on_message(f"Done: {len(shapes)} shapes ({track_id} tracks, {source_id} sources)")
        return AutoLabelingResult(shapes, replace=self.replace)

    def unload(self):
        global _YOLO_MODEL, _YOLO_MODEL_PATH
        if self.model is not None:
            del self.model
            self.model = None
        _YOLO_MODEL = None
        _YOLO_MODEL_PATH = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
