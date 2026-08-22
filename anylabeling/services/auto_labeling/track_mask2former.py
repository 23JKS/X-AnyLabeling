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
import heapq
import traceback
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from PIL import Image
from PyQt6 import QtCore
from PyQt6.QtCore import QCoreApplication

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.label_file import LabelFile
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


class FineTuneCancelledError(Exception):
    """Raised when fine-tuning is cancelled by the user."""


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
    """骨架化：scikit-image > cv2.ximgproc > 距离变换脊线 > 距离变换阈值"""
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
    try:
        skel = cv2.ximgproc.thinning(mask)
        if skel is not None and skel.sum() >= 3:
            return skel
    except (AttributeError, cv2.error):
        pass
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    local_max = cv2.dilate(dist, np.ones((3, 3))) == dist
    skel = (local_max & (dist > dist.max() * 0.1)).astype(np.uint8) & mask
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


def centerline_to_ribbon(centerline, track_width):
    """从中心线 + 宽度直接计算 ribbon 多边形（法向量偏移法）。

    与 mask_to_ribbon 中的 ribbon 计算逻辑完全一致，但跳过了骨架化步骤。
    用于手动绘制的 track（已有精确中心线）。

    Args:
        centerline: [(x, y), ...] 中心线折点列表，图像坐标
        track_width: int 完整宽度（像素）

    Returns:
        ribbon: [(x, y), ...] ribbon 多边形顶点列表
    """
    half_width = track_width / 2.0
    n = len(centerline)
    if n < 2:
        return []

    # (y, x) 格式用于法向量计算
    pts = np.array([(y, x) for x, y in centerline], dtype=np.float64)
    left, right = [], []
    for i in range(n):
        y, x = pts[i]
        if i == 0:
            dy, dx = pts[1][0] - y, pts[1][1] - x
        elif i == n - 1:
            dy, dx = y - pts[-2][0], x - pts[-2][1]
        else:
            dy = pts[i + 1][0] - pts[i - 1][0]
            dx = pts[i + 1][1] - pts[i - 1][1]
        norm = max(np.sqrt(dx * dx + dy * dy), 1e-6)
        nx, ny = -dy / norm, dx / norm

        if i == 0 or i == n - 1:
            ny = 0.0
            nx = 1.0 if nx > 0 else -1.0

        left.append((float(x + nx * half_width), float(y + ny * half_width)))
        right.append((float(x - nx * half_width), float(y - ny * half_width)))

    return left + list(reversed(right))


def centerline_to_ribbon_mask(centerline, track_width, H, W):
    """从中心线 + 宽度直接生成 ribbon 填充 mask（避免 polylines→骨架化的绕路）。

    Args:
        centerline: [(x, y), ...] 中心线折点列表
        track_width: int 完整宽度（像素）
        H, W: int 图像尺寸

    Returns:
        mask: (H, W) uint8 二值 mask
    """
    ribbon = centerline_to_ribbon(centerline, track_width)
    mask = np.zeros((H, W), dtype=np.uint8)
    if ribbon and len(ribbon) >= 3:
        cv2.fillPoly(mask, [np.array(ribbon, dtype=np.int32)], 1)
    return mask


def _offset_centerline(centerline, offset):
    """Offset each point of a centerline along its normal direction.

    Args:
        centerline: [(x, y), ...]
        offset: float, positive = right side, negative = left side

    Returns:
        [(x, y), ...]
    """
    n = len(centerline)
    result = []
    for i in range(n):
        cx, cy = centerline[i]
        if i == 0:
            dx = centerline[1][0] - cx
            dy = centerline[1][1] - cy
        elif i == n - 1:
            dx = cx - centerline[-2][0]
            dy = cy - centerline[-2][1]
        else:
            dx = centerline[i + 1][0] - centerline[i - 1][0]
            dy = centerline[i + 1][1] - centerline[i - 1][1]
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-6:
            result.append((cx, cy))
            continue
        nx, ny = -dy / length, dx / length
        # Force horizontal at endpoints
        if i == 0 or i == n - 1:
            ny = 0.0
            nx = 1.0 if nx > 0 else -1.0
        result.append((cx + nx * offset, cy + ny * offset))
    return result


def _extend_centerline_y(centerline, y_min, y_max):
    """Extend the first/last point along the tangent so the centerline's
    Y range covers [y_min, y_max].

    Args:
        centerline: [(x, y), ...]
        y_min, y_max: target Y range

    Returns:
        [(x, y), ...] with extended endpoints
    """
    if len(centerline) < 2:
        return centerline
    cl = list(centerline)

    # Extend start point backwards
    dx = cl[1][0] - cl[0][0]
    dy = cl[1][1] - cl[0][1]
    length = (dx * dx + dy * dy) ** 0.5
    if length > 1e-6 and abs(dy) > 1e-6:
        tx, ty = dx / length, dy / length
        t = (cl[0][1] - y_min) / ty
        if abs(t) > 1e-6:
            cl[0] = (cl[0][0] - t * tx, cl[0][1] - t * ty)

    # Extend end point forward
    dx = cl[-1][0] - cl[-2][0]
    dy = cl[-1][1] - cl[-2][1]
    length = (dx * dx + dy * dy) ** 0.5
    if length > 1e-6 and abs(dy) > 1e-6:
        tx, ty = dx / length, dy / length
        t = (y_max - cl[-1][1]) / ty
        if abs(t) > 1e-6:
            cl[-1] = (cl[-1][0] + t * tx, cl[-1][1] + t * ty)

    return cl


def mask_to_ribbon(mask, keypoint_interval=30.0, other_masks=None,
                   force_bg_side: float = 0.0):
    """实例分割 mask → 中心线 + 半宽 → ribbon 多边形 + 背景条带

    Args:
        mask: (H, W) uint8 binary mask of this track
        keypoint_interval: 中心线上关键点的像素间距
        other_masks: (H, W) uint8 combined mask of *other* instances in the same plate.
                     用来裁切背景条带，保证bg不碰到其他轨迹。
        force_bg_side: +1 强制右侧, -1 强制左侧, 0 自动选最佳侧
    Returns:
        ribbon: [(x, y), ...] 带状多边形顶点列表, 或空列表
        centerline: [(x, y), ...] 中心线关键点折线, 或空列表
        half_width: float 半宽（像素）
        bg_polygon: [(x, y), ...] 背景条带多边形顶点列表, 或空列表
        bg_gap: float 背景条带与轨迹之间的安全间距（像素）
        bg_side: float +1 或 -1，背景条带在哪一侧；无背景时为 0
    """
    h, w = mask.shape
    mask = mask.astype(np.uint8)
    if mask.sum() < 10:
        return [], [], 0.0, [], max(5.0, 3.0), 0.0

    # 1. 骨架化
    skel = _skeletonize(mask)

    # 2. 计算半宽（距离变换 + 骨架上的中位数）
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    vals = dist[skel > 0]
    half_width = float(np.median(vals)) if len(vals) > 0 else 3.0

    # 3. 骨架 → 关键点路径
    pts = np.column_stack(np.where(skel > 0))
    if len(pts) < 2:
        return [], [], half_width, [], max(5.0, half_width), 0.0

    # 找端点 (8邻域只有1个邻居的点)
    endpoints = [(int(y), int(x)) for y, x in pts
                 if _neighbors_8(y, x, h, w, skel) == 1]
    if len(endpoints) < 2:
        endpoints = [tuple(pts[0]), tuple(pts[-1])]

    # Dijkstra 最短路径
    path = _dijkstra_path(skel, endpoints[0], endpoints[-1])
    if path is None or len(path) < 3:
        order = np.argsort(pts[:, 1]) if h > w else np.argsort(pts[:, 0])
        path = [(int(pts[i, 0]), int(pts[i, 1])) for i in order]

    # 移除路径中的 Y 回退段，防止中心线回环（轨迹始终竖向，Y 单调递增）
    mono = [path[0]]
    for p in path[1:]:
        if p[0] >= mono[-1][0]:
            mono.append(p)
    if len(mono) >= 3:
        path = mono
    elif path[0][0] > path[-1][0]:
        path = list(reversed(path))

    # 4. 按像素间距采样关键点 → 中心线折线
    path_arr = np.array(path, dtype=np.float64)
    dists = np.zeros(len(path_arr))
    for j in range(1, len(path_arr)):
        dists[j] = dists[j-1] + np.linalg.norm(path_arr[j] - path_arr[j-1])
    total_len = dists[-1]
    n_samples = max(2, int(total_len / keypoint_interval) + 1)
    targets = np.linspace(0, total_len, n_samples)
    indices = np.searchsorted(dists, targets)
    indices = np.clip(indices, 0, len(path_arr) - 1)
    indices = np.unique(indices)
    # 确保包含首尾端点
    if indices[0] != 0:
        indices = np.concatenate([[0], indices])
    if indices[-1] != len(path_arr) - 1:
        indices = np.concatenate([indices, [len(path_arr) - 1]])
    keypoints_yx = path_arr[indices]
    if len(keypoints_yx) < 2:
        return [], [], half_width, [], max(5.0, half_width), 0.0

    # 平滑
    if len(keypoints_yx) >= 5:
        keypoints_yx = _smooth_path(keypoints_yx, sigma=1.5)
    keypoints_yx[0] = path[0]
    keypoints_yx[-1] = path[-1]

    # 确保中心线从上到下 (Y 递增)；若反了则整体反转
    if keypoints_yx[-1][0] < keypoints_yx[0][0]:
        keypoints_yx = keypoints_yx[::-1]

    # 中心线: (y, x) → [(x, y), ...]
    centerline = [(float(x), float(y)) for y, x in keypoints_yx]

    # 5. 中心线 + 法向量方向 ± 半宽 → ribbon
    n = len(keypoints_yx)
    left, right = [], []
    for i in range(n):
        y, x = keypoints_yx[i]
        if i == 0:
            dy, dx = keypoints_yx[1][0] - y, keypoints_yx[1][1] - x
        elif i == n - 1:
            dy, dx = y - keypoints_yx[-2][0], x - keypoints_yx[-2][1]
        else:
            dy = keypoints_yx[i + 1][0] - keypoints_yx[i - 1][0]
            dx = keypoints_yx[i + 1][1] - keypoints_yx[i - 1][1]
        norm = max(np.sqrt(dx * dx + dy * dy), 1e-6)
        nx, ny = -dy / norm, dx / norm

        # 端点强制水平 ny=0
        if i == 0 or i == n - 1:
            ny = 0.0
            nx = 1.0 if nx > 0 else -1.0

        left.append((float(x + nx * half_width), float(y + ny * half_width)))
        right.append((float(x - nx * half_width), float(y - ny * half_width)))

    ribbon = left + list(reversed(right))

    # 6. Background strip — same width as track, with a gap so no track pixels
    #    are included.  The background polygon is generated by offsetting the
    #    track centerline along the normal direction (exactly the same way the
    #    track ribbon is built), so its width is identical to the track width.
    #    A filled mask is still used to decide which side is better and to
    #    detect overlap with other tracks, but the final polygon comes from the
    #    offset points (not from findContours), guaranteeing width consistency.
    bg_polygon = []
    bg_hw = max(1.0, half_width)                   # 与轨迹等宽（不再 -1）
    bg_gap = max(30.0, half_width)                   # safety gap (px)
    bg_offset = half_width + bg_gap + bg_hw         # centerline → bg center distance

    # Mask to avoid: other instances in the same plate (not this track itself)
    avoid = other_masks if other_masks is not None else np.zeros_like(mask)

    best_bg = None
    best_bg_side = 0.0
    best_bg_length = 0.0

    # Unit vector of the centerline's principal direction (for length projection)
    cl_pts = np.array(keypoints_yx)  # (n, 2) [y, x]
    cl_vec = cl_pts[-1] - cl_pts[0]
    cl_len = float(np.linalg.norm(cl_vec))
    cl_dir = cl_vec / cl_len if cl_len >= 1 else np.array([0.0, 1.0])

    sides_to_try = [force_bg_side] if abs(force_bg_side) > 0.001 else [+1.0, -1.0]

    for sign in sides_to_try:
        bg_left, bg_right = [], []
        for i in range(n):
            y, x = keypoints_yx[i]
            if i == 0:
                dy, dx = keypoints_yx[1][0] - y, keypoints_yx[1][1] - x
            elif i == n - 1:
                dy, dx = y - keypoints_yx[-2][0], x - keypoints_yx[-2][1]
            else:
                dy = keypoints_yx[i + 1][0] - keypoints_yx[i - 1][0]
                dx = keypoints_yx[i + 1][1] - keypoints_yx[i - 1][1]
            norm = max(np.sqrt(dx * dx + dy * dy), 1e-6)
            nx, ny = -dy / norm, dx / norm

            # 端点强制水平 ny=0（与 track ribbon 一致）
            if i == 0 or i == n - 1:
                ny = 0.0
                nx = 1.0 if nx > 0 else -1.0

            bg_cx = x + sign * nx * bg_offset
            bg_cy = y + sign * ny * bg_offset
            bg_left.append((float(bg_cx + nx * bg_hw), float(bg_cy + ny * bg_hw)))
            bg_right.append((float(bg_cx - nx * bg_hw), float(bg_cy - ny * bg_hw)))

        bg_poly = np.array(bg_left + list(reversed(bg_right)), dtype=np.float64)
        if len(bg_poly) < 6:
            continue

        bg_mask_side = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(bg_mask_side, [bg_poly.astype(np.int32)], 1)

        # 删除与*其他*轨迹像素重叠的行 → 可能截断成多段
        overlap = (avoid > 0) & (bg_mask_side > 0)
        bad_rows = np.unique(np.where(overlap)[0])
        for r in bad_rows:
            bg_mask_side[r, :] = 0

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            bg_mask_side, connectivity=8)
        if num_labels > 1:
            for lbl in range(1, num_labels):
                comp_mask = (labels == lbl).astype(np.uint8)
                ys, xs = np.where(comp_mask)
                if len(ys) < 10:
                    continue
                # 沿中心线方向的投影长度
                dots = ys * cl_dir[0] + xs * cl_dir[1]
                proj_len = float(dots.max() - dots.min())
                if proj_len > best_bg_length:
                    best_bg_length = proj_len
                    best_bg_side = sign
                    # 直接用偏移点作为背景多边形（与轨迹 ribbon 同方式生成，
                    # 宽度严格等于 half_width，不依赖 findContours 轮廓）
                    best_bg = [[float(p[0]), float(p[1])] for p in bg_poly]

    if best_bg is not None:
        # 让背景多边形的 Y 范围严格覆盖对应轨迹的 Y 范围。
        # 轨迹 ribbon 端点 ny=0，故其 Y 范围 = 中心线端点 Y 范围；
        # 而轨迹 mask 的 Y 范围（track_y_min~track_y_max）因 mask 有宽度会略大。
        # 这里把背景多边形整体平移并拉伸 Y，使其 Y 范围与轨迹 mask 完全一致，
        # X 方向保持不变（宽度由法向偏移保证）。
        track_ys = np.where(mask > 0)[0]
        if len(track_ys) > 0:
            track_y_min = float(track_ys.min())
            track_y_max = float(track_ys.max())
            bg_arr = np.array(best_bg, dtype=np.float64)  # (N, 2) [x, y]
            bg_y_min = bg_arr[:, 1].min()
            bg_y_max = bg_arr[:, 1].max()
            if bg_y_max - bg_y_min > 1.0:
                # 平移 + 拉伸 Y 到轨迹的 Y 范围（仅 Y，不影响宽度）
                ratio = (track_y_max - track_y_min) / (bg_y_max - bg_y_min)
                bg_arr[:, 1] = track_y_min + (bg_arr[:, 1] - bg_y_min) * ratio
            else:
                bg_arr[:, 1] = (track_y_min + track_y_max) / 2.0
            best_bg = [[float(p[0]), float(p[1])] for p in bg_arr]
        bg_polygon = best_bg

    return ribbon, centerline, half_width, bg_polygon, bg_gap, best_bg_side


# ======================================================================
#  Mask2Former helpers (inlined from infer1.py)
# ======================================================================

def _adapt_first_conv_to_grayscale(model):
    """Replace Swin patch embedding's first conv: 3→1 input channel.

    Works with both SwinModel (embeddings.patch_embeddings) and
    SwinBackbone (swin.embeddings.patch_embeddings).
    """
    import torch.nn as nn
    encoder = model.model.pixel_level_module.encoder
    # SwinBackbone wraps a SwinModel in .swin, SwinModel has it directly
    root = getattr(encoder, "swin", encoder)
    emb = root.embeddings.patch_embeddings
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
    if hasattr(emb, "num_channels"):
        emb.num_channels = 1


def _imread_unicode(path: str) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def _segment_plate(crop, model, device, model_abs_path, conf=0.5):
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
    # Load the processor from the local checkpoint directory (which ships with
    # preprocessor_config.json) instead of the online HuggingFace repo. The
    # online repo requires a local HF cache that does not exist on a fresh
    # machine; with local_files_only=True that raises and every plate is
    # skipped, producing an empty annotation file.
    processor = Mask2FormerImageProcessor.from_pretrained(
        str(model_abs_path),
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


def _load_tiff_log_compressed(image_path):
    """Load an image for training, applying log-compression to TIFF files.

    This produces the *same* single-channel uint8 pixels the model sees during
    inference (and the same pixels shown on the canvas for TIFF previews).
    The original TIFF file is never modified — the log-compressed image exists
    only in memory.

    Args:
        image_path: str absolute path to the image file.

    Returns:
        uint8 grayscale numpy array (H, W), or None on failure.
    """
    ext = os.path.splitext(image_path)[1].lower()
    if ext in (".tif", ".tiff"):
        try:
            if HAS_TIFFFILE:
                img = tifffile.imread(image_path).astype(np.float32)
            else:
                img = np.array(Image.open(image_path)).astype(np.float32)
            if img.ndim == 3:
                img = img[0]
            log_img = np.log1p(img)
            finite = np.isfinite(log_img)
            if finite.any():
                vmin, vmax = np.percentile(log_img[finite], [1, 99.9])
            else:
                vmin, vmax = 0.0, 1.0
            if vmax > vmin:
                img = np.clip(
                    (log_img - vmin) / (vmax - vmin) * 255, 0, 255
                ).astype(np.uint8)
                img[~finite] = 0
            else:
                img = np.zeros(log_img.shape, dtype=np.uint8)
            return img
        except Exception as e:  # noqa
            logger.warning(f"Failed to load TIFF {image_path}: {e}")
            return None

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        try:
            img = np.array(Image.open(image_path).convert("L"))
        except Exception as e:  # noqa
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None
    return img


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
            "button_detect_plates",
            "button_draw_background",
            "button_fine_tune",
            "input_fine_tune_epochs",
            "edit_fine_tune_epochs",
            "input_fine_tune_split",
            "edit_fine_tune_split",
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
        if self.plate_model_path:
            self.plate_model_path = self.get_model_abs_path(
                self.config, "plate_model_path"
            )
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
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            str(self.model_abs_path),
            num_labels=2,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
        _adapt_first_conv_to_grayscale(self.model)

        # from_pretrained 已加载所有权重，只需手动覆盖 1ch conv
        # （_adapt_first_conv_to_grayscale 用均值初始化了 1ch conv，
        #   需要用 checkpoint 中训练好的 1ch conv 权重覆盖它）
        state_path = Path(self.model_abs_path) / "model.safetensors"
        if not state_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {state_path}"
            )
        from transformers.modeling_utils import load_state_dict
        ckpt = load_state_dict(str(state_path))

        encoder = self.model.model.pixel_level_module.encoder
        root = getattr(encoder, "swin", encoder)
        proj = root.embeddings.patch_embeddings.projection

        ckpt_proj_key = None
        for k in ckpt:
            if "patch_embeddings.projection.weight" in k:
                ckpt_proj_key = k
                break

        if ckpt_proj_key and proj.in_channels == 1:
            proj.weight.data = ckpt[ckpt_proj_key]
            ckpt_bias_key = ckpt_proj_key.replace(".weight", ".bias")
            if ckpt_bias_key in ckpt and proj.bias is not None:
                proj.bias.data = ckpt[ckpt_bias_key]

        self.model = self.model.to(self.device)
        self.model.eval()

        # --- 验证 1ch conv 权重 ---
        proj_weight_ckpt = ckpt.get(ckpt_proj_key) if ckpt_proj_key else None
        if proj_weight_ckpt is not None:
            loaded = root.embeddings.patch_embeddings.projection.weight.data
            match = torch.allclose(loaded, proj_weight_ckpt.to(loaded.device))
            print(f"[Mask2Former] 1ch conv weight: {'MATCHED' if match else 'MISMATCHED'}")
        # ----------------------------------------------------------------

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

        # Merge overlapping or tightly adjacent plates
        if len(plates) > 1:
            plates = sorted(plates, key=lambda b: b[0])
            merged = []
            cur_x, cur_y, cur_w, cur_h = plates[0]
            for px, py, pw, ph in plates[1:]:
                overlap_x = max(0, min(cur_x + cur_w, px + pw) - max(cur_x, px))
                min_w = min(cur_w, pw)
                if overlap_x > min_w * 0.5:
                    new_x = min(cur_x, px)
                    new_y = min(cur_y, py)
                    new_w = max(cur_x + cur_w, px + pw) - new_x
                    new_h = max(cur_y + cur_h, py + ph) - new_y
                    cur_x, cur_y, cur_w, cur_h = new_x, new_y, new_w, new_h
                else:
                    merged.append((cur_x, cur_y, cur_w, cur_h))
                    cur_x, cur_y, cur_w, cur_h = px, py, pw, ph
            merged.append((cur_x, cur_y, cur_w, cur_h))
            plates = merged

        # Snap plate top to image top to avoid cutting off tracks that
        # extend all the way to the upper boundary.
        plates = [
            (x, 0, w, y + h) for (x, y, w, h) in plates
        ]

        self.on_message(f"Stage 1: {len(plates)} plate(s) detected")

        # Stage 2: segment each plate
        all_tracks = []
        all_sources = []

        for plate_idx, (px, py, pw, ph) in enumerate(plates):
            ex, ey, ew, eh = expand_box_to_multiple((px, py, pw, ph), (H, W), multiple=32)
            crop = image_gray[ey:ey + eh, ex:ex + ew]
            try:
                instances = _segment_plate(
                    crop, self.model, self.device, self.model_abs_path,
                    conf=self.conf_threshold,
                )
            except Exception as e:
                logger.warning(f"Segmentation error for plate ({px},{py}): {e}")
                continue

            plate_area = ew * eh
            plate_bbox = [ex, ey, ew, eh]
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

        def _bottom_x(shape):
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

        # Track shapes → linestrip with centerline + width (background drawn on demand)
        track_shapes = []
        for idx, inst in enumerate(all_tracks):
            ribbon, centerline, hw, _bg_polygon, _bg_gap, _bg_side = mask_to_ribbon(
                inst["mask"], keypoint_interval=self._keypoint_interval,
                other_masks=None, force_bg_side=0.0)

            if not ribbon or len(ribbon) < 3:
                # Fallback: use mask contour
                contours, _ = cv2.findContours(inst["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                cnt = max(contours, key=cv2.contourArea)
                ribbon = [[float(p[0][0]), float(p[0][1])] for p in cnt]
                if not centerline or len(centerline) < 2:
                    # Extract centerline from mask skeleton
                    try:
                        from skimage.morphology import skeletonize
                        from skimage.graph import route_through_array
                        skel = skeletonize(inst["mask"].astype(bool))
                        ys, xs = np.where(skel)
                        if len(ys) >= 2:
                            # Simple: top-to-bottom keypoint sampling
                            ys_u = np.unique(ys)
                            centerline = []
                            for y in ys_u:
                                x_vals = xs[ys == y]
                                centerline.append([float(np.mean(x_vals)), float(y)])
                    except Exception:
                        centerline = []
                hw = hw if hw else 5.0

            if not centerline or len(centerline) < 2:
                continue

            track_len = 0.0
            for i in range(1, len(centerline) if centerline else 0):
                dx = centerline[i][0] - centerline[i - 1][0]
                dy = centerline[i][1] - centerline[i - 1][1]
                track_len += (dx * dx + dy * dy) ** 0.5
            if track_len < self._min_track_length and len(centerline) >= 2:
                continue

            track_width = int(round(hw * 2))
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
                shape.add_point(QtCore.QPointF(float(pt[0]), float(pt[1])))
            track_shapes.append(shape)

        track_shapes.sort(key=lambda s: _bottom_x(s))
        for shape in track_shapes:
            track_id += 1
            shape.label = f"track_{track_id}"
            shapes.append(shape)

        self.on_message(f"Done: {len(shapes)} shapes ({track_id} tracks, {source_id} sources)")
        return AutoLabelingResult(shapes, replace=self.replace)

    # ------------------------------------------------------------------
    #  Fine-tuning (uses all "checked" images in the current directory)
    # ------------------------------------------------------------------

    def _build_training_sample(self, image_path, label_path):
        """Reconstruct a training sample from an image + its label file.

        Returns (img_uint8, masks, class_ids) or None if the sample is empty
        or unreadable. The image is log-compressed *in memory* (original TIFF
        is left untouched). Masks are rebuilt from stored annotations:
        track linestrips -> ribbon mask (centerline + width), source polygons
        -> filled polygon mask.
        """
        import re

        img = _load_tiff_log_compressed(image_path)
        if img is None or img.size == 0:
            return None
        H, W = img.shape[:2]

        try:
            lf = LabelFile(label_path)
        except Exception as e:  # noqa
            logger.warning(f"Failed to load label file {label_path}: {e}")
            return None

        masks = []
        class_ids = []
        for shape in lf.shapes:
            label = shape.label or ""
            if re.match(r"^track(_\d+)?$", label):
                cls_id = 0
                od = shape.other_data or {}
                centerline = od.get("centerline")
                track_width = int(od.get("track_width", self._default_track_width))
                if centerline and len(centerline) >= 2:
                    mask = centerline_to_ribbon_mask(
                        centerline, track_width, H, W
                    )
                else:
                    mask = np.zeros((H, W), dtype=np.uint8)
                    pts = np.array(
                        [[int(p.x()), int(p.y())] for p in shape.points],
                        dtype=np.int32,
                    )
                    if len(pts) >= 2:
                        cv2.polylines(
                            mask, [pts], isClosed=False, color=1,
                            thickness=max(1, track_width),
                        )
            elif re.match(r"^source(_\d+)?$", label):
                cls_id = 1
                mask = np.zeros((H, W), dtype=np.uint8)
                pts = np.array(
                    [[int(p.x()), int(p.y())] for p in shape.points],
                    dtype=np.int32,
                )
                if len(pts) >= 3:
                    cv2.fillPoly(mask, [pts], 1)
            else:
                continue

            if mask is None or int(mask.sum()) < 4:
                continue
            masks.append(mask.astype(np.float32))
            class_ids.append(cls_id)

        if not masks:
            return None
        return img, masks, class_ids

    def _pad_training_sample(self, img_u8, masks):
        """Pad an image and its masks to multiples of 32 without resizing.

        The Swin backbone requires both dimensions to be divisible by 32, but
        the reference training loop never resizes plate crops (they are
        already 32× multiples). To keep native resolution and aspect ratio
        here, we only pad the right/bottom edges with the background value
        (0), which becomes -1.0 after Normalize(0.5, 0.5) — matching the
        reference collate_fn's -1.0 pixel padding.
        """
        H, W = img_u8.shape[:2]
        pad_h = (-H) % 32
        pad_w = (-W) % 32
        img_padded = np.pad(
            img_u8, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
        )
        masks_padded = []
        for mask in masks:
            m = np.pad(
                mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
            )
            masks_padded.append(m.astype(np.float32))
        return img_padded, masks_padded

    def fine_tune_on_checked_images(
        self,
        data_items,
        epochs=None,
        lr=None,
        train_ratio=None,
        progress_cb=None,
        cancel_event=None,
    ):
        """Fine-tune the Mask2Former model using checked images.

        Runs synchronously (call it from a worker thread to keep the UI
        responsive). Progress is reported via ``self.on_message`` (status
        text) and ``progress_cb(percent, message)`` (UI progress bar).

        Images are used at their native resolution (only padded to multiples
        of 32 for the Swin backbone) — matching the reference training loop,
        which never resizes plate crops.

        Args:
            data_items: list of (image_path, label_path) tuples.
            epochs: number of training epochs (default from config, else 3).
            lr: learning rate (default from config, else 5e-5).
            train_ratio: fraction of samples used for training (0-1); the
                remaining samples form the validation set.
            progress_cb: optional callable(percent:int, message:str) invoked
                as training progresses.
            cancel_event: optional threading.Event; when set, training is
                aborted and FineTuneCancelledError is raised.

        Returns:
            (num_samples, saved_path) tuple.
        """
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        import random
        import shutil
        from safetensors.torch import save_file

        if self.model is None:
            raise RuntimeError(
                QCoreApplication.translate(
                    "Model", "Model is not loaded."
                )
            )

        epochs = int(epochs or self.config.get("fine_tune_epochs", 3))
        lr = float(lr if lr is not None else self.config.get("fine_tune_lr", 5e-5))
        train_ratio = float(
            train_ratio
            if train_ratio is not None
            else self.config.get("fine_tune_train_ratio", 0.8)
        )
        train_ratio = min(max(train_ratio, 0.0), 1.0)

        def _check_cancel():
            if cancel_event is not None and cancel_event.is_set():
                raise FineTuneCancelledError(
                    QCoreApplication.translate(
                        "Model", "Fine-tuning cancelled by user."
                    )
                )

        def _report(percent, message):
            if progress_cb is not None:
                try:
                    progress_cb(int(percent), str(message))
                except Exception:  # noqa
                    pass

        # 1. Prepare training samples (log-compressed, in memory).
        _check_cancel()
        _report(
            0,
            QCoreApplication.translate("Model", "Preparing fine-tune data..."),
        )
        samples = []
        total_files = max(1, len(data_items))
        for i, (image_path, label_path) in enumerate(data_items):
            _check_cancel()
            sample = self._build_training_sample(image_path, label_path)
            if sample is not None:
                samples.append(sample)
            _report(
                int(10 * (i + 1) / total_files),
                QCoreApplication.translate(
                    "Model", "Preparing sample {i}/{n}..."
                ).format(i=i + 1, n=len(data_items)),
            )

        if not samples:
            raise RuntimeError(
                QCoreApplication.translate(
                    "Model",
                    "No valid checked samples found to fine-tune.",
                )
            )

        # 2. Train/validation split.
        random.shuffle(samples)
        n_train = max(1, int(round(len(samples) * train_ratio)))
        if len(samples) > 1:
            n_train = max(1, min(n_train, len(samples) - 1))
        train_samples = samples[:n_train]
        val_samples = samples[n_train:]

        self.on_message(
            QCoreApplication.translate(
                "Model",
                "Collected {count} sample(s): {t} train / {v} val. "
                "Starting fine-tune...",
            ).format(
                count=len(samples),
                t=len(train_samples),
                v=len(val_samples),
            )
        )
        _report(
            10,
            QCoreApplication.translate("Model", "Starting training..."),
        )

        device = self.device
        model = self.model
        model.to(device)
        model.train()

        weight_decay = float(
            self.config.get("fine_tune_weight_decay", 1e-4)
        )
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs)
        )

        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
            ToTensorV2(),
        ])
        val_transform = A.Compose([
            A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
            ToTensorV2(),
        ])

        def _forward_loss(img_u8, masks, class_ids, transform=transform):
            img_padded, masks_padded = self._pad_training_sample(
                img_u8, masks
            )
            transformed = transform(image=img_padded, masks=masks_padded)
            inp = transformed["image"].float().unsqueeze(0).to(device)
            mask_tensor = torch.stack(
                [m.float() for m in transformed["masks"]]
            ).to(device)
            class_tensor = torch.as_tensor(
                class_ids, dtype=torch.long, device=device
            )
            return model(
                pixel_values=inp,
                mask_labels=[mask_tensor],
                class_labels=[class_tensor],
            ).loss

        # Save paths + helper (one-time backup of the original weights).
        out_dir = Path(self.model_abs_path)
        save_path = out_dir / "model.safetensors"
        backup_path = out_dir / "model.safetensors.orig"

        def _save_model(state_dict):
            if save_path.is_file() and not backup_path.is_file():
                shutil.copy2(str(save_path), str(backup_path))
            save_file(state_dict, str(save_path))

        # Baseline validation loss of the ORIGINAL model (before training).
        # The fine-tuned model is only saved when it beats this baseline.
        baseline_val_loss = None
        if val_samples:
            _check_cancel()
            model.eval()
            base_loss = 0.0
            bn = 0
            with torch.no_grad():
                for img_u8, masks, class_ids in val_samples:
                    _check_cancel()
                    bl = _forward_loss(
                        img_u8, masks, class_ids, transform=val_transform
                    )
                    if bl is not None:
                        base_loss += float(bl.item())
                        bn += 1
            model.train()
            if bn:
                baseline_val_loss = base_loss / bn
            self.on_message(
                QCoreApplication.translate(
                    "Model", "Baseline val_loss={loss:.4f}"
                ).format(
                    loss=baseline_val_loss if baseline_val_loss is not None else 0.0
                )
            )

        best_val_loss = baseline_val_loss
        best_state_dict = None
        saved = False

        total_steps = max(1, epochs * len(train_samples))
        step = 0
        for epoch in range(epochs):
            _check_cancel()
            model.train()
            epoch_loss = 0.0
            n = 0
            order = list(range(len(train_samples)))
            random.shuffle(order)
            for idx in order:
                _check_cancel()
                img_u8, masks, class_ids = train_samples[idx]
                loss = _forward_loss(img_u8, masks, class_ids)
                if loss is None:
                    continue
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()

                epoch_loss += float(loss.item())
                n += 1
                step += 1
                _report(
                    10 + int(80 * step / total_steps),
                    QCoreApplication.translate(
                        "Model", "Epoch {e}/{E} ({s}/{S}) loss={loss:.4f}"
                    ).format(
                        e=epoch + 1,
                        E=epochs,
                        s=step,
                        S=total_steps,
                        loss=float(loss.item()),
                    ),
                )

            scheduler.step()
            avg = epoch_loss / n if n else 0.0

            # Validation on the held-out split.
            val_msg = ""
            epoch_val = None
            if val_samples:
                _check_cancel()
                model.eval()
                val_loss = 0.0
                vn = 0
                with torch.no_grad():
                    for img_u8, masks, class_ids in val_samples:
                        _check_cancel()
                        vloss = _forward_loss(
                            img_u8, masks, class_ids, transform=val_transform
                        )
                        if vloss is not None:
                            val_loss += float(vloss.item())
                            vn += 1
                model.train()
                if vn:
                    epoch_val = val_loss / vn
                val_msg = QCoreApplication.translate(
                    "Model", " val_loss={loss:.4f}"
                ).format(loss=epoch_val if epoch_val is not None else 0.0)

                # Save only when this epoch beats the original model's loss.
                if epoch_val is not None and (
                    best_val_loss is None or epoch_val < best_val_loss
                ):
                    best_val_loss = epoch_val
                    best_state_dict = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                    _check_cancel()
                    _save_model(best_state_dict)
                    saved = True
                    val_msg += QCoreApplication.translate(
                        "Model", " (saved)"
                    )

            self.on_message(
                QCoreApplication.translate(
                    "Model", "Epoch {epoch}/{total} loss={loss:.4f}"
                ).format(epoch=epoch + 1, total=epochs, loss=avg)
                + val_msg
            )

        model.eval()

        # 3. Restore the best weights (if any) and finalize.
        _check_cancel()
        if best_state_dict is not None:
            _report(
                92,
                QCoreApplication.translate("Model", "Restoring best model..."),
            )
            model.load_state_dict(best_state_dict)

        _report(
            100,
            QCoreApplication.translate("Model", "Fine-tuning complete."),
        )

        if saved:
            self.on_message(
                QCoreApplication.translate(
                    "Model", "Fine-tune done. Saved best model to {path}"
                ).format(path=str(save_path))
            )
            return len(samples), str(save_path)

        self.on_message(
            QCoreApplication.translate(
                "Model",
                "Fine-tuning finished but no epoch improved over the "
                "original model (baseline val_loss={base:.4f}). Model "
                "left unchanged.",
            ).format(
                base=baseline_val_loss if baseline_val_loss is not None else 0.0
            )
        )
        return len(samples), None

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

    def recompute_background_strips(self, shapes, H, W):
        """Recompute background strips after user manually adds/edits track(s).

        Called from label_widget when a new track polygon is drawn.
        Groups tracks by plate, re-selects which track gets the bg strip,
        and generates the background polygon directly from the track's own
        centerline and track_width (no re-skeletonization).

        Background centerline = track centerline offset by
        (track_hw + bg_gap) along the normal direction.
        Background width = track's own track_width (identical to the band
        polygon drawn on canvas).

        Args:
            shapes: list of Shape objects from canvas (all shapes)
            H, W: image height and width (pixels)

        Returns:
            (new_bg_shapes, old_bg_shapes): new background Shapes to add,
            and existing background Shapes to remove.
        """
        import re

        # Separate tracks, background shapes, and source shapes
        tracks = []
        bg_shapes_existing = []
        for s in shapes:
            if re.match(r"^track(_\d+)?$", s.label):
                tracks.append(s)
            elif s.other_data and s.other_data.get("_is_bg_strip"):
                bg_shapes_existing.append(s)

        if not tracks:
            return [], bg_shapes_existing

        # Collect existing plate_bboxes from tracks that have them
        plate_bboxes = {}  # pid -> [x, y, w, h]
        for t in tracks:
            pid = t.other_data.get("plate_id") if t.other_data else None
            if pid is not None:
                pb = t.other_data.get("plate_bbox")
                if pb is not None and pid not in plate_bboxes:
                    plate_bboxes[pid] = pb

        # Assign tracks without plate_id (manually drawn) to nearest plate
        next_pid = max(plate_bboxes.keys(), default=-1) + 1
        for t in tracks:
            pid = t.other_data.get("plate_id") if t.other_data else None
            if pid is not None:
                continue
            # Compute centroid
            pts = t.points
            if not pts:
                continue
            cx = sum(p.x() for p in pts) / len(pts)
            cy = sum(p.y() for p in pts) / len(pts)

            best_pid = None
            best_dist = float("inf")
            for pid_check, (px, py, pw, ph) in plate_bboxes.items():
                pcx = px + pw / 2.0
                pcy = py + ph / 2.0
                dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_pid = pid_check

            if best_pid is not None and best_dist < max(W, H) * 0.3:
                t.other_data["plate_id"] = best_pid
            else:
                # Assign to new virtual plate
                t.other_data["plate_id"] = next_pid
                xs = [p.x() for p in pts]
                ys = [p.y() for p in pts]
                plate_bboxes[next_pid] = [
                    int(min(xs)), int(min(ys)),
                    int(max(xs) - min(xs)), int(max(ys) - min(ys)),
                ]
                next_pid += 1

        # Group tracks by plate_id
        tracks_by_plate = {}  # pid -> [index in tracks list]
        for i, t in enumerate(tracks):
            pid = t.other_data.get("plate_id") if t.other_data else None
            if pid is not None:
                tracks_by_plate.setdefault(pid, []).append(i)

        # Rasterize all track polygons to masks.
        # For tracks with centerline data (AI-detected or manually drawn with
        # computed ribbon), use fillPoly with the ribbon polygon for accuracy.
        # Otherwise fall back to polylines (legacy / corrupt data).
        track_masks = []
        for t in tracks:
            od = t.other_data if t.other_data else {}
            centerline = od.get("centerline")
            track_width = od.get("track_width", 14)

            if centerline and len(centerline) >= 2:
                mask = centerline_to_ribbon_mask(centerline, track_width, H, W)
            else:
                mask = np.zeros((H, W), dtype=np.uint8)
                pts_arr = np.array([[int(p.x()), int(p.y())] for p in t.points], dtype=np.int32)
                if len(pts_arr) >= 2:
                    hw_r = track_width // 2
                    cv2.polylines(mask, [pts_arr], isClosed=False, color=1,
                                  thickness=max(1, hw_r * 2))
            track_masks.append(mask)

        # For each plate, determine which track gets the bg strip and generate it
        new_bg_shapes = []
        for pid, idxs in tracks_by_plate.items():
            if len(idxs) == 1:
                bg_track_idx = idxs[0]
                forced_side = 0.0
                other = None
            else:
                # Top → 判断整体偏左偏右; Bottom → 选为哪个轨迹画背景
                top_xs = []
                bottom_xs = []
                for i in idxs:
                    mask = track_masks[i]
                    ys, xp = np.where(mask > 0)
                    if len(ys) == 0:
                        top_xs.append(float("nan"))
                        bottom_xs.append(float("nan"))
                    else:
                        top_y = ys.min()
                        top_xs.append(float(xp[ys == top_y].mean()))
                        bottom_y = ys.max()
                        bottom_xs.append(float(xp[ys == bottom_y].mean()))

                pid_bbox = plate_bboxes.get(pid)
                if pid_bbox is None:
                    combined = np.zeros((H, W), dtype=np.uint8)
                    for i in idxs:
                        combined[track_masks[i] > 0] = 1
                    ys_all, xs_all = np.where(combined > 0)
                    if len(ys_all) == 0:
                        plate_cx = W / 2.0
                    else:
                        plate_cx = (xs_all.min() + xs_all.max()) / 2.0
                else:
                    plate_cx = pid_bbox[0] + pid_bbox[2] / 2.0

                avg_top_x = np.nanmean(top_xs)

                if avg_top_x < plate_cx:
                    # 整体偏左 → 选底部最靠右的轨迹（最靠近中心）→ bg 在右侧 (-1)
                    middle_i = idxs[int(np.nanargmax(bottom_xs))]
                    forced_side = -1.0
                else:
                    # 整体偏右 → 选底部最靠左的轨迹（最靠近中心）→ bg 在左侧 (+1)
                    middle_i = idxs[int(np.nanargmin(bottom_xs))]
                    forced_side = +1.0

                bg_track_idx = middle_i
                # Build other_masks for non-selected tracks in this plate
                other = np.zeros((H, W), dtype=np.uint8)
                for j in idxs:
                    if j != middle_i:
                        other[track_masks[j] > 0] = 1

            # ── Build background directly from track's own data ──
            track = tracks[bg_track_idx]
            track_cl = track.other_data.get("centerline")
            track_width = track.other_data.get("track_width", 14)
            if not track_cl or len(track_cl) < 2:
                continue
            track_hw = track_width / 2.0
            bg_gap = max(30.0, track_hw)
            bg_offset = bg_gap + track_width  # = track_hw + bg_gap + track_hw  → 轨道边→背景边 = bg_gap

            # Track mask Y range for endpoint extension
            track_mask = track_masks[bg_track_idx]
            track_ys = np.where(track_mask > 0)[0]
            if len(track_ys) == 0:
                continue
            track_y_min = float(track_ys.min())
            track_y_max = float(track_ys.max())

            # Determine bg_side
            if abs(forced_side) > 0.001:
                bg_side = forced_side
            else:
                # Single track: try both sides, pick the one with less
                # overlap against other masks in the same plate
                best_side_val = +1.0
                best_overlap = float("inf")
                for sign in [+1.0, -1.0]:
                    test_cl = _offset_centerline(track_cl, sign * bg_offset)
                    test_cl = _extend_centerline_y(test_cl, track_y_min, track_y_max)
                    test_band = centerline_to_ribbon(test_cl, track_width)
                    if not test_band or len(test_band) < 6:
                        continue
                    test_mask = np.zeros((H, W), dtype=np.uint8)
                    cv2.fillPoly(test_mask, [np.array(test_band, dtype=np.int32)], 1)
                    overlap = int(np.sum(test_mask & (other > 0))) if other is not None else 0
                    if overlap < best_overlap:
                        best_overlap = overlap
                        best_side_val = sign
                bg_side = best_side_val

            # Background centerline = track centerline offset along normal
            bg_centerline = _offset_centerline(track_cl, bg_side * bg_offset)

            # Extend centerline endpoints so Y range covers the track's full
            # Y span (track_y_min … track_y_max)
            bg_centerline = _extend_centerline_y(bg_centerline, track_y_min, track_y_max)

            bg_shape = Shape(
                label="background",
                score=0.0,
                shape_type="linestrip",
                flags={},
            )
            bg_shape.other_data["_is_bg_strip"] = True
            bg_shape.other_data["bg_gap"] = bg_gap
            bg_shape.other_data["bg_side"] = bg_side
            bg_shape.other_data["plate_id"] = pid
            bg_shape.other_data["track_width"] = track_width
            bg_shape.other_data["centerline"] = bg_centerline
            bg_shape.other_data["ref_centerline"] = track_cl  # 原始参考轨迹中心线
            for x, y in bg_centerline:
                bg_shape.add_point(QtCore.QPointF(float(x), float(y)))
            new_bg_shapes.append(bg_shape)

        return new_bg_shapes, bg_shapes_existing
