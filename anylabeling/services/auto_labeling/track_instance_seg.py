# -*- coding: utf-8 -*-
"""Particle track instance segmentation model for X-AnyLabeling.

Plate-first pipeline (mirrors project/infer.py):
  1. YOLO plate detection (CV fallback / whole image).
  2. Per-plate single forward pass -> semantic + embedding + heatmap.
  3. Origin peak detection (max-pool NMS).
  4. Polar-guided DBSCAN clustering -> track instances.
  5. Optional arc-residual refinement (split / merge), UI toggle.

All model code inlined to avoid cross-thread file handle issues on Windows.
"""
import os
import gc
import traceback
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import List, Optional, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2

from PyQt6 import QtCore
from PyQt6.QtCore import QCoreApplication

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult


# ======================================================================
#  Model architecture (inlined from project/models/)
# ======================================================================

class UNetPlusPlusBackbone(nn.Module):
    """U-Net++ backbone using segmentation-models-pytorch."""

    def __init__(
        self,
        encoder_name: str = "resnet18",
        encoder_weights: Optional[str] = None,
        in_channels: int = 1,
        decoder_channels: tuple = (256, 128, 64, 64, 64),
    ):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
            decoder_channels=decoder_channels,
        )
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.out_channels = decoder_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class SemanticHead(nn.Module):
    """1-channel track-area logits."""

    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.conv(features)


class HeatmapHead(nn.Module):
    """1-channel origin heatmap logits (CenterNet-style)."""

    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )
        nn.init.constant_(self.conv[-1].bias, -4.6)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.conv(features)


class EmbeddingHead(nn.Module):
    """D-dimensional per-pixel embedding vectors."""

    def __init__(self, in_channels: int = 256, embedding_dim: int = 8):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, embedding_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.conv(features)


class TrackSegModel(nn.Module):
    """UNet++ backbone + semantic / embedding / heatmap heads."""

    def __init__(
        self,
        encoder_name: str = "resnet18",
        encoder_weights: Optional[str] = None,
        in_channels: int = 1,
        embedding_dim: int = 8,
    ):
        super().__init__()
        self.backbone = UNetPlusPlusBackbone(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
        )
        feat_channels = self.backbone.out_channels
        self.semantic_head = SemanticHead(in_channels=feat_channels)
        self.embedding_head = EmbeddingHead(
            in_channels=feat_channels, embedding_dim=embedding_dim
        )
        self.heatmap_head = HeatmapHead(in_channels=feat_channels)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        semantic = self.semantic_head(features)
        embedding = self.embedding_head(features)
        embedding = F.normalize(embedding, p=2, dim=1)
        heatmap = self.heatmap_head(features)
        return semantic, embedding, heatmap


# ======================================================================
#  Geometry helpers (inlined from project/utils/plates.py, arc_geometry.py)
# ======================================================================

def _snap_to_multiple(value: int, multiple: int = 32) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _expand_axis_to_multiple(
    start: int, length: int, limit: int, multiple: int = 32
) -> Tuple[int, int]:
    """Expand one axis to a multiple, compensating on the opposite side."""
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


def expand_box_to_multiple(
    box: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    multiple: int = 32,
) -> Tuple[int, int, int, int]:
    """Expand (x, y, w, h) inside image bounds to multiples of 32."""
    x, y, w, h = box
    h_img, w_img = image_shape[:2]
    ex, ew = _expand_axis_to_multiple(x, w, w_img, multiple)
    ey, eh = _expand_axis_to_multiple(y, h, h_img, multiple)
    return ex, ey, ew, eh


def compute_polar_maps(
    origin: Tuple[float, float], height: int, width: int
) -> Tuple[np.ndarray, np.ndarray]:
    """(theta, radius) maps relative to *origin*; theta in [0, 2pi)."""
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    dx = xs - origin[0]
    dy = ys - origin[1]
    radius = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)
    return theta, radius


def augment_embedding(
    embedding: np.ndarray,
    origin: Tuple[float, float],
    alpha: float = 0.25,
    beta: float = 0.0,
) -> np.ndarray:
    """Append polar channels (alpha*cos, alpha*sin[, beta*log1p(r)])."""
    H, W = embedding.shape[1], embedding.shape[2]
    theta, radius = compute_polar_maps(origin, H, W)

    cos_a = np.cos(theta) * alpha
    sin_a = np.sin(theta) * alpha
    extra = [cos_a, sin_a]

    if beta > 0:
        extra.append(np.log1p(radius) * beta)

    return np.concatenate([embedding, np.stack(extra, axis=0)], axis=0)


# ======================================================================
#  Plate detection (YOLO with CV / whole-image fallback)
# ======================================================================

def _split_by_vertical_gaps(
    box: Tuple[int, int, int, int],
    binary: np.ndarray,
    gray: np.ndarray,
) -> List[Tuple[int, int, int, int]]:
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

    split_points = sorted(set(split_points))

    parts = []
    left = 0
    for split_x in split_points:
        if split_x - left >= min_part_width:
            parts.append((x + left, y, split_x - left, h))
            left = split_x
    if w - left >= min_part_width:
        parts.append((x + left, y, w - left, h))

    return parts if len(parts) > 1 else [box]


def find_plates_cv(
    image_gray: np.ndarray,
    min_area_ratio: float = 0.01,
    close_kernel: int = 25,
) -> List[Tuple[int, int, int, int]]:
    """Classic-CV plate detection: Otsu + closing + connected components."""
    _, binary = cv2.threshold(
        image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    raw_binary = binary.copy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (close_kernel, close_kernel)
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    h_img, w_img = image_gray.shape[:2]
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
        plates.extend(_split_by_vertical_gaps(box, raw_binary, image_gray))

    return sorted(plates, key=lambda p: p[0])


# ======================================================================
#  Clustering (inlined from project/utils/clustering.py)
# ======================================================================

def cluster_embeddings(
    semantic_prob: np.ndarray,
    embedding: np.ndarray,
    threshold: float = 0.5,
    eps: float = 0.5,
    min_samples: int = 10,
    origin: Optional[Tuple[float, float]] = None,
    polar_alpha: float = 0.25,
    polar_beta: float = 0.0,
    spatial_weight: float = 1.0,
) -> Tuple[np.ndarray, int]:
    """DBSCAN clustering of embeddings on foreground pixels.

    Embedding vectors are augmented with normalized spatial coordinates
    [x/W, y/H] so that DBSCAN respects both feature similarity and
    physical proximity.  *spatial_weight* scales the spatial influence.
    """
    from sklearn.cluster import DBSCAN

    H, W = semantic_prob.shape
    fg_mask = semantic_prob > threshold

    if fg_mask.sum() < min_samples:
        return np.zeros((H, W), dtype=np.int32), 0

    ys, xs = np.where(fg_mask)
    emb_vecs = embedding[:, ys, xs].transpose(1, 0)  # (N, D)

    norms = np.linalg.norm(emb_vecs, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    emb_vecs = emb_vecs / norms

    # Append normalized spatial coordinates so DBSCAN respects proximity
    if spatial_weight > 0:
        spatial = np.stack(
            [xs.astype(np.float32) / max(W, 1),
             ys.astype(np.float32) / max(H, 1)],
            axis=1,
        )  # (N, 2), range [0, 1]
        spatial *= spatial_weight
        emb_vecs = np.concatenate([emb_vecs, spatial], axis=1)

    # Legacy polar augmentation (disabled by default, polar_alpha=0)
    if origin is not None and polar_alpha > 0:
        augmented = augment_embedding(
            embedding, origin, alpha=polar_alpha, beta=polar_beta
        )
        extra = augmented[:, ys, xs].transpose(1, 0)[:, embedding.shape[0]:]
        extra_norms = np.linalg.norm(extra, axis=1, keepdims=True)
        extra_norms = np.clip(extra_norms, 1e-8, None)
        extra = extra / extra_norms
        emb_vecs = np.concatenate([emb_vecs, extra], axis=1)

    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clusterer.fit_predict(emb_vecs)

    instance_map = np.zeros((H, W), dtype=np.int32)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]

    for idx, label in enumerate(unique_labels, start=1):
        mask = labels == label
        instance_map[ys[mask], xs[mask]] = idx

    return instance_map, len(unique_labels)


def _cleanup_small(instance_map: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components smaller than *min_size* pixels."""
    cleaned = np.zeros_like(instance_map)
    next_id = 1
    for inst_id in [i for i in np.unique(instance_map) if i > 0]:
        mask = (instance_map == inst_id).astype(np.uint8)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        for lbl in range(1, len(stats)):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_size:
                cleaned[labels == lbl] = next_id
                next_id += 1
    return cleaned


def _detect_origins(
    heatmap: np.ndarray,
    threshold: float = 0.3,
    nms_kernel: int = 15,
) -> list:
    """CenterNet-style peak detection: max-pool NMS + threshold.

    Returns at most one (x, y, score) tuple — the highest-scoring peak.
    """
    hm_t = torch.from_numpy(heatmap)[None, None]
    pad = nms_kernel // 2
    hmax = F.max_pool2d(hm_t, nms_kernel, stride=1, padding=pad)
    peaks = (hm_t == hmax) & (hm_t >= threshold)
    ys, xs = torch.nonzero(peaks[0, 0], as_tuple=True)
    origins = [
        (int(x), int(y), float(heatmap[y, x]))
        for x, y in zip(xs.tolist(), ys.tolist())
    ]
    origins.sort(key=lambda p: -p[2])
    return origins[:1]


# ======================================================================
#  Arc-residual refinement (numpy port of ArcConsistencyLoss fitting)
# ======================================================================

RHO_MIN = 5.0  # px — theta is ill-defined too close to the origin


def fit_arc(
    ys: np.ndarray,
    xs: np.ndarray,
    weights: np.ndarray,
    origin: Tuple[float, float],
) -> Optional[dict]:
    """Weighted LLS fit of rho = a*cos(theta) + b*sin(theta).

    A circle through the origin. Returns dict(a, b, R, phi, nrmse) or
    None if the fit is degenerate.
    """
    dx = xs.astype(np.float64) - origin[0]
    dy = ys.astype(np.float64) - origin[1]
    rho = np.sqrt(dx * dx + dy * dy)

    valid = rho > RHO_MIN
    if valid.sum() < 10:
        return None
    rho = rho[valid]
    theta = np.arctan2(dy[valid], dx[valid])
    w = weights[valid].astype(np.float64)
    w_sum = w.sum()
    if w_sum < 1.0:
        return None

    c = np.cos(theta)
    s = np.sin(theta)
    m00 = (w * c * c).sum()
    m01 = (w * c * s).sum()
    m11 = (w * s * s).sum()
    v0 = (w * rho * c).sum()
    v1 = (w * rho * s).sum()
    ridge = 1e-4 * w_sum
    M = np.array([[m00 + ridge, m01], [m01, m11 + ridge]])
    v = np.array([v0, v1])
    try:
        a_fit, b_fit = np.linalg.solve(M, v)
    except np.linalg.LinAlgError:
        return None

    R = 0.5 * np.sqrt(a_fit * a_fit + b_fit * b_fit)
    if R < 1.0:
        return None

    resid = rho - a_fit * c - b_fit * s
    mse = (w * resid * resid).sum() / w_sum
    nrmse = np.sqrt(max(mse, 1e-8)) / R

    return {
        "a": float(a_fit),
        "b": float(b_fit),
        "R": float(R),
        "phi": float(np.arctan2(b_fit, a_fit)),
        "nrmse": float(nrmse),
    }


def _relabel(instance_map: np.ndarray) -> np.ndarray:
    """Compact instance IDs to 1..N."""
    out = np.zeros_like(instance_map)
    for new_id, old_id in enumerate(
        [i for i in np.unique(instance_map) if i > 0], start=1
    ):
        out[instance_map == old_id] = new_id
    return out


def _endpoint_gap(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Minimum pixel distance between two masks (subsampled for speed)."""
    ya, xa = np.where(mask_a)
    yb, xb = np.where(mask_b)
    if len(ya) == 0 or len(yb) == 0:
        return np.inf
    step_a = max(1, len(ya) // 200)
    step_b = max(1, len(yb) // 200)
    pa = np.stack([xa[::step_a], ya[::step_a]], axis=1).astype(np.float64)
    pb = np.stack([xb[::step_b], yb[::step_b]], axis=1).astype(np.float64)
    d2 = ((pa[:, None, :] - pb[None, :, :]) ** 2).sum(axis=2)
    return float(np.sqrt(d2.min()))


def arc_refine(
    instance_map: np.ndarray,
    semantic_prob: np.ndarray,
    embedding: np.ndarray,
    origin: Tuple[float, float],
    split_nrmse: float = 0.05,
    merge_nrmse: float = 0.03,
    merge_r_tol: float = 0.10,
    merge_phi_tol: float = 0.15,
    merge_gap: float = 40.0,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 10,
    polar_alpha: float = 0.25,
    polar_beta: float = 0.0,
) -> np.ndarray:
    """Arc-residual verification of clusters (requires origin).

    Split: clusters whose arc-fit nrmse exceeds *split_nrmse* are
    re-clustered with a tighter DBSCAN eps.
    Merge: cluster pairs that fit the same origin-circle (both good fits,
    similar R and phi, small spatial gap) are merged.
    """
    # ---- Split pass ----
    result = instance_map.copy()
    for inst_id in [i for i in np.unique(result) if i > 0]:
        mask = result == inst_id
        ys, xs = np.where(mask)
        fit = fit_arc(ys, xs, semantic_prob[ys, xs], origin)
        if fit is None or fit["nrmse"] <= split_nrmse:
            continue
        # Re-cluster this cluster's pixels with tighter eps
        sub_sem = np.where(mask, semantic_prob, 0.0)
        sub_map, n_sub = cluster_embeddings(
            sub_sem,
            embedding,
            threshold=0.0,
            eps=dbscan_eps * 0.6,
            min_samples=dbscan_min_samples,
            origin=origin,
            polar_alpha=polar_alpha,
            polar_beta=polar_beta,
        )
        if n_sub <= 1:
            continue
        result[mask] = 0
        next_id = int(result.max())
        for sub_id in range(1, n_sub + 1):
            next_id += 1
            result[sub_map == sub_id] = next_id

    # ---- Merge pass ----
    ids = [i for i in np.unique(result) if i > 0]
    fits = {}
    masks = {}
    for i in ids:
        m = result == i
        ys, xs = np.where(m)
        fits[i] = fit_arc(ys, xs, semantic_prob[ys, xs], origin)
        masks[i] = m

    merged_into = {}

    def _root(i):
        while i in merged_into:
            i = merged_into[i]
        return i

    for a_idx in range(len(ids)):
        for b_idx in range(a_idx + 1, len(ids)):
            ia, ib = _root(ids[a_idx]), _root(ids[b_idx])
            if ia == ib:
                continue
            fa, fb = fits.get(ia), fits.get(ib)
            if fa is None or fb is None:
                continue
            if fa["nrmse"] > merge_nrmse or fb["nrmse"] > merge_nrmse:
                continue
            r_max = max(fa["R"], fb["R"])
            if abs(fa["R"] - fb["R"]) / r_max > merge_r_tol:
                continue
            dphi = abs(fa["phi"] - fb["phi"])
            dphi = min(dphi, 2 * np.pi - dphi)
            if dphi > merge_phi_tol:
                continue
            ma = result == ia
            mb = result == ib
            if _endpoint_gap(ma, mb) > merge_gap:
                continue
            # Merge ib into ia and re-fit the union
            result[mb] = ia
            merged_into[ib] = ia
            ys, xs = np.where(result == ia)
            fits[ia] = fit_arc(ys, xs, semantic_prob[ys, xs], origin)

    return _relabel(result)


# ======================================================================
#  X-AnyLabeling model class
# ======================================================================

class TrackInstanceSeg(Model):
    """Plate-first track instance segmentation → polygon + point output."""

    class Meta:
        required_config_names = ["type", "name", "display_name"]
        widgets = [
            "button_run",
            "input_conf",
            "edit_conf",
            "toggle_preserve_existing_annotations",
            "toggle_arc_residual",
        ]
        output_modes = {
            "polygon": QCoreApplication.translate("Model", "Polygon"),
            "point": QCoreApplication.translate("Model", "Point"),
        }
        default_output_mode = "polygon"

    def __init__(self, model_config, on_message) -> None:
        super().__init__(model_config, on_message)

        self.model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not self.model_abs_path or not os.path.isfile(self.model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model", "Checkpoint not found: {path}"
                ).format(path=self.model_abs_path)
            )

        self.plate_model_path = self.config.get("plate_model_path", "")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.cfg = None
        self.plate_model = None
        self._load_model()

        self.replace = True
        self.conf_threshold = self.config.get("conf_threshold", 0.5)

        # Arc-residual defaults (overridden by UI if toggles added)
        arc_cfg = self.config.get("arc_residual", {})
        self._arc_residual_enabled = False
        self._arc_split_nrmse = arc_cfg.get("split_nrmse", 0.05)
        self._arc_merge_nrmse = arc_cfg.get("merge_nrmse", 0.03)
        self._arc_merge_r_tol = arc_cfg.get("merge_r_tol", 0.10)
        self._arc_merge_phi_tol = arc_cfg.get("merge_phi_tol", 0.15)
        self._arc_merge_gap = arc_cfg.get("merge_gap", 40)

    # ------------------------------------------------------------------
    #  Loading / unloading
    # ------------------------------------------------------------------

    def _load_model(self):
        self.on_message(
            QCoreApplication.translate("Model", "Loading track instance seg model…")
        )
        state = torch.load(
            self.model_abs_path, map_location=self.device, weights_only=False
        )
        self.cfg = state["config"]

        mc = self.cfg.get("model", {})
        self.model = TrackSegModel(
            encoder_name=mc.get("encoder", "resnet18"),
            encoder_weights=None,
            in_channels=mc.get("in_channels", 1),
            embedding_dim=mc.get("embedding_dim", 8),
        ).to(self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()

        # Lazy-load plate YOLO
        if self.plate_model_path and os.path.isfile(self.plate_model_path):
            try:
                from ultralytics import YOLO

                self.plate_model = YOLO(self.plate_model_path)
                self.on_message(
                    QCoreApplication.translate("Model", "Plate YOLO loaded.")
                )
            except Exception as e:
                logger.warning(f"Plate YOLO load failed: {e} — using CV fallback")
                self.plate_model = None

        self.on_message(
            QCoreApplication.translate("Model", "Track instance seg model loaded.")
        )

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.plate_model is not None:
            del self.plate_model
            self.plate_model = None
        gc.collect()

    # ------------------------------------------------------------------
    #  UI callbacks
    # ------------------------------------------------------------------

    def set_auto_labeling_conf(self, value):
        """Confidence threshold (passed through to downstream if needed)."""
        self.conf_threshold = value

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        self.replace = not state

    def set_auto_labeling_arc_residual(self, enabled: bool):
        self._arc_residual_enabled = enabled

    def set_auto_labeling_arc_split_nrmse(self, value: float):
        self._arc_split_nrmse = float(value)

    def set_auto_labeling_arc_merge_nrmse(self, value: float):
        self._arc_merge_nrmse = float(value)

    # ------------------------------------------------------------------
    #  Prediction
    # ------------------------------------------------------------------

    def predict_shapes(self, image, image_path=None):
        if self.model is None:
            return AutoLabelingResult([], replace=self.replace)

        try:
            return self._predict_shapes_impl(image, image_path)
        except Exception:
            logger.error(
                f"TrackInstanceSeg predict_shapes failed:\n{traceback.format_exc()}"
            )
            return AutoLabelingResult([], replace=self.replace)

    def _predict_shapes_impl(self, image, image_path=None):
        if self.model is None:
            return AutoLabelingResult([], replace=self.replace)

        # --- 1. Convert to grayscale numpy ---
        try:
            image_rgb = qt_img_to_rgb_cv_img(image)
            image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        except Exception as e:
            logger.warning(f"Image conversion error: {e}")
            return AutoLabelingResult([], replace=self.replace)

        H_img, W_img = image_gray.shape[:2]

        # --- 2. Plate detection ---
        plates = self._detect_plates(image_rgb, image_gray)

        # --- 3. Postprocess config ---
        pp = self.cfg.get("postprocess", {})

        transform = A.Compose(
            [A.Normalize(mean=(0.5,), std=(0.5,)), ToTensorV2()]
        )

        # --- 4. Per-plate inference + clustering (avoids global DBSCAN OOM) ---
        all_shapes = []
        all_origins = []  # (gx, gy, score) in global coords
        next_inst_id = 0  # track counter across plates

        for px, py, pw, ph in plates:
            ex, ey, ew, eh = expand_box_to_multiple(
                (px, py, pw, ph), (H_img, W_img), multiple=32
            )
            patch = image_gray[ey:ey + eh, ex:ex + ew]
            if patch.size == 0:
                continue

            transformed = transform(image=patch)
            patch_t = transformed["image"].unsqueeze(0).to(self.device)

            with torch.no_grad():
                sem, emb, heat = self.model(patch_t)

            sem_np = torch.sigmoid(sem[0, 0]).cpu().numpy()
            emb_np = emb[0].cpu().numpy()
            heat_np = torch.sigmoid(heat[0, 0]).cpu().numpy()
            del patch_t

            # Origin detection within this patch
            origins_patch = _detect_origins(
                heat_np,
                threshold=pp.get("heatmap_threshold", 0.3),
                nms_kernel=pp.get("peak_nms_kernel", 15),
            )
            # Offset to global coords and collect
            for ox, oy, score in origins_patch:
                all_origins.append((ex + ox, ey + oy, score))

            origin_patch = (
                (origins_patch[0][0], origins_patch[0][1])
                if origins_patch
                else None
            )

            # Cluster within this patch only
            patch_inst_map, n_inst = cluster_embeddings(
                sem_np,
                emb_np,
                threshold=pp.get("semantic_threshold", 0.5),
                eps=pp.get("dbscan_eps", 0.5),
                min_samples=pp.get("dbscan_min_samples", 10),
                origin=origin_patch,
                polar_alpha=0,
                polar_beta=0,
            )

            if n_inst == 0:
                continue

            # Cleanup tiny fragments
            patch_inst_map = _cleanup_small(
                patch_inst_map, pp.get("min_component_size", 50)
            )

            # Arc-residual refinement (per-plate, optional)
            if (
                self._arc_residual_enabled
                and origin_patch is not None
                and n_inst > 0
            ):
                patch_inst_map = arc_refine(
                    patch_inst_map,
                    sem_np,
                    emb_np,
                    origin_patch,
                    split_nrmse=self._arc_split_nrmse,
                    merge_nrmse=self._arc_merge_nrmse,
                    merge_r_tol=self._arc_merge_r_tol,
                    merge_phi_tol=self._arc_merge_phi_tol,
                    merge_gap=self._arc_merge_gap,
                    dbscan_eps=pp.get("dbscan_eps", 0.5),
                    dbscan_min_samples=pp.get("dbscan_min_samples", 10),
                    polar_alpha=0,
                    polar_beta=0,
                )

            # Convert this patch's instances to globally-offset shapes
            patch_shapes = self._patch_masks_to_shapes(
                patch_inst_map, ex, ey, next_inst_id
            )
            all_shapes.extend(patch_shapes)
            # Update counter: count unique instance ids in this patch
            unique_ids = [i for i in np.unique(patch_inst_map) if i > 0]
            next_inst_id += len(unique_ids)

        # --- 5. Origin points ---
        for ox, oy, score in all_origins:
            shape = Shape(
                label="origin",
                shape_type="point",
                flags={},
            )
            shape.add_point(QtCore.QPointF(float(ox), float(oy)))
            all_shapes.append(shape)

        return AutoLabelingResult(all_shapes, replace=self.replace)

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _detect_plates(self, image_rgb, image_gray):
        """YOLO → CV fallback → whole image."""
        plates = []

        if self.plate_model is not None:
            try:
                results = self.plate_model.predict(
                    source=image_rgb,
                    imgsz=640,
                    conf=0.25,
                    iou=0.5,
                    verbose=False,
                )
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes.xyxy.cpu().numpy():
                            x1, y1, x2, y2 = box[:4]
                            plates.append(
                                (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                            )
            except Exception as e:
                logger.warning(f"Plate YOLO inference failed: {e}")

        if not plates:
            plates = find_plates_cv(image_gray)

        if not plates:
            h, w = image_gray.shape[:2]
            plates = [(0, 0, w, h)]

        return plates

    @staticmethod
    def _patch_masks_to_shapes(instance_map, offset_x, offset_y, start_id=0):
        """Convert per-patch instance map to globally-offset Shape objects.

        Args:
            instance_map: (H, W) int32 array, instance ids 1..N.
            offset_x, offset_y: global offset of the patch top-left corner.
            start_id: starting track id for labeling (incremented across plates).

        Returns:
            list of Shape objects with globally-offset point coordinates.
        """
        shapes = []

        # Track polygons
        for inst_id in sorted(
            [i for i in np.unique(instance_map) if i > 0]
        ):
            mask = (instance_map == inst_id).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                if len(contour) < 3:
                    continue
                epsilon = 0.002 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                pts = approx.reshape(-1, 2)
                if len(pts) < 3:
                    continue
                global_label = start_id + inst_id
                shape = Shape(
                    label=f"track_{global_label}",
                    shape_type="polygon",
                    flags={},
                )
                for pt in pts:
                    shape.add_point(
                        QtCore.QPointF(
                            float(pt[0]) + offset_x,
                            float(pt[1]) + offset_y,
                        )
                    )
                shapes.append(shape)

        return shapes

