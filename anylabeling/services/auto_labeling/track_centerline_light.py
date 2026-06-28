# -*- coding: utf-8 -*-
"""Particle track centerline detection model for X-AnyLabeling.

All model code inlined to avoid cross-thread file handle issues on Windows.
"""
import os
import gc
import cv2
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Optional, Tuple
from collections import deque

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
#  Model architecture ? Lightweight (MobileNetV2 + reduced UNet++)
# ======================================================================

class LightweightBackbone(nn.Module):
    """MobileNetV2 encoder + reduced-channel UNet++ decoder."""

    def __init__(
        self,
        encoder_name: str = "mobilenet_v2",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        decoder_channels: tuple = (128, 64, 32, 16, 16),
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


class LightSemanticHead(nn.Module):
    """1-channel centerline logits with depthwise separable conv."""

    def __init__(self, in_channels: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.conv(features)


class LightEmbeddingHead(nn.Module):
    """4-dimensional per-pixel embedding vectors."""

    def __init__(self, in_channels: int = 16, embedding_dim: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, embedding_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.conv(features)


class LightTrackSegModel(nn.Module):
    """MobileNetV2 + UNet++ (reduced) + SemanticHead + EmbeddingHead."""

    def __init__(
        self,
        encoder_name: str = "mobilenet_v2",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        embedding_dim: int = 4,
        decoder_channels: tuple = (128, 64, 32, 16, 16),
    ):
        super().__init__()
        self.backbone = LightweightBackbone(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            decoder_channels=decoder_channels,
        )
        feat_channels = self.backbone.out_channels
        self.semantic_head = LightSemanticHead(in_channels=feat_channels)
        self.embedding_head = LightEmbeddingHead(
            in_channels=feat_channels, embedding_dim=embedding_dim
        )

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        semantic = self.semantic_head(features)
        embedding = self.embedding_head(features)
        return semantic, embedding


# ======================================================================
#  X-AnyLabeling model class
# ======================================================================

class TrackCenterlineLight(Model):

    class Meta:
        required_config_names = ["type", "name", "display_name"]
        widgets = [
            "button_run",
            "toggle_preserve_existing_annotations",
            "edit_min_track_length",
            "input_min_track_length",
            "edit_keypoint_interval",
            "input_keypoint_interval",
        ]
        output_modes = {
            "polygon": QCoreApplication.translate("Model", "Polygon"),
        }
        default_output_mode = "linestrip"

    def __init__(self, model_config, on_message) -> None:
        super().__init__(model_config, on_message)

        self.model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not self.model_abs_path or not os.path.isfile(self.model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model", "Checkpoint not found: {path}"
                ).format(path=self.model_abs_path)
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.cfg = None
        self._load_model()
        self.replace = True

    def _load_model(self):
        self.on_message("Loading centerline model...")
        state = torch.load(
            self.model_abs_path, map_location=self.device, weights_only=False
        )
        self.cfg = state["config"]

        self.model = LightTrackSegModel(
            encoder_name=self.cfg["model"]["encoder"],
            encoder_weights=None,
            in_channels=self.cfg["model"]["in_channels"],
            embedding_dim=self.cfg["model"]["embedding_dim"],
        ).to(self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()
        self.on_message("Centerline model loaded.")

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        self.replace = not state

    def set_auto_labeling_min_track_length(self, value):
        self._min_track_length = int(value)

    def set_auto_labeling_keypoint_interval(self, value):
        self._keypoint_interval = int(value)

    def predict_shapes(self, image, image_path=None):
        if self.model is None:
            return AutoLabelingResult([], replace=self.replace)

        import traceback
        try:
            image_bgr = qt_img_to_rgb_cv_img(image)
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)
            sem_prob, embedding = self._sliding_window_infer(image_bgr)
        except Exception as e:
            logger.warning(f"Inference error: {e}\n{traceback.format_exc()}")
            return AutoLabelingResult([], replace=self.replace)

        pp = self.cfg["postprocess"]
        min_track_length = getattr(self, "_min_track_length", pp.get("min_track_length", 30))
        instance_map, _ = cluster_embeddings(
            embedding, sem_prob,
            threshold=pp["semantic_threshold"],
            eps=pp["dbscan_eps"],
            min_samples=pp["dbscan_min_samples"],
        )
        instance_map = self._cleanup_small(instance_map, pp["min_component_size"])

        shapes = []
        instance_ids = [i for i in np.unique(instance_map) if i > 0]
        for inst_id in instance_ids:
            mask = (instance_map == inst_id).astype(np.uint8)
            polyline_points = self._mask_to_polyline(mask)
            # Filter by actual polyline length (not just pixel area)
            if polyline_points is not None:
                track_len = self._polyline_length(polyline_points)
                if track_len < min_track_length:
                    continue
            if polyline_points is not None and len(polyline_points) >= 2:
                shape = Shape(
                    label=f"track_{inst_id}",
                    shape_type="linestrip",
                    flags={},
                )
                # Store centerline for later band expansion
                shape.other_data["centerline"] = polyline_points
                for pt in polyline_points:
                    shape.add_point(QtCore.QPointF(pt[0], pt[1]))
                shapes.append(shape)

        return AutoLabelingResult(shapes, replace=self.replace)

    # --- Sliding-window inference ---

    def _sliding_window_infer(self, image_bgr: np.ndarray):
        patch_size = self.cfg["data"]["patch_size"]
        embedding_dim = self.cfg["model"]["embedding_dim"]

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        H_orig, W_orig = image_rgb.shape[:2]
        stride = patch_size // 2
        pad_h = (stride - H_orig % stride) % stride
        pad_w = (stride - W_orig % stride) % stride
        if pad_h or pad_w:
            image_rgb = np.pad(
                image_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect"
            )
        H_pad, W_pad = image_rgb.shape[:2]

        transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        sem_accum = np.zeros((H_pad, W_pad), dtype=np.float32)
        sem_count = np.zeros((H_pad, W_pad), dtype=np.float32)
        emb_accum = np.zeros((embedding_dim, H_pad, W_pad), dtype=np.float32)

        ys = list(range(0, H_pad - patch_size + 1, stride))
        xs = list(range(0, W_pad - patch_size + 1, stride))
        if H_pad >= patch_size:
            y_edge = H_pad - patch_size
            if y_edge not in ys:
                ys.append(y_edge)
        if W_pad >= patch_size:
            x_edge = W_pad - patch_size
            if x_edge not in xs:
                xs.append(x_edge)

        with torch.no_grad():
            for y in ys:
                for x in xs:
                    patch_rgb = image_rgb[y: y + patch_size, x: x + patch_size]
                    patch_t = transform(image=patch_rgb)["image"]
                    patch_t = patch_t.unsqueeze(0).to(self.device)
                    sem_pred, emb_pred = self.model(patch_t)
                    sem_prob = torch.sigmoid(sem_pred[0, 0]).cpu().numpy()
                    emb_vec = emb_pred[0].cpu().numpy()
                    del sem_pred, emb_pred, patch_t
                    sem_accum[y: y + patch_size, x: x + patch_size] += sem_prob
                    sem_count[y: y + patch_size, x: x + patch_size] += 1.0
                    emb_accum[:, y: y + patch_size, x: x + patch_size] += emb_vec

        sem_count = np.clip(sem_count, 1e-8, None)
        sem_prob = sem_accum / sem_count
        embedding = emb_accum / sem_count[np.newaxis, :, :]
        del sem_accum, sem_count, emb_accum
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return sem_prob[:H_orig, :W_orig], embedding[:, :H_orig, :W_orig]

    # --- Post-processing ---

    @staticmethod
    def _cleanup_small(instance_map: np.ndarray, min_size: int) -> np.ndarray:
        cleaned = np.zeros_like(instance_map)
        inst_ids = [i for i in np.unique(instance_map) if i > 0]
        next_id = 1
        for inst_id in inst_ids:
            mask = (instance_map == inst_id).astype(np.uint8)
            _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for lbl in range(1, len(stats)):
                if stats[lbl, cv2.CC_STAT_AREA] >= min_size:
                    cleaned[labels == lbl] = next_id
                    next_id += 1
        return cleaned

    @staticmethod
    def _polyline_to_band_polygon(points, half_width):
        if len(points) < 2:
            return points
        n = len(points)
        left_pts = []
        right_pts = []
        for i in range(n):
            if i == 0:
                dx = points[1][0] - points[0][0]
                dy = points[1][1] - points[0][1]
            elif i == n - 1:
                dx = points[n-1][0] - points[n-2][0]
                dy = points[n-1][1] - points[n-2][1]
            else:
                dx = points[i+1][0] - points[i-1][0]
                dy = points[i+1][1] - points[i-1][1]
            length = (dx*dx + dy*dy) ** 0.5
            if length < 1e-6:
                continue
            nx = -dy / length
            ny = dx / length
            left_pts.append((points[i][0] + nx * half_width, points[i][1] + ny * half_width))
            right_pts.append((points[i][0] - nx * half_width, points[i][1] - ny * half_width))
        if len(left_pts) < 2:
            return points
        return left_pts + list(reversed(right_pts))

    @staticmethod
    def _polyline_length(points: list) -> float:
        """Euclidean length of a polyline."""
        total = 0.0
        for i in range(1, len(points)):
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]
            total += (dx * dx + dy * dy) ** 0.5
        return total

    def _mask_to_polyline(self,mask: np.ndarray, num_keypoints: int = 20):
        skeleton = TrackCenterline._skeletonize(mask)
        if skeleton.sum() < 2:
            return None

        ys, xs = np.where(skeleton)
        coords = list(zip(xs, ys))
        if len(coords) < 2:
            return None

        # Find endpoints
        endpoints = []
        for x, y in coords:
            neighbors = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                        if skeleton[ny, nx]:
                            neighbors += 1
            if neighbors == 1:
                endpoints.append((x, y))

        if len(endpoints) < 2:
            endpoints = [coords[0], coords[-1]]

        start, end = endpoints[0], endpoints[-1]
        path = TrackCenterline._bfs_skeleton_path(skeleton, start, end)

        if path is None or len(path) < 2:
            path = sorted(coords, key=lambda p: (p[1], p[0]))

        interval = float(getattr(self, "_keypoint_interval", 5))
        sampled = [path[0]]
        acc_dist = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            acc_dist += (dx * dx + dy * dy) ** 0.5
            if acc_dist >= interval:
                sampled.append(path[i])
                acc_dist = 0.0
        if sampled[-1] != path[-1]:
            sampled.append(path[-1])
        return [(int(p[0]), int(p[1])) for p in sampled]

    @staticmethod
    def _skeletonize(mask: np.ndarray) -> np.ndarray:
        binary = (mask > 0).astype(np.uint8)
        skeleton = np.zeros_like(binary)
        skel = binary.copy()
        while True:
            eroded = cv2.erode(skel, np.ones((3, 3), np.uint8))
            temp = cv2.dilate(eroded, np.ones((3, 3), np.uint8))
            temp = cv2.subtract(skel, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            skel = eroded.copy()
            if cv2.countNonZero(skel) == 0:
                break
        return skeleton

    @staticmethod
    def _bfs_skeleton_path(skeleton: np.ndarray, start: tuple, end: tuple):
        H, W = skeleton.shape
        visited = {start}
        queue = deque()
        queue.append((start, [start]))
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
        while queue:
            (x, y), path = queue.popleft()
            if (x, y) == end:
                return path
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    if skeleton[ny, nx] > 0 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append(((nx, ny), path + [(nx, ny)]))
        return None

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None

