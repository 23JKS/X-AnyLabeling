"""Watershed-based particle track segmentation for X-AnyLabeling."""
import os
import traceback
import numpy as np
import cv2
from PIL import Image
from PyQt6 import QtCore
from PyQt6.QtCore import QCoreApplication
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from .model import Model
from .types import AutoLabelingResult

try:
    from scipy import ndimage
    from skimage.morphology import skeletonize
    from skimage import morphology, segmentation
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


class TrackWatershed(Model):
    class Meta:
        required_config_names = ["type", "name", "display_name"]
        widgets = [
            "button_run",
            "input_conf",
            "edit_conf",
            "toggle_preserve_existing_annotations",
        ]
        output_modes = {
            "polygon": QCoreApplication.translate("Model", "Polygon"),
        }
        default_output_mode = "polygon"

    def __init__(self, model_config, on_message) -> None:
        super().__init__(model_config, on_message)
        if not HAS_SCIPY:
            raise ImportError(
                "scipy and scikit-image required. "
                "Install with: pip install scipy scikit-image"
            )
        self.threshold_percentile = self.config.get(
            "threshold_percentile", 92.0
        )
        self.threshold_hi_percentile = self.config.get(
            "threshold_hi_percentile", 97.0
        )
        self.min_size = self.config.get("min_size", 500)
        self.dilation_radius = self.config.get("dilation_radius", 3)
        self.border_y_fraction = self.config.get("border_y_fraction", 0.75)
        self.replace = True

    def _msg(self, text):
        """Send status message through the model manager."""
        self.on_message(text)
        logger.info(text)

    def set_auto_labeling_conf(self, value):
        if value > 0:
            self.threshold_percentile = 80.0 + value * 15.0
            self._msg(
                f"Threshold set to {self.threshold_percentile:.2f}%"
            )

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        self.replace = not state

    def _load_image_array(self, image_path):
        ext = os.path.splitext(image_path)[1].lower()
        if ext in (".tif", ".tiff"):
            self._msg(f"Loading TIF: {os.path.basename(image_path)}")
            if HAS_TIFFFILE:
                img = tifffile.imread(image_path).astype(np.float32)
            else:
                img = np.array(Image.open(image_path)).astype(np.float32)
            if img.ndim == 3:
                img = img[0]
            log_img = np.log1p(img)
            vmin, vmax = np.percentile(log_img, [1, 99.9])
            img = np.clip(
                (log_img - vmin) / (vmax - vmin) * 255, 0, 255
            ).astype(np.uint8)
        else:
            img = np.array(Image.open(image_path))
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def predict_shapes(self, image, image_path=None, filename=None, **kwargs):
        fpath = image_path or filename
        if fpath is None:
            self._msg("ERROR: No image path")
            return AutoLabelingResult([], replace=self.replace)

        self._msg(f"Segmenting: {os.path.basename(fpath)}")

        try:
            img = self._load_image_array(fpath)
        except Exception as e:
            self._msg(f"ERROR loading image: {e}")
            logger.error(traceback.format_exc())
            return AutoLabelingResult([], replace=self.replace)

        h, w = img.shape[:2]
        self._msg(f"Image: {w}x{h}, type={img.dtype}, range=[{img.min()},{img.max()}]")

        img_f = img.astype(np.float32)

        # ---- Step 1: Threshold ----
        blur_ksize = max(3, int(self.dilation_radius * 2 + 1))
        blurred = cv2.GaussianBlur(img_f, (blur_ksize, blur_ksize), 0)
        p_lo = np.percentile(blurred, self.threshold_percentile)
        binary = (blurred > p_lo).astype(np.uint8)
        self._msg(f"Threshold p{self.threshold_percentile:.1f}={p_lo:.1f}, px={binary.sum()}")

        # ---- Step 2: Clean noise then close gaps ----
        from skimage import morphology as morph
        cleaned = morph.remove_small_objects(
            binary.astype(bool), min_size=self.min_size
        ).astype(np.uint8)

        close_radius = max(2, self.dilation_radius)
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_radius * 2 + 1, close_radius * 2 + 1)
        )
        closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)

        # ---- Step 3: Connected components ----
        labels, n_labels = ndimage.label(closed)
        self._msg(f"Found {n_labels} connected regions")

        # ---- Step 4: Skeletonize each region, then dilate to uniform band ----
        band_width = self.dilation_radius  # half-width of the band
        shapes = []

        for i in range(1, n_labels + 1):
            mask = (labels == i)
            area = mask.sum()
            if area < self.min_size:
                continue

            # Skeletonize to get centerline
            skel = skeletonize(mask)
            skel_px = skel.sum()

            if skel_px < 10:  # too small, skip
                continue

            # Dilate skeleton by band_width to create uniform band
            dilate_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (band_width * 2 + 1, band_width * 2 + 1)
            )
            band = cv2.dilate(skel.astype(np.uint8), dilate_kernel, iterations=1)

            # Find contour of the band -> smooth polygon
            contours, _ = cv2.findContours(
                band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            if len(contour) < 5:
                continue

            # Gentle simplification for smooth arcs
            epsilon = 0.0005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) < 5:
                continue

            # Classify
            ys = np.where(mask)[0]
            cls_name = "origin" if ys.mean() > h * self.border_y_fraction else "track"

            shape = Shape(label=cls_name, score=1.0, shape_type="polygon")
            for pt in approx:
                shape.add_point(QtCore.QPointF(float(pt[0][0]), float(pt[0][1])))
            shape.closed = True
            shapes.append(shape)

        n_tracks = sum(1 for s in shapes if s.label == "track")
        n_origins = sum(1 for s in shapes if s.label == "origin")
        self._msg(f"Done: {len(shapes)} arc bands ({n_tracks} tracks, {n_origins} origins)")
        return AutoLabelingResult(shapes, replace=self.replace)

    def unload(self):
        pass
