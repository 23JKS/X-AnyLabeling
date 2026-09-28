"""Plate panel widget for displaying per-plate visualizations below the main canvas.

After track_mask2former inference, this panel shows each detected plate as a
cropped thumbnail with its own tracks, sources, and background strip rendered.
Tracks are renumbered per-plate starting from 1.
"""

import os
import tempfile
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

# Debug logs go to a writable per-machine temp folder: no hard-coded paths,
# so the app can run on any computer without a fixed user directory.
_DEBUG_DIR = os.path.join(tempfile.gettempdir(), "xanylabeling_debug")


def _dbg(msg):
    """Append one debug line. Never raises — logging must not break the app."""
    try:
        os.makedirs(_DEBUG_DIR, exist_ok=True)
        with open(os.path.join(_DEBUG_DIR, "plate_panel.log"), "a",
                  encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass

from anylabeling.views.labeling.utils.colormap import label_colormap
from anylabeling.services.auto_labeling.track_mask2former import (
    centerline_to_ribbon, _offset_centerline, _extend_centerline_y)

# Dynamic color palette: each unique label gets its own color (like the main canvas)
_COLORMAP = label_colormap()


def _color_for_label(label, palette_offset=0):
    """Assign a distinct color to each unique label, cycling through the palette.

    Computes a weak hash of the label string to get a stable palette index,
    then offset by palette_offset to allow different panels to use different
    starting colors (avoids all tracks being the same green).
    """
    h = abs(hash(label)) % len(_COLORMAP)
    idx = (palette_offset + h) % len(_COLORMAP)
    r, g, b = _COLORMAP[idx].tolist()
    return QtGui.QColor(r, g, b, 200)

_PLATE_LABEL_FONT = None


def _get_label_font():
    global _PLATE_LABEL_FONT
    if _PLATE_LABEL_FONT is None:
        _PLATE_LABEL_FONT = QtGui.QFont("Segoe UI", 10)
        _PLATE_LABEL_FONT.setBold(True)
    return _PLATE_LABEL_FONT


class PlateThumbnail(QtWidgets.QWidget):
    """A single plate crop rendered with its annotated shapes."""

    def __init__(self, plate_id, crop_rgb, shapes, bbox=None, parent=None):
        super().__init__(parent)
        self.plate_id = plate_id
        self.crop = crop_rgb  # numpy (H, W, 3) uint8
        self.shapes = shapes  # list of dicts: {label, points, closed}
        self._bbox = bbox     # [x, y, w, h] in original image coords
        self._margin = 12
        self._label_height = 28
        self._frame_radius = 8
        self._set_size()

    def _set_size(self):
        h, w = self.crop.shape[:2]
        self._img_w = w
        self._img_h = h
        self.setFixedSize(
            w + self._margin * 2 + 4,   # +4 for frame padding
            h + self._margin * 2 + self._label_height + 4,
        )

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        w = rect.width()
        h = rect.height()
        img_w = self._img_w
        img_h = self._img_h
        m = self._margin
        lh = self._label_height
        r = self._frame_radius

        # Card frame background
        frame_rect = QtCore.QRectF(1, 1, w - 2, h - 2)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(45, 45, 48))
        p.drawRoundedRect(frame_rect, r, r)

        # Card border
        pen = QtGui.QPen(QtGui.QColor(80, 80, 85), 1.5)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(frame_rect, r, r)

        # Plate label — pill-style background
        label_text = f"Plate {self.plate_id}"
        p.setFont(_get_label_font())
        fm = QtGui.QFontMetrics(_get_label_font())
        tw = fm.horizontalAdvance(label_text) + 16
        label_rect = QtCore.QRectF(
            m + 2, 6, tw, lh - 4,
        )
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(0, 122, 204, 180))
        p.drawRoundedRect(label_rect, 6, 6)
        p.setPen(QtGui.QColor(255, 255, 255))
        p.drawText(label_rect, QtCore.Qt.AlignmentFlag.AlignCenter, label_text)

        # Cropped image with subtle inner border
        img_x = m + 2
        img_y = lh + m - 4
        qimg = QtGui.QImage(
            self.crop.data, img_w, img_h,
            img_w * 3, QtGui.QImage.Format.Format_RGB888,
        )
        p.drawImage(img_x, img_y, qimg)

        # Inner border around crop
        crop_border = QtCore.QRectF(img_x - 1, img_y - 1, img_w + 2, img_h + 2)
        p.setPen(QtGui.QPen(QtGui.QColor(60, 60, 65), 1))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawRect(crop_border)

        # Translate to crop origin for shape drawing
        p.save()
        p.translate(img_x, img_y)

        # Draw shapes
        for i, s in enumerate(self.shapes):
            pts = s["points"]
            if len(pts) < 1:
                continue
            label = s.get("label", "")
            color = _color_for_label(label)

            if label.startswith("source"):
                cx = sum(x for x, _ in pts) / len(pts)
                cy = sum(y for _, y in pts) / len(pts)
                radius = max(3, 4)
                p.setPen(QtGui.QPen(color, 2))
                p.setBrush(QtGui.QColor(
                    color.red(), color.green(), color.blue(), 180))
                p.drawEllipse(QtCore.QPointF(cx, cy), radius, radius)
                p.setPen(QtGui.QColor(255, 255, 255))
                font = QtGui.QFont("Segoe UI", 9)
                font.setBold(True)
                p.setFont(font)
                fm2 = QtGui.QFontMetrics(font)
                text = s.get("display_label", label)
                tw2 = fm2.horizontalAdvance(text)
                p.drawText(
                    QtCore.QRectF(cx - tw2 / 2 - 2, cy - 8, tw2 + 4, 16),
                    QtCore.Qt.AlignmentFlag.AlignCenter,
                    text,
                )
                continue

            path = QtGui.QPainterPath()
            path.moveTo(QtCore.QPointF(pts[0][0], pts[0][1]))
            for x, y in pts[1:]:
                path.lineTo(QtCore.QPointF(x, y))
            path.closeSubpath()

            # Outline
            pen2 = QtGui.QPen(color)
            pen2.setWidth(2)
            p.setPen(pen2)
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawPath(path)

            # Label text at centroid
            if len(pts) >= 3:
                cx = sum(x for x, _ in pts) / len(pts)
                cy = sum(y for _, y in pts) / len(pts)
                p.setPen(QtGui.QColor(255, 255, 255))
                font = QtGui.QFont("Segoe UI", 9)
                font.setBold(True)
                p.setFont(font)
                fm2 = QtGui.QFontMetrics(font)
                text = s.get("display_label", label)
                tw2 = fm2.horizontalAdvance(text)
                p.drawText(
                    QtCore.QRectF(cx - tw2 / 2 - 2, cy - 8, tw2 + 4, 16),
                    QtCore.Qt.AlignmentFlag.AlignCenter,
                    text,
                )

        p.restore()
        p.end()


class PlatePanel(QtWidgets.QWidget):
    """Horizontally scrollable panel of plate thumbnails."""

    visibility_toggled = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVisible(False)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.setMinimumHeight(80)
        self._plates = []  # list of dicts: {plate_id, crop_rgb, shapes}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Toggle bar
        self._toggle = QtWidgets.QCheckBox("Show Plate Annotations (hide main)")
        self._toggle.setChecked(True)
        self._toggle.toggled.connect(self.visibility_toggled.emit)
        self._toggle.setStyleSheet(
            "QCheckBox { color: #ccc; font-size: 11px; padding: 2px 6px; }"
        )
        self._toggle.setVisible(False)
        layout.addWidget(self._toggle, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)

        # Scroll area for thumbnails
        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        self._scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._scroll.setMaximumHeight(200)
        self._scroll.setStyleSheet(
            "QScrollArea { border: 1px solid #3a3a3a; background: #1e1e1e; }"
        )

        self._container = QtWidgets.QWidget()
        self._container_layout = QtWidgets.QHBoxLayout(self._container)
        self._container_layout.setContentsMargins(6, 6, 6, 6)
        self._container_layout.setSpacing(10)
        self._container_layout.addStretch()
        self._scroll.setWidget(self._container)

        layout.addWidget(self._scroll)

    def showEvent(self, event):
        """Force parent layout to re-allocate space."""
        super().showEvent(event)
        p = self.parentWidget()
        if p and p.layout():
            p.layout().invalidate()
            p.layout().activate()

    def set_plates(self, plates_data, image_path=None):
        """Populate the panel with plate thumbnails.

        Args:
            plates_data: list of dicts, each with:
                - plate_id: int
                - crop_rgb: numpy (H, W, 3) uint8 RGB image of plate
                - shapes: list of dicts, each with:
                    - label: str (e.g. "track_1", "source_1", "background")
                    - display_label: str (what to show on the thumbnail)
                    - points: list of [x, y] in crop coordinates
        """
        # Clear existing thumbnails
        while self._container_layout.count() > 0:
            item = self._container_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        self._plates = plates_data

        if not plates_data:
            self.setVisible(False)
            return

        for pd in plates_data:
            thumb = PlateThumbnail(
                pd["plate_id"],
                pd["crop_rgb"],
                pd["shapes"],
                bbox=pd.get("bbox"),
            )
            self._container_layout.addWidget(thumb)

        self._container_layout.addStretch()

        # Auto-resize scroll area height based on tallest thumbnail
        max_h = 0
        for pd in plates_data:
            h = pd["crop_rgb"].shape[0]
            max_h = max(max_h, h)
        self._scroll.setMaximumHeight(
            min(max_h + 70, 350)
        )

        self._toggle.setVisible(True)
        self.show()
        self.adjustSize()
        self.updateGeometry()

    def clear(self):
        """Hide panel and clear all thumbnails."""
        while self._container_layout.count() > 0:
            item = self._container_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._plates = []
        self._toggle.setVisible(False)
        self.setVisible(False)

    def is_checked(self):
        return self._toggle.isChecked()


class _ZoomableScrollArea(QtWidgets.QScrollArea):
    """Scroll area that supports Ctrl+wheel zoom by calling a callback."""

    def __init__(self, on_zoom_in, on_zoom_out, parent=None):
        super().__init__(parent)
        self._on_zoom_in = on_zoom_in
        self._on_zoom_out = on_zoom_out

    def wheelEvent(self, event):
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self._on_zoom_in()
            elif delta < 0:
                self._on_zoom_out()
            return
        super().wheelEvent(event)


class PlateDialog(QtWidgets.QDialog):
    """Modeless popup that shows plates one-at-a-time with prev/next navigation
    and a label visibility panel on the right."""

    def __init__(self, plates_data, parent=None, image_path=None):
        super().__init__(parent)
        self._image_path = image_path
        self.setWindowTitle("Plate View")
        self.setMinimumSize(400, 300)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

        self._plates_data = plates_data
        self._index = 0
        self._zoom = 1.0  # Ctrl+wheel zoom factor
        self._current_pixmap = None
        self._visible_labels = {}  # label -> bool, populated per plate
        self._label_items = {}     # label -> QListWidgetItem, for checkbox signals

        # --- Load persisted settings ---------------------------------------
        # Defaults come from the packaged model config; the user's own choices
        # (table paths etc.) are stored per machine in the user profile, so
        # they are remembered on the next launch and also work when the
        # program folder itself is not writable (e.g. Program Files).
        import yaml
        import importlib.resources as pkg_resources
        import anylabeling.configs.auto_labeling as auto_labeling_configs
        self._cfg_path = os.path.join(
            os.path.expanduser("~"),
            ".xanylabeling",
            "track_mask2former_settings.yaml",
        )
        _cfg = {}
        try:
            with open(
                pkg_resources.files(auto_labeling_configs)
                / "track_mask2former.yaml",
                "r",
                encoding="utf-8",
            ) as f:
                _cfg = yaml.safe_load(f) or {}
        except Exception:
            pass
        try:
            with open(self._cfg_path, "r", encoding="utf-8") as f:
                _cfg.update(yaml.safe_load(f) or {})
        except Exception:
            pass
        self._r_to_e_path_default = _cfg.get("r_to_e_path", "")
        self._al_range_energy_path_default = _cfg.get("al_range_energy_path", "")
        self._r_tp_cm_default = float(_cfg.get("r_tp_cm", 90.0))
        self._al_sheet_default = _cfg.get("al_range_energy_sheet", "Al")

        # --- Styles ---
        self.setStyleSheet("""
            QDialog { background: #1e1e1e; }
            QPushButton { background: #3a3a3a; color: #ddd; padding: 8px 16px;
                          border: 1px solid #555; border-radius: 4px; font-size: 13px; }
            QPushButton:hover { background: #4a4a4a; }
            QPushButton:disabled { background: #2a2a2a; color: #666; }
            QLabel#plate_title { color: #ddd; font-size: 16px; font-weight: bold; }
            QLabel#plate_index { color: #aaa; font-size: 13px; }
            QListWidget { background: #2d2d30; border: 1px solid #3a3a3a; color: #ddd; }
            QListWidget::item { padding: 4px 8px; border-bottom: 1px solid #3a3a3a; }
            QListWidget::item:hover { background: #3a3a3f; }
        """)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        # === Title bar ===
        header = QtWidgets.QHBoxLayout()
        self._title_label = QtWidgets.QLabel(f"Plate {self._index + 1}")
        self._title_label.setObjectName("plate_title")
        header.addWidget(self._title_label)
        header.addStretch()
        self._index_label = QtWidgets.QLabel()
        self._index_label.setObjectName("plate_index")
        header.addWidget(self._index_label)
        root.addLayout(header)

        # === Body: horizontal split (image left, labels right) ===
        self._splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # --- Image display (zoomable via Ctrl+wheel) ---
        scroll = _ZoomableScrollArea(self._on_zoom_in, self._on_zoom_out)
        scroll.setWidgetResizable(False)  # allow zoomed image to be larger than viewport
        scroll.setStyleSheet("QScrollArea { border: 1px solid #3a3a3a; background: #252526; }")

        self._image_label = QtWidgets.QLabel()
        self._image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        scroll.setWidget(self._image_label)
        self._splitter.addWidget(scroll)

        # --- Label visibility panel ---
        label_panel = QtWidgets.QWidget()
        label_layout = QtWidgets.QVBoxLayout(label_panel)
        label_layout.setContentsMargins(0, 0, 0, 0)
        label_layout.setSpacing(4)

        label_header = QtWidgets.QLabel("Labels")
        label_header.setStyleSheet("color: #aaa; font-size: 12px; font-weight: bold; padding: 2px 6px;")
        label_layout.addWidget(label_header)

        # Toggle all checkbox
        self._toggle_all_cb = QtWidgets.QCheckBox("All")
        self._toggle_all_cb.setChecked(True)
        self._toggle_all_cb.setStyleSheet("color: #ccc; font-size: 11px; padding: 2px 6px;")
        self._toggle_all_cb.toggled.connect(self._on_toggle_all)
        label_layout.addWidget(self._toggle_all_cb)

        self._label_list = QtWidgets.QListWidget()
        self._label_list.setMaximumWidth(160)
        self._label_list.setMinimumWidth(130)
        self._label_list.itemChanged.connect(self._on_label_toggled)
        label_layout.addWidget(self._label_list, stretch=1)

        # --- Spectrum controls (below label list) ---
        spec_sep = QtWidgets.QFrame()
        spec_sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        spec_sep.setStyleSheet("color: #3a3a3a;")
        label_layout.addWidget(spec_sep)

        _STYLE = (
            "background: #2d2d30; color: #ddd; border: 1px solid #3a3a3a;"
            " padding: 2px 4px; font-size: 11px;"
        )
        _LBL = "color: #fff; font-size: 11px;"

        def _row(label_text, widget, extra=None):
            """Helper: label | widget [ | extra] on one row."""
            row = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel(label_text)
            lbl.setStyleSheet(_LBL)
            lbl.setFixedWidth(50)
            row.addWidget(lbl)
            row.addWidget(widget, stretch=1)
            if extra is not None:
                row.addWidget(extra)
            label_layout.addLayout(row)

        # t_shot
        self._t_shot_edit = QtWidgets.QDateTimeEdit(QtCore.QDateTime.currentDateTime())
        self._t_shot_edit.setDisplayFormat("yyyy-MM-dd hh:mm")
        self._t_shot_edit.setCalendarPopup(True)
        self._t_shot_edit.setToolTip("质子发射时间")
        self._t_shot_edit.setStyleSheet("QDateTimeEdit { " + _STYLE + " }")
        _row("t_shot:", self._t_shot_edit)

        # t_scan
        self._t_scan_edit = QtWidgets.QDateTimeEdit(QtCore.QDateTime.currentDateTime())
        self._t_scan_edit.setDisplayFormat("yyyy-MM-dd hh:mm")
        self._t_scan_edit.setCalendarPopup(True)
        self._t_scan_edit.setToolTip("IP扫描时间")
        self._t_scan_edit.setStyleSheet("QDateTimeEdit { " + _STYLE + " }")
        _row("t_scan:", self._t_scan_edit)

        # r_tp
        self._r_tp_edit = QtWidgets.QDoubleSpinBox()
        self._r_tp_edit.setRange(0.1, 500.0)
        self._r_tp_edit.setDecimals(2)
        self._r_tp_edit.setValue(self._r_tp_cm_default)
        self._r_tp_edit.setToolTip("靶点到IP板的距离 (cm)")
        self._r_tp_edit.setStyleSheet("QDoubleSpinBox { " + _STYLE + " }")
        self._r_tp_edit.valueChanged.connect(
            lambda v: self._save_config_value("r_tp_cm", v))
        _row("r_tp:", self._r_tp_edit)

        # R→E table path
        self._r_to_e_path_edit = QtWidgets.QLineEdit()
        self._r_to_e_path_edit.setText(self._r_to_e_path_default)
        self._r_to_e_path_edit.setPlaceholderText("R(mm) E(MeV) 表路径")
        self._r_to_e_path_edit.setToolTip("R→E 查找表文件路径")
        self._r_to_e_path_edit.setStyleSheet("QLineEdit { " + _STYLE + " }")
        self._r_to_e_path_edit.editingFinished.connect(
            lambda: self._save_config_value(
                "r_to_e_path", self._r_to_e_path_edit.text()))
        r2e_browse = QtWidgets.QPushButton("…")
        r2e_browse.setFixedWidth(24)
        r2e_browse.setToolTip("浏览选择 R→E 表文件")
        r2e_browse.setStyleSheet(
            "QPushButton { background: #3a3a3a; color: #ddd; border: 1px solid #555;"
            " padding: 0px 4px; font-size: 11px; }")
        r2e_browse.clicked.connect(self._on_browse_r_to_e)
        _row("R→E:", self._r_to_e_path_edit, r2e_browse)

        # Al range-energy table path
        self._al_range_energy_path_edit = QtWidgets.QLineEdit()
        self._al_range_energy_path_edit.setText(self._al_range_energy_path_default)
        self._al_range_energy_path_edit.setPlaceholderText("Al range-energy 表路径")
        self._al_range_energy_path_edit.setToolTip("Al range-energy 表文件路径")
        self._al_range_energy_path_edit.setStyleSheet("QLineEdit { " + _STYLE + " }")
        self._al_range_energy_path_edit.editingFinished.connect(
            lambda: self._save_config_value(
                "al_range_energy_path", self._al_range_energy_path_edit.text()))
        al_browse = QtWidgets.QPushButton("…")
        al_browse.setFixedWidth(24)
        al_browse.setToolTip("浏览选择 Al range-energy 表文件")
        al_browse.setStyleSheet(
            "QPushButton { background: #3a3a3a; color: #ddd; border: 1px solid #555;"
            " padding: 0px 4px; font-size: 11px; }")
        al_browse.clicked.connect(self._on_browse_al_range_energy)
        _row("Al:", self._al_range_energy_path_edit, al_browse)

        # Sheet selector for Excel (Al / Cu)
        self._al_sheet_combo = QtWidgets.QComboBox()
        self._al_sheet_combo.addItems(["Al", "Cu"])
        idx = self._al_sheet_combo.findText(
            self._al_sheet_default, QtCore.Qt.MatchFlag.MatchFixedString)
        if idx >= 0:
            self._al_sheet_combo.setCurrentIndex(idx)
        self._al_sheet_combo.setToolTip("Excel 工作表名称")
        self._al_sheet_combo.setStyleSheet("QComboBox { " + _STYLE + " }")
        self._al_sheet_combo.currentTextChanged.connect(
            lambda v: self._save_config_value("al_range_energy_sheet", v))
        _row("Sheet:", self._al_sheet_combo)

        self._ylim_min_edit = QtWidgets.QLineEdit()
        self._ylim_min_edit.setText("5e8")
        self._ylim_min_edit.setStyleSheet("QLineEdit { " + _STYLE + " }")
        self._ylim_min_edit.setToolTip("Y 轴下限")
        _row("Y min:", self._ylim_min_edit)

        self._ylim_max_edit = QtWidgets.QLineEdit()
        self._ylim_max_edit.setText("1e13")
        self._ylim_max_edit.setStyleSheet("QLineEdit { " + _STYLE + " }")
        self._ylim_max_edit.setToolTip("Y 轴上限")
        _row("Y max:", self._ylim_max_edit)

        self._spectrum_btn = QtWidgets.QPushButton("计算能谱 (dN/dE/dΩ)")
        self._spectrum_btn.setStyleSheet("""
            QPushButton {
                background: #0e639c; color: white; padding: 6px 8px;
                border: 1px solid #1177bb; border-radius: 4px;
                font-size: 11px; font-weight: bold;
            }
            QPushButton:hover { background: #1177bb; }
            QPushButton:disabled { background: #3a3a3a; color: #666; }
        """)
        self._spectrum_btn.clicked.connect(self._on_compute_spectrum)
        label_layout.addWidget(self._spectrum_btn)

        self._splitter.addWidget(label_panel)
        self._splitter.setStretchFactor(0, 0)  # image side: no stretch (fixed)
        self._splitter.setStretchFactor(1, 1)  # label panel: fills remaining
        root.addWidget(self._splitter, stretch=1)

        # === Navigation + Close ===
        nav = QtWidgets.QHBoxLayout()

        self._prev_btn = QtWidgets.QPushButton("←  Previous")
        self._prev_btn.clicked.connect(self._on_prev)
        nav.addWidget(self._prev_btn)

        nav.addStretch()

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        nav.addWidget(close_btn)

        nav.addStretch()

        self._next_btn = QtWidgets.QPushButton("Next  →")
        self._next_btn.clicked.connect(self._on_next)
        nav.addWidget(self._next_btn)

        root.addLayout(nav)

        self._show_current()

    # ------------------------------------------------------------------
    def _on_zoom_in(self):
        """Ctrl+wheel up: zoom in by 15%."""
        self._zoom *= 1.15
        self._redraw()

    def _on_zoom_out(self):
        """Ctrl+wheel down: zoom out by 15%."""
        self._zoom /= 1.15
        self._zoom = max(self._zoom, 0.05)  # don't let it go to zero
        self._redraw()

    # ------------------------------------------------------------------
    def _on_toggle_all(self, checked):
        for label in self._visible_labels:
            self._visible_labels[label] = checked
        # Update list items without re-triggering signals
        self._label_list.blockSignals(True)
        for label, item in self._label_items.items():
            item.setCheckState(QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)
        self._label_list.blockSignals(False)
        self._redraw()

    def _on_label_toggled(self, item):
        label = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if label is None:
            return
        self._visible_labels[label] = (item.checkState() == QtCore.Qt.CheckState.Checked)
        # Update "All" checkbox state
        all_checked = all(self._visible_labels.values())
        none_checked = not any(self._visible_labels.values())
        self._toggle_all_cb.blockSignals(True)
        if all_checked:
            self._toggle_all_cb.setCheckState(QtCore.Qt.CheckState.Checked)
        elif none_checked:
            self._toggle_all_cb.setCheckState(QtCore.Qt.CheckState.Unchecked)
        else:
            self._toggle_all_cb.setCheckState(QtCore.Qt.CheckState.PartiallyChecked)
        self._toggle_all_cb.blockSignals(False)
        self._redraw()

    def _populate_label_list(self, pd):
        """Rebuild the label list for a given plate data."""
        self._label_list.blockSignals(True)
        self._label_list.clear()
        self._label_items.clear()
        self._visible_labels.clear()

        for s in pd["shapes"]:
            label = s.get("label", "")
            if label not in self._visible_labels:
                self._visible_labels[label] = True
                item = QtWidgets.QListWidgetItem()
                item.setData(QtCore.Qt.ItemDataRole.UserRole, label)
                color = _color_for_label(label)
                item.setCheckState(QtCore.Qt.CheckState.Checked)
                # Color swatch + text
                item.setText(s.get("display_label", label))
                item.setForeground(QtGui.QBrush(color))
                self._label_list.addItem(item)
                self._label_items[label] = item

        self._label_list.blockSignals(False)
        self._toggle_all_cb.blockSignals(True)
        self._toggle_all_cb.setChecked(True)
        self._toggle_all_cb.blockSignals(False)

    def _redraw(self):
        """Redraw the current pixmap scaled to fit a target viewport
        (TARGET_W × TARGET_H) multiplied by zoom, maintaining aspect ratio."""
        TARGET_W = 400
        TARGET_H = 600

        pd = self._plates_data[self._index]
        crop = pd["crop_rgb"]
        h_img, w_img = crop.shape[:2]

        qimg = QtGui.QImage(
            crop.data, w_img, h_img,
            w_img * 3, QtGui.QImage.Format.Format_RGB888,
        )

        # Base scale to fit target viewport, then apply zoom
        base_scale = min(TARGET_W / w_img, TARGET_H / h_img)
        scale = base_scale * self._zoom
        scale = max(scale, 0.01)
        dw = max(1, int(w_img * scale))
        dh = max(1, int(h_img * scale))

        pix = QtGui.QPixmap(dw, dh)
        pix.fill(QtGui.QColor("#252526"))

        painter = QtGui.QPainter(pix)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Draw scaled crop
        scaled_img = qimg.scaled(
            dw, dh,
            QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawImage(0, 0, scaled_img)

        # Shapes: scale coords to pixmap
        for s in pd["shapes"]:
            label = s.get("label", "")
            if not self._visible_labels.get(label, True):
                continue
            pts = s["points"]
            if len(pts) < 1:
                continue
            color = _color_for_label(label)

            if label.startswith("source"):
                cx = sum(x for x, _ in pts) / len(pts)
                cy = sum(y for _, y in pts) / len(pts)
                radius = max(3, int(4 * scale))
                painter.setPen(QtGui.QPen(color, max(1, int(2 * scale))))
                painter.setBrush(QtGui.QColor(
                    color.red(), color.green(), color.blue(), 180))
                painter.drawEllipse(
                    QtCore.QPointF(cx * scale, cy * scale), radius, radius)
                painter.setPen(QtGui.QColor(255, 255, 255))
                fs = max(8, int(9 * scale))
                font = QtGui.QFont("Segoe UI", fs)
                font.setBold(True)
                painter.setFont(font)
                fm = QtGui.QFontMetrics(font)
                text = s.get("display_label", label)
                tw = fm.horizontalAdvance(text)
                painter.drawText(
                    QtCore.QRectF(cx * scale - tw / 2 - 3,
                                  cy * scale - fm.height() / 2,
                                  tw + 6, fm.height()),
                    QtCore.Qt.AlignmentFlag.AlignCenter,
                    text,
                )
                continue

            path = QtGui.QPainterPath()
            path.moveTo(QtCore.QPointF(pts[0][0] * scale, pts[0][1] * scale))
            for x, y in pts[1:]:
                path.lineTo(QtCore.QPointF(x * scale, y * scale))
            path.closeSubpath()

            pen = QtGui.QPen(color)
            pen.setWidth(max(1, int(2 * scale)))
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawPath(path)

            if len(pts) >= 3:
                cx = sum(x for x, _ in pts) / len(pts) * scale
                cy = sum(y for _, y in pts) / len(pts) * scale
                painter.setPen(QtGui.QColor(255, 255, 255))
                fs = max(8, int(9 * scale))
                font = QtGui.QFont("Segoe UI", fs)
                font.setBold(True)
                painter.setFont(font)
                fm = QtGui.QFontMetrics(font)
                text = s.get("display_label", label)
                tw = fm.horizontalAdvance(text)
                painter.drawText(
                    QtCore.QRectF(cx - tw / 2 - 3, cy - fm.height() / 2, tw + 6, fm.height()),
                    QtCore.Qt.AlignmentFlag.AlignCenter,
                    text,
                )

        painter.end()
        self._image_label.setPixmap(pix)
        self._image_label.setFixedSize(dw, dh)

        # Force splitter handle to image right edge
        scroll_w = dw + 4  # +4 for scroll area frame
        self._splitter.setSizes([scroll_w, 160])

    # ------------------------------------------------------------------
    def _on_prev(self):
        if self._index > 0:
            self._index -= 1
            self._show_current()

    def _on_next(self):
        if self._index < len(self._plates_data) - 1:
            self._index += 1
            self._show_current()

    # ------------------------------------------------------------------
    def _show_current(self):
        pd = self._plates_data[self._index]
        self._title_label.setText(f"Plate {pd['plate_id']}")

        total = len(self._plates_data)
        self._index_label.setText(f"{self._index + 1} / {total}")
        self._prev_btn.setEnabled(self._index > 0)
        self._next_btn.setEnabled(self._index < total - 1)

        self._zoom = 1.0  # reset zoom when switching plates
        self._populate_label_list(pd)
        self._redraw()

        # Adapt dialog size to fit the scaled image
        pix = self._image_label.pixmap()
        if pix:
            dw = pix.width()
            dh = pix.height()
            # image + label panel (~200px) + root margins (~32px)
            self.resize(dw + 220, dh + 110)

    # ------------------------------------------------------------------
    def _save_config_value(self, key, value):
        """Persist a single key-value pair to the per-user settings file."""
        import yaml
        try:
            os.makedirs(os.path.dirname(self._cfg_path), exist_ok=True)
        except Exception:
            return  # nowhere to write: keep the in-memory value only
        try:
            with open(self._cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
        cfg[key] = value
        try:
            with open(self._cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, default_flow_style=False, allow_unicode=True)
        except Exception:
            pass  # silently ignore write failures

    def _on_browse_r_to_e(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择 R→E 表文件", "",
            "Text Files (*.txt *.dat *.csv);;All Files (*)")
        if path:
            self._r_to_e_path_edit.setText(path)
            self._save_config_value("r_to_e_path", path)

    def _on_browse_al_range_energy(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择 Al range-energy 表文件", "",
            "Text Files (*.txt *.dat *.csv *.xlsx);;All Files (*)")
        if path:
            self._al_range_energy_path_edit.setText(path)
            self._save_config_value("al_range_energy_path", path)

    # ------------------------------------------------------------------
    def _on_compute_spectrum(self):
        """Calculate dN/dE/dΩ for checked tracks and save CSV+PNG."""
        import cv2, re, yaml
        import importlib.resources as pkg_resources
        import anylabeling.configs.auto_labeling as auto_labeling_configs

        # 1. Choose save directory
        save_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择能谱图保存位置")
        if not save_dir:
            return

        # 2. Collect checked (visible) tracks
        checked = [lbl for lbl, vis in self._visible_labels.items()
                   if lbl.startswith("track_") and vis]
        if not checked:
            QtWidgets.QMessageBox.warning(self, "提示",
                "请在右侧至少勾选一个 track（可见 = 参与计算）。")
            return

        # 3. Load original IP image (use .tif in same dir, same dims as log image)
        if not self._image_path or not os.path.isfile(self._image_path):
            QtWidgets.QMessageBox.warning(self, "错误", "找不到原始图像文件。")
            return
        tif_path = os.path.splitext(self._image_path)[0] + ".tif"
        if not os.path.isfile(tif_path):
            QtWidgets.QMessageBox.warning(
                self, "错误", f"找不到同目录下的tif图像:\n{tif_path}")
            return
        img = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            QtWidgets.QMessageBox.warning(self, "错误", "图像解码失败。")
            return
        raw_ip_image = img.astype(np.float64)
        if raw_ip_image.ndim == 3:
            raw_ip_image = raw_ip_image.mean(axis=2)

        # Convert raw pixel values (QL) to PSL
        from anylabeling.services.auto_labeling.spectrum import pixel_to_psl
        ip_image = pixel_to_psl(raw_ip_image, G=img.dtype.itemsize * 8)

        # 4. Read times & r_tp → solid_angle
        t_shot = self._t_shot_edit.dateTime().toPyDateTime()
        t_scan = self._t_scan_edit.dateTime().toPyDateTime()
        r_tp_cm = self._r_tp_edit.value()                      # cm
        r_tp_m = r_tp_cm / 100.0                                # m
        pinhole_d_um = 250.0                                    # μm (MATLAB default)
        pinhole_area_m2 = np.pi * ((pinhole_d_um * 1e-6) / 2.0) ** 2
        solid_angle = pinhole_area_m2 / (r_tp_m ** 2)           # sr

        # 5. Load table paths from UI controls (persisted to YAML on change)
        r_to_e_func = None
        al_e_to_range = al_range_to_e = None
        rp = self._r_to_e_path_edit.text().strip()
        arp = self._al_range_energy_path_edit.text().strip()

        # Load other fixed physics params from YAML config
        al_thickness_cm = 0.0015
        al_density = 2.7
        ip_fading_exponent = -0.161
        ip_fading_ref_min = 30.0
        dE_MeV = -0.2
        E_bin_max = 60.0
        E_bin_min = 1.0
        try:
            cfg_path = pkg_resources.files(auto_labeling_configs) / "track_mask2former.yaml"
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            al_thickness_cm = float(cfg.get("al_thickness_cm", 0.0015))
            al_density = float(cfg.get("al_density_g_cm3", 2.7))
            ip_fading_exponent = float(cfg.get("ip_fading_exponent", -0.161))
            ip_fading_ref_min = float(cfg.get("ip_fading_ref_min", 30.0))
            dE_MeV = float(cfg.get("dE_MeV", -0.2))
            E_bin_max = float(cfg.get("E_bin_max", 60.0))
            E_bin_min = float(cfg.get("E_bin_min", 1.0))
        except Exception:
            pass

        from anylabeling.services.auto_labeling.spectrum import (
            load_r_to_e_table, load_al_range_energy_table)
        try:
            if rp and os.path.isfile(rp):
                r_to_e_func = load_r_to_e_table(rp)
            if arp and os.path.isfile(arp):
                sheet = self._al_sheet_combo.currentText()
                al_e_to_range, al_range_to_e = load_al_range_energy_table(
                    arp, sheet_name=sheet)
        except Exception as e:
            # A bad/missing table must not kill the app: report and stop.
            QtWidgets.QMessageBox.warning(
                self, "错误",
                f"能量表文件无法读取，请重新选择文件:\n{e}")
            return

        try:
            ylim_min = float(self._ylim_min_edit.text())
            ylim_max = float(self._ylim_max_edit.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, "提示", "Y 轴范围必须是数字，例如 5e8 或 1e13")
            return

        # 6. Plate bbox (shift crop-relative coords → global)
        pd = self._plates_data[self._index]
        bbox = pd.get("bbox", [0, 0, 0, 0])  # [x, y, w, h]
        offset_x, offset_y = int(bbox[0]), int(bbox[1])

        shape_map = {s["label"]: s for s in pd["shapes"]}
        bg_shape = shape_map.get("background")

        if bg_shape is None:
            QtWidgets.QMessageBox.warning(
                self, "提示",
                "图中没有背景条带，无法计算能谱。\n请先在标注界面点击「绘制背景」。")
            return

        # Helper: shift polygon points to global image coordinates
        def _global_pts(shape_dict):
            if shape_dict is None:
                return None
            pts = shape_dict.get("points", [])
            if not pts:
                return None
            return [[p[0] + offset_x, p[1] + offset_y] for p in pts]

        def _bg_band_from_centerline(cl, track_w, y_min, y_max):
            """Extend a background centerline and rebuild a track-width ribbon."""
            if not cl or len(cl) < 2:
                return None
            cl = [[float(p[0]), float(p[1])] for p in cl]
            if cl[0][1] > cl[-1][1]:
                cl = list(reversed(cl))
            cl = _extend_centerline_y(cl, y_min, y_max)
            ribbon = centerline_to_ribbon(cl, track_w) if len(cl) >= 2 else None
            if ribbon and len(ribbon) >= 3:
                return [[float(p[0]), float(p[1])] for p in ribbon]
            return None

        def _centerline_x_at_y(centerline, y):
            cl = [(float(p[0]), float(p[1])) for p in centerline]
            if cl[0][1] > cl[-1][1]:
                cl = list(reversed(cl))
            if len(cl) < 2:
                return cl[0][0] if cl else 0.0
            if y <= cl[0][1]:
                return cl[0][0]
            if y >= cl[-1][1]:
                return cl[-1][0]
            for i in range(len(cl) - 1):
                x1, y1 = cl[i]
                x2, y2 = cl[i + 1]
                if y1 <= y <= y2:
                    if abs(y2 - y1) < 1e-12:
                        return x1
                    t = (y - y1) / (y2 - y1)
                    return x1 + t * (x2 - x1)
            return cl[-1][0]

        def _dump_aligned_bg_pixels(label, track_poly, bg_cl, raw_img):
            log_dir = _DEBUG_DIR
            os.makedirs(log_dir, exist_ok=True)
            out_path = os.path.join(log_dir, f"{label}_background_pixels.txt")
            h, w = raw_img.shape
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(track_poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 1)
            with open(out_path, "w", encoding="utf-8") as f:
                rows = np.where(mask.sum(axis=1) > 0)[0]
                for y in rows:
                    track_cols = np.where(mask[y] > 0)[0]
                    bg_x = _centerline_x_at_y(bg_cl, y)
                    width = len(track_cols)
                    start = int(round(bg_x - width / 2.0))
                    if start < 0:
                        start = 0
                    if start + width > w:
                        start = max(0, w - width)
                    cols = list(range(start, start + width))
                    vals = "，".join(str(int(raw_img[y, x])) for x in cols)
                    f.write(f"{int(y)}【{vals}】\n")

        def _dump_polygon_pixels(label, tag, poly, raw_img):
            log_dir = _DEBUG_DIR
            os.makedirs(log_dir, exist_ok=True)
            out_path = os.path.join(log_dir, f"{label}_{tag}_pixels.txt")
            h, w = raw_img.shape
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 1)
            with open(out_path, "w", encoding="utf-8") as f:
                rows = np.where(mask.sum(axis=1) > 0)[0]
                for y in rows:
                    xs = np.where(mask[y] > 0)[0]
                    vals = "，".join(str(int(raw_img[y, x])) for x in xs)
                    f.write(f"{int(y)}【{vals}】\n")

        # --- Per-track bg polygon helper ---
        # Background centerline = reference track centerline offset by
        # (bg_gap + current_track_hw).  Offset varies per track because
        # each track has its own width.

        def _make_bg_for_track(ref_cl, bg_gap, bg_side, track_w,
                               track_y_min, track_y_max):
            """Generate a background band polygon by offsetting the
            reference track's centerline.

            offset = bg_side * (bg_gap + track_w / 2)

            Args:
                ref_cl: reference track centerline [(x, y), ...]
                bg_gap: safety gap in pixels
                bg_side: +1 (left) or -1 (right)
                track_w: current track width in pixels
                track_y_min, track_y_max: Y range of current track

            Returns list of (x, y) in crop-relative coords, or None.
            """
            if not ref_cl or len(ref_cl) < 2:
                _dbg(f"[_make_bg] FAIL: ref_cl is None or len={len(ref_cl) if ref_cl else 0}")
                return None

            # Offset ref centerline by (bg_gap + track_w) so that
            # gap between track band edge and bg band edge = bg_gap
            offset = bg_side * (bg_gap + track_w)
            _dbg(f"[_make_bg] offset={offset:.1f}  bg_gap={bg_gap}  bg_side={bg_side}  track_w={track_w}")
            cl = _offset_centerline(ref_cl, offset)
            if not cl or len(cl) < 2:
                _dbg(f"[_make_bg] FAIL: _offset_centerline returned None or len={len(cl) if cl else 0}")
                return None

            # Extend to cover track's Y range
            _dbg(f"[_make_bg] cl Y before extend: [{min(p[1] for p in cl):.1f}, {max(p[1] for p in cl):.1f}]  target=[{track_y_min:.1f},{track_y_max:.1f}]")
            cl = _extend_centerline_y(cl, track_y_min, track_y_max)
            _dbg(f"[_make_bg] cl Y after extend: [{min(p[1] for p in cl):.1f}, {max(p[1] for p in cl):.1f}]  len={len(cl)}")

            ribbon = centerline_to_ribbon(cl, track_w) if len(cl) >= 2 else None
            if ribbon:
                rx = [p[0] for p in ribbon]
                ry = [p[1] for p in ribbon]
                _dbg(f"[_make_bg] SUCCESS: ribbon X=[{min(rx):.1f},{max(rx):.1f}] Y=[{min(ry):.1f},{max(ry):.1f}]")
            else:
                _dbg(f"[_make_bg] FAIL: centerline_to_ribbon returned None (cl len={len(cl)})")
            return ribbon

        # Cache bg metadata once
        bg_od = bg_shape.get("other_data", {}) if bg_shape else {}
        ref_cl = bg_od.get("ref_centerline")
        # Use bg_od value or default, but force at least 15 for now
        _stored_gap = float(bg_od.get("bg_gap", 15.0))
        bg_gap = max(_stored_gap, 15.0)
        bg_side = bg_od.get("bg_side", 1.0)
        _dbg(f"[bg_shape] label='{bg_shape.get('label','?')}'  points={bg_shape.get('points','N/A')}")
        _dbg(f"[ref_cl]  type={type(ref_cl).__name__}  len={len(ref_cl) if ref_cl else 0}  first3={ref_cl[:3] if ref_cl else 'None'}")

        # 7. Compute per track
        from anylabeling.services.auto_labeling.spectrum import (
            compute_track_spectrum, save_spectrum_csv, save_spectrum_plot)

        # Find source centroid Y (origin for R→E dispersion).
        # Each plate must have EXACTLY ONE source; 0 or >1 sources
        # means we cannot determine the energy origin → refuse to compute.
        sources_in_plate = [
            s for s in pd["shapes"] if s["label"].startswith("source")
        ]
        if len(sources_in_plate) == 0:
            QtWidgets.QMessageBox.warning(
                self, "提示",
                "当前 plate 没有 source，无法确定能量原点，"
                "不能绘制能谱图。\n请先为该 plate 标注一个 source。")
            return
        if len(sources_in_plate) > 1:
            QtWidgets.QMessageBox.warning(
                self, "提示",
                f"当前 plate 有 {len(sources_in_plate)} 个 source，"
                "每个 plate 只能有一个 source，无法确定能量原点，"
                "不能绘制能谱图。\n请删除多余的 source 标注。")
            return

        _src = sources_in_plate[0]
        _src_pts = _src.get("global_points") or _src.get("points", [])
        if not _src_pts:
            QtWidgets.QMessageBox.warning(
                self, "提示",
                "当前 plate 的 source 没有坐标，无法计算能谱。")
            return
        source_y = float(np.mean([p[1] for p in _src_pts]))
        if not _src.get("global_points"):
            source_y += offset_y

        done, errs = 0, []
        for label in checked:
            ts = shape_map.get(label)
            if ts is None:
                errs.append(f"{label}: not found")
                continue
            try:
                _track_global = ts.get("global_points")
                if _track_global:
                    track_polygon = [
                        [float(p[0]), float(p[1])] for p in _track_global
                    ]
                else:
                    track_polygon = _global_pts(ts)
                if track_polygon is None or len(track_polygon) < 3:
                    errs.append(f"{label}: no polygon")
                    continue

                # Build bg polygon: shared bg centerline + track's width + track's Y range
                track_od = ts.get("other_data", {})
                track_tw = int(track_od.get("track_width", 14))
                ty_vals = [p[1] for p in track_polygon]
                track_y_min = min(ty_vals)
                track_y_max = max(ty_vals)
                bg_cl = None
                if (ref_cl is not None and len(ref_cl) >= 2
                        and track_polygon is not None):
                    # ref_cl is in global coords, so _make_bg_for_track
                    # returns a polygon already in global coords.
                    bg_cl = _offset_centerline(
                        ref_cl, bg_side * (bg_gap + track_tw))
                    bg_cl = _extend_centerline_y(
                        bg_cl, track_y_min, track_y_max)
                    _bg = _make_bg_for_track(ref_cl, bg_gap, bg_side,
                                             track_tw,
                                             track_y_min, track_y_max)
                    if _bg and len(_bg) >= 3:
                        my_bg_polygon = [[float(p[0]), float(p[1])] for p in _bg]
                    else:
                        bg_cl = (
                            bg_od.get("centerline")
                            or bg_shape.get("global_centerline")
                            or bg_shape.get("global_points")
                            or _global_pts(bg_shape)
                        )
                        my_bg_polygon = _bg_band_from_centerline(
                            bg_cl, track_tw, track_y_min, track_y_max)
                        if not my_bg_polygon:
                            _bg_global = bg_shape.get("global_points")
                            my_bg_polygon = (
                                [[float(p[0]), float(p[1])] for p in _bg_global]
                                if _bg_global else _global_pts(bg_shape)
                            )
                else:
                    bg_cl = (
                        bg_od.get("centerline")
                        or bg_shape.get("global_centerline")
                        or bg_shape.get("global_points")
                        or _global_pts(bg_shape)
                    )
                    my_bg_polygon = _bg_band_from_centerline(
                        bg_cl, track_tw, track_y_min, track_y_max)
                    if not my_bg_polygon:
                        _bg_global = bg_shape.get("global_points")
                        my_bg_polygon = (
                            [[float(p[0]), float(p[1])] for p in _bg_global]
                            if _bg_global else _global_pts(bg_shape)
                        )

                try:
                    _dump_polygon_pixels(
                        label, "track", track_polygon, raw_ip_image)
                    _dump_aligned_bg_pixels(
                        label, track_polygon, bg_cl, raw_ip_image)
                except Exception as _e:
                    _dbg(f"[dump] failed to write polygon pixels for {label}: {_e}")

                # Debug: compare track vs bg polygon widths globally
                tx = [p[0] for p in track_polygon]
                bx = [p[0] for p in my_bg_polygon]
                ty = [p[1] for p in track_polygon]
                by_ = [p[1] for p in my_bg_polygon]
                _dbg(f"[spectrum] {label}: track X [{min(tx):.1f}, {max(tx):.1f}] w={max(tx)-min(tx):.1f}  Y [{min(ty):.1f}, {max(ty):.1f}]")
                _dbg(f"[spectrum] {label}: bg    X [{min(bx):.1f}, {max(bx):.1f}] w={max(bx)-min(bx):.1f}  Y [{min(by_):.1f}, {max(by_):.1f}]")
                _dbg(f"[spectrum] {label}: (track_tw={track_tw})")

                r = compute_track_spectrum(
                    track_polygon, my_bg_polygon, ip_image,
                    bg_centerline=bg_cl,
                    r_to_e_func=r_to_e_func,
                    al_e_to_range=al_e_to_range,
                    al_range_to_e=al_range_to_e,
                    al_thickness_cm=al_thickness_cm,
                    al_density=al_density,
                    solid_angle=solid_angle,
                    t_shot=t_shot,
                    t_scan=t_scan,
                    ip_fading_exponent=ip_fading_exponent,
                    ip_fading_ref_min=ip_fading_ref_min,
                    dE_MeV=dE_MeV,
                    E_bin_max=E_bin_max,
                    E_bin_min=E_bin_min,
                    source_y=source_y)
                if r is None:
                    errs.append(f"{label}: None")
                    continue
                save_spectrum_csv(r, save_dir, label)
                save_spectrum_plot(
                    r, save_dir, label,
                    ylim_min=ylim_min,
                    ylim_max=ylim_max)

                # Also compute without bg subtraction for comparison
                r_nobg = compute_track_spectrum(
                    track_polygon, None, ip_image,
                    r_to_e_func=r_to_e_func,
                    al_e_to_range=al_e_to_range,
                    al_range_to_e=al_range_to_e,
                    al_thickness_cm=al_thickness_cm,
                    al_density=al_density,
                    solid_angle=solid_angle,
                    t_shot=t_shot,
                    t_scan=t_scan,
                    ip_fading_exponent=ip_fading_exponent,
                    ip_fading_ref_min=ip_fading_ref_min,
                    dE_MeV=dE_MeV,
                    E_bin_max=E_bin_max,
                    E_bin_min=E_bin_min,
                    source_y=source_y,
                    skip_bg=True)
                if r_nobg is not None:
                    save_spectrum_csv(r_nobg, save_dir, f"{label}_no_bg")
                    save_spectrum_plot(
                        r_nobg, save_dir, f"{label}_no_bg",
                        ylim_min=ylim_min,
                        ylim_max=ylim_max)

                done += 1
            except Exception as e:
                errs.append(f"{label}: {e}")

        if errs:
            msg = "\n".join(errs[:8])
            if len(errs) > 8:
                msg += f"\n… +{len(errs) - 8} more"
            QtWidgets.QMessageBox.warning(
                self, "部分失败", f"OK: {done}/{len(checked)}\n错误:\n{msg}")
        else:
            QtWidgets.QMessageBox.information(
                self, "完成", f"已保存 {done} 个track的能谱到:\n{save_dir}")
