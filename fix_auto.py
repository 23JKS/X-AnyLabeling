p = r"D:\desk\X-AnyLabeling\anylabeling\views\labeling\widgets\auto_labeling\auto_labeling.py"
with open(p, "r", encoding="utf-8") as f:
    c = f.read()

# Remove the programmatic widget creation block (including layout insertion)
old_block = """
        # --- Configuration for: edit_min_track_length ---
        self.input_min_track_length = QLabel("Min Track Length", self)
        self.input_min_track_length.setStyleSheet("font-size: 12px;")
        self.edit_min_track_length = QSpinBox()
        self.edit_min_track_length.setStyleSheet(get_double_spinbox_style())
        self.edit_min_track_length.setMinimum(5)
        self.edit_min_track_length.setMaximum(500)
        self.edit_min_track_length.setSingleStep(5)
        self.edit_min_track_length.setValue(30)
        self.edit_min_track_length.valueChanged.connect(self.on_min_track_length_changed)
        self.edit_min_track_length.setToolTip(
            self.tr("Minimum polyline length in pixels. Shorter tracks are filtered out.")
        )
        # Find the vertical layout that contains edit_iou
        parent_layout = None
        widget = self.edit_iou
        while widget is not None and widget is not self:
            wlayout = widget.parent().layout() if widget.parent() else None
            if wlayout and wlayout.indexOf(widget) >= 0:
                parent_layout = wlayout
                break
            widget = widget.parent()
        # Fallback: use main layout
        if parent_layout is None:
            parent_layout = self.layout()
        if parent_layout:
            iou_idx = parent_layout.indexOf(self.edit_iou)
            if iou_idx >= 0:
                parent_layout.insertWidget(iou_idx + 1, self.input_min_track_length)
                parent_layout.insertWidget(iou_idx + 2, self.edit_min_track_length)
"""

# Replace with just the signal connection
new_block = """
        # --- Configuration for: edit_min_track_length ---
        self.edit_min_track_length.valueChanged.connect(self.on_min_track_length_changed)
"""

c = c.replace(old_block, new_block, 1)

with open(p, "w", encoding="utf-8") as f:
    f.write(c)
print("OK")
