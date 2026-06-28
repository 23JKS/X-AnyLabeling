p = r'D:\desk\X-AnyLabeling\anylabeling\views\labeling\widgets\auto_labeling\auto_labeling.py'
c = open(p, encoding='utf-8').read()

# 1. Connect selection_changed signal - find a good spot after __init__
old = '        self.skip_auto_prediction = False'
new = '''        self.skip_auto_prediction = False
        self._editing_track_shape = None  # currently selected track polygon

        # Connect shape selection to track width editing
        if hasattr(self.parent, \"canvas\") and hasattr(self.parent.canvas, \"selection_changed\"):
            self.parent.canvas.selection_changed.connect(self._on_shape_selected)'''
c = c.replace(old, new, 1)

# 2. Modify on_track_width_changed to support per-shape editing
old_handler = '''    def on_track_width_changed(self, value):
        self.model_manager.set_auto_labeling_track_width(value)'''
new_handler = '''    def on_track_width_changed(self, value):
        self.model_manager.set_auto_labeling_track_width(value)
        # If a track polygon is selected, regenerate it with new width
        if self._editing_track_shape is not None:
            cl = self._editing_track_shape.other_data.get(\"centerline\")
            if cl:
                new_pts = self._regenerate_band_polygon(cl, value / 2.0)
                self._editing_track_shape.points = [QtCore.QPointF(x, y) for x, y in new_pts]
                self._editing_track_shape.other_data[\"track_width\"] = value
                self._editing_track_shape.update()
                self.parent.canvas.update()'''

c = c.replace(old_handler, new_handler, 1)

# 3. Add _on_shape_selected and _regenerate_band_polygon methods
old_method = '    def on_mask_fineness_changed(self, value):'
new_method = '''    def _regenerate_band_polygon(self, points, half_width):
        """Rebuild band polygon from centerline and new width."""
        if len(points) < 2:
            return points
        n = len(points)
        left_pts, right_pts = [], []
        for i in range(n):
            if i == 0:
                dx, dy = points[1][0] - points[0][0], points[1][1] - points[0][1]
            elif i == n - 1:
                dx, dy = points[n-1][0] - points[n-2][0], points[n-1][1] - points[n-2][1]
            else:
                dx, dy = points[i+1][0] - points[i-1][0], points[i+1][1] - points[i-1][1]
            length = (dx*dx + dy*dy) ** 0.5
            if length < 1e-6:
                continue
            nx, ny = -dy / length, dx / length
            left_pts.append((points[i][0] + nx * half_width, points[i][1] + ny * half_width))
            right_pts.append((points[i][0] - nx * half_width, points[i][1] - ny * half_width))
        if len(left_pts) < 2:
            return points
        return left_pts + list(reversed(right_pts))

    def _on_shape_selected(self, selected_shapes):
        """When a shape is clicked, check if it is a track polygon with adjustable width."""
        self._editing_track_shape = None
        if len(selected_shapes) == 1:
            shape = selected_shapes[0]
            if shape.other_data.get(\"centerline\") and shape.other_data.get(\"track_width\"):
                self._editing_track_shape = shape
                # Update slider to show this shape is width (block signals to avoid loop)
                self.edit_track_width.blockSignals(True)
                self.edit_track_width.setValue(int(shape.other_data[\"track_width\"]))
                self.edit_track_width.blockSignals(False)

    def on_mask_fineness_changed(self, value):'''
c = c.replace(old_method, new_method, 1)

open(p, 'w', encoding='utf-8').write(c)
print('OK')
