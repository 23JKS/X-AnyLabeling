# band_manager.py - Track Centerline band polygon management
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor


class BandManager:
    def __init__(self, parent):
        self.parent = parent
        self._band_mode = False
        self._centerline_shapes = []
        self._band_shapes = []
        self._editing_track_shape = None

    def toggle(self, width):
        self._band_mode = not self._band_mode
        canvas = self.parent.canvas
        canvas.shapes = [s for s in canvas.shapes if not s.other_data.get("_is_band")]
        if self._band_mode:
            self._centerline_shapes = []
            self._band_shapes = []
            kept = []
            for s in canvas.shapes:
                if s.label.startswith("track") and s.shape_type in ("linestrip","line"):
                    self._centerline_shapes.append(s)
                    if len(s.points) >= 2:
                        cl = [(p.x(), p.y()) for p in s.points]
                        s.other_data["centerline"] = cl
                        w = s.other_data.get("track_width", width)
                        bs = self._make_band(cl, w, s.label, s.line_color, s.fill_color)
                        self._band_shapes.append(bs)
                else:
                    kept.append(s)
            canvas.shapes = kept + self._band_shapes
        else:
            existing = {id(s) for s in canvas.shapes}
            for cl_s in self._centerline_shapes:
                if id(cl_s) not in existing:
                    canvas.shapes.append(cl_s)
        canvas.update()
        return self._band_mode

    def _make_band(self, centerline, width, label, line_color, fill_color):
        from .auto_labeling import AutoLabelingWidget
        pts = AutoLabelingWidget._regenerate_band_polygon(None, centerline, width/2.0)
        bs = type(self.parent.canvas.shapes[0])(label=label, shape_type="polygon", flags={})
        bs.other_data["_is_band"] = True
        bs.other_data["centerline"] = centerline
        bs.other_data["track_width"] = width
        bs.line_color = line_color
        bs.fill_color = fill_color
        bs.fill = True
        for x, y in pts:
            bs.add_point(QPointF(x, y))
        return bs

    def on_shape_selected(self, shapes, spinbox):
        self._editing_track_shape = None
        if len(shapes) == 1:
            s = shapes[0]
            if s.other_data.get("_is_band") and s.other_data.get("centerline") and s.other_data.get("track_width"):
                self._editing_track_shape = s
                spinbox.blockSignals(True)
                spinbox.setValue(int(s.other_data["track_width"]))
                spinbox.blockSignals(False)

    def on_width_changed(self, value, canvas):
        if self._editing_track_shape is None:
            return
        cl = self._editing_track_shape.other_data.get("centerline")
        if cl:
            pts = self._make_band_points(cl, value/2.0)
            self._editing_track_shape.points = [QPointF(x, y) for x, y in pts]
            self._editing_track_shape.other_data["track_width"] = value
            # self._editing_track_shape.update()  # Shape has no update method
            canvas.update()

    @staticmethod
    def _make_band_points(centerline, half_width):
        if len(centerline) < 2:
            return centerline
        n = len(centerline)
        left, right = [], []
        for i in range(n):
            if i == 0:
                dx = centerline[1][0] - centerline[0][0]
                dy = centerline[1][1] - centerline[0][1]
            elif i == n-1:
                dx = centerline[n-1][0] - centerline[n-2][0]
                dy = centerline[n-1][1] - centerline[n-2][1]
            else:
                dx = centerline[i+1][0] - centerline[i-1][0]
                dy = centerline[i+1][1] - centerline[i-1][1]
            length = (dx*dx + dy*dy) ** 0.5
            if length < 1e-6:
                continue
            nx, ny = -dy/length, dx/length
            left.append((centerline[i][0] + nx*half_width, centerline[i][1] + ny*half_width))
            right.append((centerline[i][0] - nx*half_width, centerline[i][1] - ny*half_width))
        if len(left) < 2:
            return centerline
        return left + list(reversed(right))
print(1)