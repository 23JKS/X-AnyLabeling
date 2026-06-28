p = r"D:\desk\X-AnyLabeling\anylabeling\services\auto_labeling\track_centerline.py"
c = open(p, encoding="utf-8").read()

# 1. Add _polyline_to_band_polygon method before _polyline_length
old = """    @staticmethod
    def _polyline_length(points: list) -> float:"""
new = """    @staticmethod
    def _polyline_to_band_polygon(points, half_width):
        """Expand a polyline into a band polygon by width on both sides."""
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
    def _polyline_length(points: list) -> float:"""
c = c.replace(old, new, 1)

# 2. Modify shapes generation to create band polygons
old2 = """            if polyline_points is not None and len(polyline_points) >= 2:
                shape = Shape(
                    label=\"track\",
                    shape_type=\"linestrip\",
                    flags={},
                )
                for pt in polyline_points:
                    shape.add_point(QtCore.QPointF(pt[0], pt[1]))
                shapes.append(shape)"""

new2 = """            if polyline_points is not None and len(polyline_points) >= 2:
                half_w = getattr(self, \"_track_width\", 15) / 2.0
                band_points = self._polyline_to_band_polygon(polyline_points, half_w)
                shape = Shape(
                    label=\"track\",
                    shape_type=\"polygon\",
                    flags={},
                )
                # Store centerline and width for future editing
                shape.other_data[\"centerline\"] = polyline_points
                shape.other_data[\"track_width\"] = half_w * 2
                for pt in band_points:
                    shape.add_point(QtCore.QPointF(pt[0], pt[1]))
                shapes.append(shape)"""

c = c.replace(old2, new2, 1)

# 3. Add setter for track_width
old3 = "    def set_auto_labeling_min_track_length(self, value):\n        self._min_track_length = int(value)"
new3 = """    def set_auto_labeling_min_track_length(self, value):
        self._min_track_length = int(value)

    def set_auto_labeling_track_width(self, value):
        self._track_width = int(value)"""

c = c.replace(old3, new3, 1)

open(p, "w", encoding="utf-8").write(c)
print("OK")
