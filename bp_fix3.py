p = r'D:\desk\X-AnyLabeling\anylabeling\views\labeling\widgets\auto_labeling\auto_labeling.py'
c = open(p, encoding='utf-8').read()
old = '''        for shape in shapes:
            if shape.label != \"track\":
                continue
            cl = shape.other_data.get(\"centerline\")
            if not cl:
                continue

            if self._band_mode:
                # Expand to polygon
                half_w = width / 2.0
                band_pts = self._regenerate_band_polygon(cl, half_w)
                shape.shape_type = \"polygon\"
                shape.points = [QPointF(x, y) for x, y in band_pts]
                shape.other_data[\"track_width\"] = width
            else:
                # Restore centerline
                shape.shape_type = \"linestrip\"
                shape.points = [QPointF(x, y) for x, y in cl]
                shape.other_data[\"track_width\"] = width'''
new = '''        for shape in shapes:
            if shape.label != \"track\":
                continue
            cl = shape.other_data.get(\"centerline\")
            if not cl:
                continue

            if self._band_mode:
                # Use existing width if already set, otherwise use slider value
                w = shape.other_data.get(\"track_width\", width)
                half_w = w / 2.0
                band_pts = self._regenerate_band_polygon(cl, half_w)
                shape.shape_type = \"polygon\"
                shape.points = [QPointF(x, y) for x, y in band_pts]
                shape.other_data[\"track_width\"] = w
            else:
                # Restore centerline, keep width
                shape.shape_type = \"linestrip\"
                shape.points = [QPointF(x, y) for x, y in cl]
                # Keep track_width as-is for next expansion'''
c = c.replace(old, new, 1)
open(p, 'w', encoding='utf-8').write(c)
print('OK')
