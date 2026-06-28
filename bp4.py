p = r'D:\desk\X-AnyLabeling\anylabeling\services\auto_labeling\track_centerline.py'
c = open(p, encoding='utf-8').read()
old = 'label=\"track\",\n                    shape_type=\"linestrip\",\n                    flags={},\n                )\n                for pt in polyline_points:\n                    shape.add_point(QtCore.QPointF(pt[0], pt[1]))\n                shapes.append(shape)'
new = 'label=\"track\",\n                    shape_type=\"polygon\",\n                    flags={},\n                )\n                half_w = getattr(self, \"_track_width\", 15) / 2.0\n                band_points = self._polyline_to_band_polygon(polyline_points, half_w)\n                shape.other_data[\"centerline\"] = polyline_points\n                shape.other_data[\"track_width\"] = half_w * 2\n                for pt in band_points:\n                    shape.add_point(QtCore.QPointF(pt[0], pt[1]))\n                shapes.append(shape)'
c = c.replace(old, new, 1)
open(p, 'w', encoding='utf-8').write(c)
print('step2 OK')
