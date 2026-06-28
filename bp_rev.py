p = r'D:\desk\X-AnyLabeling\anylabeling\services\auto_labeling\track_centerline.py'
c = open(p, encoding='utf-8').read()

# Revert output back to linestrip (store centerline for later expansion)
old = '''label=\"track\",
                    shape_type=\"polygon\",
                    flags={},
                )
                half_w = getattr(self, \"_track_width\", 15) / 2.0
                band_points = self._polyline_to_band_polygon(polyline_points, half_w)
                shape.other_data[\"centerline\"] = polyline_points
                shape.other_data[\"track_width\"] = half_w * 2
                for pt in band_points:
                    shape.add_point(QtCore.QPointF(pt[0], pt[1]))
                shapes.append(shape)'''

new = '''label=\"track\",
                    shape_type=\"linestrip\",
                    flags={},
                )
                # Store centerline for later band expansion
                shape.other_data[\"centerline\"] = polyline_points
                for pt in polyline_points:
                    shape.add_point(QtCore.QPointF(pt[0], pt[1]))
                shapes.append(shape)'''

c = c.replace(old, new, 1)
open(p, 'w', encoding='utf-8').write(c)
print('OK')
