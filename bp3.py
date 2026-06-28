p = r'D:\desk\X-AnyLabeling\anylabeling\services\auto_labeling\track_centerline.py'
c = open(p, encoding='utf-8').read()

# step1: add _polyline_to_band_polygon
marker = '_polyline_length(points: list) -> float:'
band_func = '_polyline_to_band_polygon(points, half_width):\n        if len(points) < 2:\n            return points\n        n = len(points)\n        left_pts = []\n        right_pts = []\n        for i in range(n):\n            if i == 0:\n                dx = points[1][0] - points[0][0]\n                dy = points[1][1] - points[0][1]\n            elif i == n - 1:\n                dx = points[n-1][0] - points[n-2][0]\n                dy = points[n-1][1] - points[n-2][1]\n            else:\n                dx = points[i+1][0] - points[i-1][0]\n                dy = points[i+1][1] - points[i-1][1]\n            length = (dx*dx + dy*dy) ** 0.5\n            if length < 1e-6:\n                continue\n            nx = -dy / length\n            ny = dx / length\n            left_pts.append((points[i][0] + nx * half_width, points[i][1] + ny * half_width))\n            right_pts.append((points[i][0] - nx * half_width, points[i][1] - ny * half_width))\n        if len(left_pts) < 2:\n            return points\n        return left_pts + list(reversed(right_pts))\n\n    @staticmethod\n    def '
c = c.replace(marker, band_func + marker, 1)

# step2: change linestrip to polygon with band expansion
open(p, 'w', encoding='utf-8').write(c)
print('step1 OK')
