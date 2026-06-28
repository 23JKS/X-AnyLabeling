p = r'D:\desk\X-AnyLabeling\anylabeling\services\auto_labeling\track_centerline.py'
c = open(p, encoding='utf-8').read()
old = '_polyline_length(points: list) -> float:'
new = '''_polyline_to_band_polygon(points, half_width):
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
'''
c = c.replace(old, new + chr(10) + old, 1)
open(p, 'w', encoding='utf-8').write(c)
print('step1 OK')
