import base64
fp = r'D:\\desk\\X-AnyLabeling\\anylabeling\\views\\labeling\\widgets\\canvas.py'
c = open(fp, 'r', encoding='utf-8').read()
old = '            index = shape.nearest_vertex(pos, self.epsilon / self.scale)'
new = '            is_band = shape.other_data.get("_is_band", False)\n            if not is_band:\n                index = shape.nearest_vertex(pos, self.epsilon / self.scale)\n                index_edge = shape.nearest_edge(pos, self.epsilon / self.scale)\n            else:\n                index = None\n                index_edge = None\n            ' + old
c = c.replace(old, new, 1)
open(fp, 'w', encoding='utf-8').write(c)
print('OK')