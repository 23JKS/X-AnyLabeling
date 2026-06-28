p = r'D:\desk\X-AnyLabeling\anylabeling\services\auto_labeling\track_centerline.py'
c = open(p, encoding='utf-8').read()
old = 'self._min_track_length = int(value)'
new = 'self._min_track_length = int(value)\n\n    def set_auto_labeling_track_width(self, value):\n        self._track_width = int(value)'
c = c.replace(old, new, 1)
open(p, 'w', encoding='utf-8').write(c)
print('step3 OK')
