path = r'D:\desk\X-AnyLabeling\anylabeling\services\auto_labeling\track_centerline.py'
c = open(path, encoding='utf-8').read()
old = '        self.replace = not state'
new = '''        self.replace = not state

    def set_auto_labeling_min_track_length(self, value):
        self._min_track_length = int(value)'''
c = c.replace(old, new, 1)

# Also update predict_shapes to use self._min_track_length if set
old2 = '        min_track_length = pp.get(\"min_track_length\", 30)'
new2 = '        min_track_length = getattr(self, \"_min_track_length\", pp.get(\"min_track_length\", 30))'
c = c.replace(old2, new2, 1)

open(path, 'w', encoding='utf-8').write(c)
print('OK')
