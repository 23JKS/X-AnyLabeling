p=r'D:\desk\X-AnyLabeling\anylabeling\views\labeling\widgets\auto_labeling\auto_labeling.py'
c=open(p,encoding='utf-8').read()
old_start=c.find('self._band_shapes.append(band_shape)')
old_end=c.find('self.button_toggle_band.setText',old_start)
old=c[old_start:old_end]
new='self._band_shapes.append(band_shape)\n        else:\n            self.parent.canvas.shapes = [s for s in self.parent.canvas.shapes if not (s.label.endswith("_band") and s.other_data.get("_is_band"))]\n            self._band_shapes = []\n\n        '
c=c.replace(old,new,1)
open(p,'w',encoding='utf-8').write(c)
print('5')