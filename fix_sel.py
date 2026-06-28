p=r'D:\desk\X-AnyLabeling\anylabeling\views\labeling\widgets\auto_labeling\auto_labeling.py'
c=open(p,encoding='utf-8').read()
old='        if hasattr(self.parent, "canvas") and hasattr(self.parent.canvas, "selection_changed"):\n            self.parent.canvas.selection_changed.connect(self._on_shape_selected)'
new='        try:\n            self.parent.canvas.selection_changed.connect(self._on_shape_selected)\n        except Exception:\n            pass'
c=c.replace(old,new,1)
open(p,'w',encoding='utf-8').write(c)
print(1)