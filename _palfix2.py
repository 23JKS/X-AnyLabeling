path = r'D:\desk\X-AnyLabeling\anylabeling\views\labeling\widgets\auto_labeling\auto_labeling.py'
c = open(path, encoding='utf-8').read()
old = 'QLabel(self.tr(\"Min Track Length\"))'
new = 'QLabel(\"Min Track Length\", self)'
c = c.replace(old, new, 1)
open(path, 'w', encoding='utf-8').write(c)
print('OK')
