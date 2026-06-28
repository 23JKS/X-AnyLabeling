p = r'D:\desk\X-AnyLabeling\anylabeling\views\labeling\widgets\auto_labeling\auto_labeling.py'
c = open(p, encoding='utf-8').read()
c = c.replace('if shape.label != \"track\":', 'if not shape.label.startswith(\"track\"):')
open(p, 'w', encoding='utf-8').write(c)
print('OK')
