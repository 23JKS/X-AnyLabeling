p = r'D:\desk\X-AnyLabeling\anylabeling\views\labeling\widgets\auto_labeling\auto_labeling.ui'
c = open(p, encoding='utf-8').read()
old = '<string>轨迹宽度</string>'
new = '<string>轨迹宽度(默认)</string>'
c = c.replace(old, new, 1)
open(p, 'w', encoding='utf-8').write(c)
print('OK')
