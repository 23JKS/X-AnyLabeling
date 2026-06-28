p = r'D:\desk\X-AnyLabeling\anylabeling\views\labeling\widgets\auto_labeling\auto_labeling.ui'
c = open(p, encoding='utf-8').read()
c = c.replace('轨迹宽度(默认)', '新扩展轨迹的默认宽度', 1)
open(p, 'w', encoding='utf-8').write(c)
print('OK')
