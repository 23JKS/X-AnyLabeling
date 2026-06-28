p = r'D:\desk\X-AnyLabeling\anylabeling\services\auto_labeling\track_centerline.py'
c = open(p, encoding='utf-8').read()
old = 'label=\"track\"'
new = 'label=f\"track_{inst_id}\"'
c = c.replace(old, new, 1)
open(p, 'w', encoding='utf-8').write(c)
print('OK')
