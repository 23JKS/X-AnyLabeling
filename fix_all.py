p=r'D:\desk\X-AnyLabeling\anylabeling\views\labeling\widgets\auto_labeling\auto_labeling.py'
c=open(p,encoding='utf-8').read()
q=chr(34)
# 1. Change band label to match centerline label
old1='label=shape.label + "_band"'
new1='label=shape.label'
c=c.replace(old1,new1,1)
# 2. Also fix the label check in filter
old2='or shape.label.endswith("_band")'
new2='or shape.other_data.get("_is_band")'
c=c.replace(old2,new2,1)
# 3. Fix the else clause filter too
old3='s.label.endswith("_band") and s.other_data.get("_is_band")'
new3='s.other_data.get("_is_band")'
c=c.replace(old3,new3,1)
open(p,'w',encoding='utf-8').write(c)
print(1)