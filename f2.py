p=r'D:\desk\X-AnyLabeling\anylabeling\views\labeling\widgets\auto_labeling\auto_labeling.py'
c=open(p,encoding='utf-8').read()
q=chr(34)
o='if shape.other_data.get("centerline") and shape.other_data.get("track_width"):'
n='if shape.other_data.get("_is_band") and shape.other_data.get("centerline") and shape.other_data.get("track_width"):'
c=c.replace(o,n,1)
open(p,'w',encoding='utf-8').write(c)
print(1)