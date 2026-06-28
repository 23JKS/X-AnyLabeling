p=r'D:\desk\X-AnyLabeling\anylabeling\services\auto_labeling\track_centerline.py'
c=open(p).read()
c=c.replace(', "button_export_labels"','')
open(p,'w').write(c)
print(1)