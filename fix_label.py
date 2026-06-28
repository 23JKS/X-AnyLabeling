path = r"D:\desk\X-AnyLabeling\anylabeling\services\auto_labeling\track_centerline.py"
c = open(path, encoding="utf-8").read()
c = c.replace('label=f"track_{inst_id}"', 'label="track"', 1)
open(path, "w", encoding="utf-8").write(c)
print("OK")
