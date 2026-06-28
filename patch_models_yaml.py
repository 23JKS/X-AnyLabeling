path = r"D:\desk\X-AnyLabeling\anylabeling\configs\models.yaml"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

new_entry = '\n- model_name: track_centerline_r18\n  config_file: :/track_centerline_r18.yaml\n'

if "track_centerline" in content:
    print("Already in models.yaml")
else:
    content += new_entry
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Added to models.yaml")
