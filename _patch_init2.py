path = r"D:\desk\X-AnyLabeling\anylabeling\services\auto_labeling\__init__.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old = "_AUTO_LABELING_MASK_FINENESS_MODELS = ["
new = (
    "_AUTO_LABELING_MIN_TRACK_LENGTH_MODELS = [\n"
    '    "track_centerline",\n'
    "]\n\n\n"
    "# --- set_mask_fineness ---\n"
    "_AUTO_LABELING_MASK_FINENESS_MODELS = ["
)

if "_AUTO_LABELING_MIN_TRACK_LENGTH" in content:
    print("Already exists")
else:
    content = content.replace(old, new, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Added")
