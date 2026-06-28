path = r"D:\desk\X-AnyLabeling\anylabeling\services\auto_labeling\__init__.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Add track_centerline to _CUSTOM_MODELS (after the last entry)
old = '    "deimv2",\n]'
new = '    "deimv2",\n    "track_centerline",\n]'

if "track_centerline" in content:
    print("track_centerline already in _CUSTOM_MODELS")
elif old in content:
    content = content.replace(old, new, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Added track_centerline to _CUSTOM_MODELS")
else:
    print(f"ERROR: Could not find insertion point. Last 200 chars: {content[-200:]}")
