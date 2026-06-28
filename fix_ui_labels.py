
import re

path = r"D:\desk\X-AnyLabeling\anylabeling\views\labeling\widgets\auto_labeling\auto_labeling.ui"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Chinese labels as unicode escapes
labels = {
    "min_track": "\u6700\u5c0f\u8f68\u8ff9\u957f\u5ea6",
    "keypoint": "\u5173\u952e\u70b9\u91c7\u6837\u95f4\u9694",
    "default_w": "\u65b0\u6269\u5c55\u591a\u8fb9\u5f62\u7684\u9ed8\u8ba4\u5bbd\u5ea6",
    "per_track": "track_n \u5f53\u524d\u5bbd\u5ea6",
    "expand": "\u6269\u5c55\u4e3a\u5e26\u72b6",
}

# Replace input_min_track_length label
old = re.compile(r'(<widget class="QLabel" name="input_min_track_length">.*?<string>)\?+(</string>)', re.DOTALL)
content = old.sub(r'\1' + labels["min_track"] + r'\2', content)

# Replace input_keypoint_interval label
old = re.compile(r'(<widget class="QLabel" name="input_keypoint_interval">.*?<string>)\?+(</string>)', re.DOTALL)
content = old.sub(r'\1' + labels["keypoint"] + r'\2', content)

# Replace input_track_width label
old = re.compile(r'(<widget class="QLabel" name="input_track_width">.*?<string>)\?+(</string>)', re.DOTALL)
content = old.sub(r'\1' + labels["default_w"] + r'\2', content)

# Replace input_per_track_width label
old = re.compile(r'(<widget class="QLabel" name="input_per_track_width">.*?<string>)[^<]*(</string>)', re.DOTALL)
content = old.sub(r'\1' + labels["per_track"] + r'\2', content)

# Replace button_toggle_band text
old = re.compile(r'(<widget class="QPushButton" name="button_toggle_band">.*?<string>)\?+(</string>)', re.DOTALL)
content = old.sub(r'\1' + labels["expand"] + r'\2', content)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

# Verify
with open(path, "rb") as f:
    raw = f.read()
for m in re.finditer(rb"<string>([^<]+)</string>", raw):
    s = m.group(1)
    if any(b > 127 for b in s):
        print("OK UTF-8:", s.decode("utf-8"))

print("Done fixing .ui Chinese labels")
