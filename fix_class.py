import sys
sys.stdout.reconfigure(encoding="utf-8")
path = r"D:\desk\X-AnyLabeling\anylabeling\services\auto_labeling\track_centerline.py"
lines = open(path, "r", encoding="utf-8-sig").readlines()

# Find where __init__ is (missing class wrapper)
for i, line in enumerate(lines):
    if line.strip().startswith("def __init__(self, model_config"):
        # Insert class header before this line
        insert = [
            "\n",
            "# ======================================================================\n",
            "#  X-AnyLabeling model class\n",
            "# ======================================================================\n",
            "\n",
            "class TrackCenterline(Model):\n",
            "\n",
        ]
        new_lines = lines[:i] + insert + lines[i:]
        open(path, "w", encoding="utf-8").write("".join(new_lines))
        print(f"Inserted class header before line {i+1}")
        break
