
import re

# 1. model_manager.py
fp1 = r"D:\desk\X-AnyLabeling\anylabeling\services\auto_labeling\model_manager.py"
c = open(fp1, "r", encoding="utf-8").read()
old = "def set_auto_labeling_track_width(self, value):"
new = "def set_auto_labeling_keypoint_interval(self, value):\n        if hasattr(self.loaded_model_config, \"set_auto_labeling_keypoint_interval\"):\n            self.loaded_model_config[\"model\"].set_auto_labeling_keypoint_interval(value)\n\n    def set_auto_labeling_track_width(self, value):"
c = c.replace(old, new, 1)
open(fp1, "w", encoding="utf-8").write(c)
print("1 done")

# 2. track_centerline.py
fp2 = r"D:\desk\X-AnyLabeling\anylabeling\services\auto_labeling\track_centerline.py"
c = open(fp2, "r", encoding="utf-8").read()
old_set = "def set_auto_labeling_track_width(self, value):\n        self._track_width = int(value)\n\n    def predict_shapes"
new_set = "def set_auto_labeling_track_width(self, value):\n        self._track_width = int(value)\n\n    def set_auto_labeling_keypoint_interval(self, value):\n        self._keypoint_interval = int(value)\n\n    def predict_shapes"
c = c.replace(old_set, new_set, 1)
c = c.replace("interval = 5.0", "interval = float(getattr(self, \"_keypoint_interval\", 5))", 1)
open(fp2, "w", encoding="utf-8").write(c)
print("2 done")

# 3. track_centerline_light.py
fp3 = r"D:\desk\X-AnyLabeling\anylabeling\services\auto_labeling\track_centerline_light.py"
c = open(fp3, "r", encoding="utf-8").read()
c = c.replace(old_set, new_set, 1)
c = c.replace("interval = 5.0", "interval = float(getattr(self, \"_keypoint_interval\", 5))", 1)
open(fp3, "w", encoding="utf-8").write(c)
print("3 done")
