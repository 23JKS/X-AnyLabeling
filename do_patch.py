import re
p = r"D:\desk\X-AnyLabeling\anylabeling\views\labeling\widgets\auto_labeling\auto_labeling.py"
c = open(p, encoding="utf-8").read()
old_fn = '''    _band_mode = False

    def _on_toggle_band(self):
        """Toggle all track shapes between centerline linestrip and band polygon."""
        self._band_mode = not self._band_mode
        shapes = self.parent.canvas.shapes
        width = self.edit_track_width.value()
'''
idx = c.find(old_fn)
end_marker = '''    def _on_shape_selected(self, selected_shapes):'''
end_idx = c.find(end_marker, idx)
body = c[idx:end_idx]
print(f"found at {idx}, len={len(body)}, ends at {end_idx}")
