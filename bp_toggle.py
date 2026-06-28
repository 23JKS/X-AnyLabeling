p = r'D:\desk\X-AnyLabeling\anylabeling\views\labeling\widgets\auto_labeling\auto_labeling.py'
c = open(p, encoding='utf-8').read()

# Add button_toggle_band to supported_remote_widgets
old = '\"edit_track_width\",\n        \"input_track_width\",'
new = '\"edit_track_width\",\n        \"input_track_width\",\n        \"button_toggle_band\",'
c = c.replace(old, new, 1)

# Add to hide_labeling_widgets
old2 = '\"edit_track_width\",\n            \"input_track_width\",'
new2 = '\"edit_track_width\",\n            \"input_track_width\",\n            \"button_toggle_band\",'
c = c.replace(old2, new2, 1)

# Connect button signal - find a good spot after __init__
old3 = 'self.model_manager.new_model_status.connect(self.on_new_model_status)'
new3 = 'self.model_manager.new_model_status.connect(self.on_new_model_status)\n        self.button_toggle_band.clicked.connect(self._on_toggle_band)'
c = c.replace(old3, new3, 1)

# Add toggle band method before _on_shape_selected
old_method = '    def _on_shape_selected(self, selected_shapes):'
toggle_fn = '''    _band_mode = False

    def _on_toggle_band(self):
        """Toggle all track shapes between centerline linestrip and band polygon."""
        self._band_mode = not self._band_mode
        shapes = self.parent.canvas.shapes
        width = self.edit_track_width.value()

        for shape in shapes:
            if shape.label != \"track\":
                continue
            cl = shape.other_data.get(\"centerline\")
            if not cl:
                continue

            if self._band_mode:
                # Expand to polygon
                half_w = width / 2.0
                band_pts = self._regenerate_band_polygon(cl, half_w)
                shape.shape_type = \"polygon\"
                shape.points = [QPointF(x, y) for x, y in band_pts]
                shape.other_data[\"track_width\"] = width
            else:
                # Restore centerline
                shape.shape_type = \"linestrip\"
                shape.points = [QPointF(x, y) for x, y in cl]
                shape.other_data[\"track_width\"] = width

            shape.update()

        self.button_toggle_band.setText(
            self.tr(\"显示中心线\") if self._band_mode else self.tr(\"扩展为带状\")
        )
        self.parent.canvas.update()
'''
c = c.replace(old_method, toggle_fn + '\n' + old_method, 1)

open(p, 'w', encoding='utf-8').write(c)
print('OK')
