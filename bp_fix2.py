p = r'D:\desk\X-AnyLabeling\anylabeling\views\labeling\widgets\auto_labeling\auto_labeling.py'
c = open(p, encoding='utf-8').read()
old = '''    def _on_shape_selected(self, selected_shapes):
        """When a shape is clicked, check if it is a track polygon with adjustable width."""
        self._editing_track_shape = None
        if len(selected_shapes) == 1:
            shape = selected_shapes[0]
            if shape.other_data.get(\"centerline\") and shape.other_data.get(\"track_width\"):
                self._editing_track_shape = shape
                # Update slider to show this shape is width (block signals to avoid loop)
                self.edit_track_width.blockSignals(True)
                self.edit_track_width.setValue(int(shape.other_data[\"track_width\"]))
                self.edit_track_width.blockSignals(False)'''
new = '''    def _on_shape_selected(self, selected_shapes):
        """When a track polygon is clicked, show its width on the slider."""
        self._editing_track_shape = None
        if len(selected_shapes) == 1:
            shape = selected_shapes[0]
            if shape.other_data.get(\"centerline\") and shape.other_data.get(\"track_width\"):
                self._editing_track_shape = shape
                self.edit_track_width.blockSignals(True)
                self.edit_track_width.setValue(int(shape.other_data[\"track_width\"]))
                self.edit_track_width.blockSignals(False)'''
c = c.replace(old, new, 1)
open(p, 'w', encoding='utf-8').write(c)
print('OK')
