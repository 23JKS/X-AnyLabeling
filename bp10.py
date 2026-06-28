p = r'D:\desk\X-AnyLabeling\anylabeling\views\labeling\widgets\auto_labeling\auto_labeling.py'
c = open(p, encoding='utf-8').read()

# 1. Add import
old = '_AUTO_LABELING_MIN_TRACK_LENGTH_MODELS,'
new = '_AUTO_LABELING_MIN_TRACK_LENGTH_MODELS,\n    _AUTO_LABELING_TRACK_WIDTH_MODELS,'
c = c.replace(old, new, 1)

# 2. Add to supported_remote_widgets
old = '\"edit_min_track_length\",\n        \"input_min_track_length\",'
new = '\"edit_min_track_length\",\n        \"input_min_track_length\",\n        \"edit_track_width\",\n        \"input_track_width\",'
c = c.replace(old, new, 1)

# 3. Add signal connection after edit_min_track_length setup
old = 'self.edit_min_track_length.valueChanged.connect(self.on_min_track_length_changed)'
new = 'self.edit_min_track_length.valueChanged.connect(self.on_min_track_length_changed)\n        self.edit_track_width.valueChanged.connect(self.on_track_width_changed)'
c = c.replace(old, new, 1)

# 4. Add handler method before on_min_track_length_changed
old = '    def on_min_track_length_changed(self, value):'
new = '    def on_track_width_changed(self, value):\n        self.model_manager.set_auto_labeling_track_width(value)\n\n    def on_min_track_length_changed(self, value):'
c = c.replace(old, new, 1)

# 5. Add to hide_labeling_widgets
old = '\"edit_min_track_length\",\n            \"input_min_track_length\",'
new = '\"edit_min_track_length\",\n            \"input_min_track_length\",\n            \"edit_track_width\",\n            \"input_track_width\",'
c = c.replace(old, new, 1)

open(p, 'w', encoding='utf-8').write(c)
print('OK')
