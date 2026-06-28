p = r'D:\desk\X-AnyLabeling\anylabeling\services\auto_labeling\model_manager.py'
c = open(p, encoding='utf-8').read()
# add import
old = '_AUTO_LABELING_MIN_TRACK_LENGTH_MODELS,'
new = '_AUTO_LABELING_MIN_TRACK_LENGTH_MODELS,\n    _AUTO_LABELING_TRACK_WIDTH_MODELS,'
c = c.replace(old, new, 1)
# add method before set_mask_fineness
old2 = '    def set_auto_labeling_min_track_length(self, value):'
new2 = '    def set_auto_labeling_track_width(self, value):\n        if (\n            self.loaded_model_config is None\n            or self.loaded_model_config[\"type\"]\n            not in _AUTO_LABELING_TRACK_WIDTH_MODELS\n        ):\n            return\n        self.loaded_model_config[\"model\"].set_auto_labeling_track_width(value)\n\n' + old2
c = c.replace(old2, new2, 1)
open(p, 'w', encoding='utf-8').write(c)
print('OK')
