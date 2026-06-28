path = r'D:\desk\X-AnyLabeling\anylabeling\services\auto_labeling\model_manager.py'
c = open(path, encoding='utf-8').read()

# Add import
old = '_AUTO_LABELING_MASK_FINENESS_MODELS,'
new = old + '\n    _AUTO_LABELING_MIN_TRACK_LENGTH_MODELS,'
c = c.replace(old, new, 1)

# Add method before set_mask_fineness
old2 = '    def set_mask_fineness(self, epsilon):'
new2 = '''    def set_auto_labeling_min_track_length(self, value):
        if (
            self.loaded_model_config is None
            or self.loaded_model_config[\"type\"]
            not in _AUTO_LABELING_MIN_TRACK_LENGTH_MODELS
        ):
            return
        self.loaded_model_config[\"model\"].set_auto_labeling_min_track_length(value)

''' + old2
c = c.replace(old2, new2, 1)

open(path, 'w', encoding='utf-8').write(c)
print('OK')
