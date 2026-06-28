p = r'D:\desk\X-AnyLabeling\anylabeling\services\auto_labeling\__init__.py'
c = open(p, encoding='utf-8').read()
old = '_AUTO_LABELING_MIN_TRACK_LENGTH_MODELS = ['
new = '_AUTO_LABELING_TRACK_WIDTH_MODELS = [\n    \"track_centerline\",\n]\n\n\n# --- set_min_track_length ---\n_AUTO_LABELING_MIN_TRACK_LENGTH_MODELS = ['
c = c.replace(old, new, 1)
open(p, 'w', encoding='utf-8').write(c)
print('OK')
