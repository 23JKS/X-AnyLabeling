p=r'D:\desk\X-AnyLabeling\anylabeling\services\auto_labeling\track_centerline.py'
c=open(p, encoding='utf-8').read()
old='widgets = [\"button_run\", \"toggle_preserve_existing_annotations\", \"button_toggle_band\", \"edit_min_track_length\", \"edit_track_width\"]'
new='widgets = [\"button_run\", \"toggle_preserve_existing_annotations\", \"button_toggle_band\", \"input_min_track_length\", \"edit_min_track_length\", \"input_track_width\", \"edit_track_width\"]'
c=c.replace(old, new, 1)
open(p, 'w', encoding='utf-8').write(c)
print('OK')
