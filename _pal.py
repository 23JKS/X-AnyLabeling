import re

path = r'D:\desk\X-AnyLabeling\anylabeling\views\labeling\widgets\auto_labeling\auto_labeling.py'
c = open(path, encoding='utf-8').read()

# 1. Add QSpinBox import
if 'QSpinBox' not in c.split('from PyQt6.QtWidgets import')[1].split(')')[0]:
    c = c.replace(
        'from PyQt6.QtWidgets import (',
        'from PyQt6.QtWidgets import (\n    QSpinBox,'
    )

# 2. Add _AUTO_LABELING_MIN_TRACK_LENGTH_MODELS import
c = c.replace(
    '    _AUTO_LABELING_MASK_FINENESS_MODELS,',
    '    _AUTO_LABELING_MASK_FINENESS_MODELS,\n    _AUTO_LABELING_MIN_TRACK_LENGTH_MODELS,'
)

# 3. Add to supported_remote_widgets (the list at class level)
c = c.replace(
    '        \"mask_fineness_value_label\",\n    ]',
    '        \"mask_fineness_value_label\",\n        \"edit_min_track_length\",\n        \"input_min_track_length\",\n    ]'
)

# 4. Create programmatic widgets after edit_iou setup (~line 288)
# Find the section after edit_iou.valueChanged.connect
old4 = '''        self.edit_iou.valueChanged.connect(self.on_iou_value_changed)'''
new4 = '''        self.edit_iou.valueChanged.connect(self.on_iou_value_changed)

        # --- Configuration for: edit_min_track_length ---
        self.input_min_track_length = QLabel(self.tr(\"Min Track Length\"))
        self.input_min_track_length.setStyleSheet(\"font-size: 12px;\")
        self.edit_min_track_length = QSpinBox()
        self.edit_min_track_length.setStyleSheet(get_double_spinbox_style())
        self.edit_min_track_length.setMinimum(5)
        self.edit_min_track_length.setMaximum(500)
        self.edit_min_track_length.setSingleStep(5)
        self.edit_min_track_length.setValue(30)
        self.edit_min_track_length.valueChanged.connect(self.on_min_track_length_changed)
        self.edit_min_track_length.setToolTip(
            self.tr(\"Minimum polyline length in pixels. Shorter tracks are filtered out.\")
        )
        # Insert into layout after edit_iou
        parent_layout = self.edit_iou.parent().layout()
        if parent_layout:
            iou_idx = parent_layout.indexOf(self.edit_iou)
            parent_layout.insertWidget(iou_idx + 1, self.input_min_track_length)
            parent_layout.insertWidget(iou_idx + 2, self.edit_min_track_length)'''

c = c.replace(old4, new4, 1)

# 5. Add on_min_track_length_changed method before on_mask_fineness_changed
old5 = '    def on_mask_fineness_changed(self, value):'
new5 = '''    def on_min_track_length_changed(self, value):
        self.model_manager.set_auto_labeling_min_track_length(value)

    def on_mask_fineness_changed(self, value):'''
c = c.replace(old5, new5, 1)

# 6. Add to hide_labeling_widgets list
c = c.replace(
    '            \"mask_fineness_value_label\",\n        ]',
    '            \"mask_fineness_value_label\",\n            \"edit_min_track_length\",\n            \"input_min_track_length\",\n        ]'
)

open(path, 'w', encoding='utf-8').write(c)
print('OK')
