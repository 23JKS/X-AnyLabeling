path = r'D:\desk\X-AnyLabeling\anylabeling\views\labeling\widgets\auto_labeling\auto_labeling.py'
c = open(path, encoding='utf-8').read()

# The problem: QLabel created without parent, and layout insertion might fail
# Fix: create with self as parent, and use a more robust layout insertion

old = '''        # Insert into layout after edit_iou
        parent_layout = self.edit_iou.parent().layout()
        if parent_layout:
            iou_idx = parent_layout.indexOf(self.edit_iou)
            parent_layout.insertWidget(iou_idx + 1, self.input_min_track_length)
            parent_layout.insertWidget(iou_idx + 2, self.edit_min_track_length)'''

new = '''        # Find the vertical layout that contains edit_iou
        parent_layout = None
        widget = self.edit_iou
        while widget is not None and widget is not self:
            wlayout = widget.parent().layout() if widget.parent() else None
            if wlayout and wlayout.indexOf(widget) >= 0:
                parent_layout = wlayout
                break
            widget = widget.parent()
        # Fallback: use main layout
        if parent_layout is None:
            parent_layout = self.layout()
        if parent_layout:
            iou_idx = parent_layout.indexOf(self.edit_iou)
            if iou_idx >= 0:
                parent_layout.insertWidget(iou_idx + 1, self.input_min_track_length)
                parent_layout.insertWidget(iou_idx + 2, self.edit_min_track_length)'''

c = c.replace(old, new, 1)
open(path, 'w', encoding='utf-8').write(c)
print('OK')
