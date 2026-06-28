p = r'D:\desk\X-AnyLabeling\anylabeling\views\labeling\widgets\auto_labeling\auto_labeling.py'
c = open(p, encoding='utf-8').read()
old = 'from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QPoint, QTimer'
new = 'from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QPoint, QPointF, QTimer'
c = c.replace(old, new, 1)
# Fix QtCore.QPointF to just QPointF
c = c.replace('QtCore.QPointF', 'QPointF')
open(p, 'w', encoding='utf-8').write(c)
print('OK')
