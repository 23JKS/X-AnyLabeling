p = r'D:\desk\X-AnyLabeling\anylabeling\views\labeling\widgets\auto_labeling\auto_labeling.ui'
c = open(p, encoding='utf-8').read()
old = 'name=\"input_min_track_length\">\n       <property name=\"text\">\n        <string>最小轨迹长度</string>'
new = 'name=\"input_min_track_length\">\n       <property name=\"minimumSize\">\n        <size>\n         <width>120</width>\n         <height>0</height>\n        </size>\n       </property>\n       <property name=\"text\">\n        <string>最小轨迹长度</string>'
c = c.replace(old, new, 1)
open(p, 'w', encoding='utf-8').write(c)
print('OK')
