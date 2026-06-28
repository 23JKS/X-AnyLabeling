p = r'D:\desk\X-AnyLabeling\anylabeling\views\labeling\widgets\auto_labeling\auto_labeling.ui'
c = open(p, encoding='utf-8').read()
old = '''      <widget class=\"QLabel\" name=\"input_track_width\">
       <property name=\"text\">
        <string>新扩展轨迹的默认宽度</string>
       </property>
      </widget>'''
new = '''      <widget class=\"QLabel\" name=\"input_track_width\">
       <property name=\"minimumSize\">
        <size>
         <width>160</width>
         <height>0</height>
        </size>
       </property>
       <property name=\"text\">
        <string>新扩展轨迹的默认宽度</string>
       </property>
      </widget>'''
c = c.replace(old, new, 1)
open(p, 'w', encoding='utf-8').write(c)
print('OK')
