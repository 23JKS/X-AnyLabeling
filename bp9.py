p = r'D:\desk\X-AnyLabeling\anylabeling\views\labeling\widgets\auto_labeling\auto_labeling.ui'
with open(p, 'r', encoding='utf-8') as f:
    lines = f.readlines()

insert_xml = '''     <item>
      <widget class=\"QLabel\" name=\"input_track_width\">
       <property name=\"text\">
        <string>轨迹宽度</string>
       </property>
      </widget>
     </item>
     <item>
      <widget class=\"QSpinBox\" name=\"edit_track_width\">
       <property name=\"minimum\">
        <number>2</number>
       </property>
       <property name=\"maximum\">
        <number>100</number>
       </property>
       <property name=\"singleStep\">
        <number>2</number>
       </property>
       <property name=\"value\">
        <number>15</number>
       </property>
      </widget>
     </item>
'''

# Find the line with edit_min_track_length closing widget tag and insert after
for i, line in enumerate(lines):
    if 'name=\"edit_min_track_length\"' in line:
        # find the closing </widget> for this widget
        depth = 1
        j = i + 1
        while j < len(lines) and depth > 0:
            if '<widget' in lines[j]:
                depth += 1
            if '</widget>' in lines[j]:
                depth -= 1
            j += 1
        # now j is after </widget>, find </item>
        while j < len(lines) and '</item>' not in lines[j]:
            j += 1
        # insert after </item>
        lines.insert(j + 1, insert_xml)
        break

with open(p, 'w', encoding='utf-8') as f:
    f.writelines(lines)
print('OK')
