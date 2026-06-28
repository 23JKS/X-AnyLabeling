p = r'D:\desk\X-AnyLabeling\anylabeling\views\labeling\widgets\auto_labeling\auto_labeling.ui'
with open(p, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the button_run widget and insert toggle after it
for i, line in enumerate(lines):
    if 'name=\"button_run\"' in line:
        depth = 1
        j = i + 1
        while j < len(lines) and depth > 0:
            if '<widget' in lines[j]:
                depth += 1
            if '</widget>' in lines[j]:
                depth -= 1
            j += 1
        while j < len(lines) and '</item>' not in lines[j]:
            j += 1
        # insert toggle button after this item
        btn_xml = '''     <item>
      <widget class=\"QPushButton\" name=\"button_toggle_band\">
       <property name=\"text\">
        <string>扩展为带状</string>
       </property>
      </widget>
     </item>
'''
        lines.insert(j + 1, btn_xml)
        break

with open(p, 'w', encoding='utf-8') as f:
    f.writelines(lines)
print('OK')
