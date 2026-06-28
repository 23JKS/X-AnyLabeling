p=r'D:\desk\X-AnyLabeling\anylabeling\views\labeling\widgets\auto_labeling\auto_labeling.py'
c=open(p,encoding='utf-8').read()
q=chr(34)
old_fn_start=c.find('def _on_toggle_band(self):')
old_fn_end=c.find('def _on_shape_selected',old_fn_start)
old_body=c[old_fn_start:old_fn_end]
# Write old body to check
print('old body length:',len(old_body))
print(old_body[:500])