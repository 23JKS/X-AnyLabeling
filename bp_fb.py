p = r'D:\desk\X-AnyLabeling\anylabeling\views\labeling\widgets\auto_labeling\auto_labeling.py'
c = open(p, encoding='utf-8').read()

old = '''            cl = shape.other_data.get(\"centerline\")
            if not cl:
                continue'''

new = '''            cl = shape.other_data.get(\"centerline\")
            if not cl:
                # Fallback: use current linestrip points as centerline
                if shape.shape_type in (\"linestrip\", \"line\") and len(shape.points) >= 2:
                    cl = [(p.x(), p.y()) for p in shape.points]
                    shape.other_data[\"centerline\"] = cl
                else:
                    continue'''

c = c.replace(old, new, 1)
open(p, 'w', encoding='utf-8').write(c)
print('OK')
