import numpy as np
from PIL import Image

# 读取 TIF（可能是 16-bit 或 32-bit float）
img = Image.open(r'C:\Users\Lenovo\Desktop\data\11.tif')
data = np.array(img, dtype=np.float32)

# Log 压缩
data_compressed = np.log1p(data)  # log(1 + x)

# 归一化到 0-255
data_normalized = (data_compressed - data_compressed.min()) / \
                  (data_compressed.max() - data_compressed.min()) * 255
data_normalized = data_normalized.astype(np.uint8)

# 保存为 PNG
Image.fromarray(data_normalized).save(r'C:\Users\Lenovo\Desktop\data\output.png')