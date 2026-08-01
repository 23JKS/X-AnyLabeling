import cv2
import os
import numpy as np

p = r"C:\Users\Lenovo\Desktop\data\4.tif"
print("path:", p)
print("exists:", os.path.isfile(p))

img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
print("img is None:", img is None)
if img is not None:
    print("dtype:", img.dtype)
    print("shape:", img.shape)
    print("min:", img.min(), "max:", img.max())
    print("itemsize bytes:", img.dtype.itemsize, "-> bit:", img.dtype.itemsize * 8)
    # unique value count to detect if it's really 8-bit stored as 16-bit
    uniq = np.unique(img)
    print("num unique values:", len(uniq))
    print("first 10 unique:", uniq[:10])
    print("last 10 unique:", uniq[-10:])
