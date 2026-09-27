import numpy as np
from PIL import Image

d = np.load(r"C:\Users\Lenovo\Shivxng3010\banner_build\portrait_dark_dither.npy")
vis = np.ones((340, 300, 3), dtype=np.uint8) * 10
vis[d == 1] = [167, 139, 250]
Image.fromarray(vis).save(r"C:\Users\Lenovo\Shivxng3010\banner_build\debug_dither_dark2.png")
print("dots:", int(d.sum()))
