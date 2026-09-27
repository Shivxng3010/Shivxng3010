import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

d = np.load(r"C:\Users\Lenovo\Shivxng3010\banner_build\portrait_dark_dither.npy")
sub = d[250:310, 100:200].astype(float)
rs = sub.mean(axis=1)
print("jacket dither row density rows 250-309:")
for i, v in enumerate(rs):
    bar = "#" * int(v * 60)
    print(f"{250+i}: {v:.2f} {bar}")
