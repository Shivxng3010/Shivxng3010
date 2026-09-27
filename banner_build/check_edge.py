import numpy as np
from PIL import Image

img = Image.open(r"D:\Downloads\nccc.jpg")
cropped = img.crop((10, 25, 638, 737))
resized = cropped.resize((300, 340), Image.Resampling.LANCZOS)
arr = np.array(resized).astype(np.float32)
lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
print("left cols mean lum:", [round(float(lum[20:320, c].mean()), 1) for c in range(0, 10)])
print("right cols mean lum:", [round(float(lum[20:320, c].mean()), 1) for c in range(290, 300)])
print("top rows mean lum:", [round(float(lum[r, 20:280].mean()), 1) for r in range(0, 8)])
