import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

img = Image.open(r"D:\Downloads\nccc.jpg")
cropped = img.crop((20, 25, 638, 725))
resized = cropped.resize((300, 340), Image.Resampling.LANCZOS)
g = np.array(ImageOps.grayscale(resized)).astype(np.float32)

# pure backdrop columns on the left (x 4..36), rows 60..300 (no subject there?)
bg = g[60:300, 4:36]
rm = bg.mean(axis=1)
dev = np.abs(rm[1:-1] - (rm[:-2] + rm[2:]) / 2.0)
print("backdrop row-mean overall:", round(float(rm.mean()), 1))
print("backdrop row dev mean:", round(float(dev.mean()), 2), "max:", round(float(dev.max()), 2))
worst = np.argsort(dev)[-8:]
print("worst backdrop rows:", sorted(int(w) + 61 for w in worst))
for w in sorted(worst):
    r = int(w) + 61
    print(f"  row {r}: mean={rm[r-60]:.1f} neighbors={rm[r-61]:.1f},{rm[r-59]:.1f}")

# right backdrop columns (x 270..295), rows 60..200
bg2 = g[60:200, 270:295]
rm2 = bg2.mean(axis=1)
dev2 = np.abs(rm2[1:-1] - (rm2[:-2] + rm2[2:]) / 2.0)
print("right-backdrop row dev mean:", round(float(dev2.mean()), 2), "max:", round(float(dev2.max()), 2))
