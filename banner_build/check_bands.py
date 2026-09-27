import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

img = Image.open(r"D:\Downloads\nccc.jpg")
cropped = img.crop((20, 25, 638, 725))
resized = cropped.resize((300, 340), Image.Resampling.LANCZOS)

# 1) raw resized gray row means (before pipeline processing)
raw = np.array(ImageOps.grayscale(resized)).astype(np.float32)
rr = raw.mean(axis=1)
print("RAW row means, find sharp dips (diff of neighbors):")
d2 = np.abs(rr[1:-1] - (rr[:-2] + rr[2:]) / 2.0)
top = np.argsort(d2)[-12:]
print("most anomalous raw rows:", sorted(int(t) + 1 for t in top))
for t in sorted(top):
    r = int(t) + 1
    print(f"  row {r}: mean={rr[r]:.1f} neighbors={rr[r-1]:.1f},{rr[r+1]:.1f}")

# 2) pipeline-processed gray row means
gray = ImageOps.grayscale(resized)
gray = ImageOps.autocontrast(gray, cutoff=1)
gray = ImageEnhance.Contrast(gray).enhance(1.3)
gray = gray.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
g = np.array(gray).astype(np.float32)
gr = g.mean(axis=1)
e2 = np.abs(gr[1:-1] - (gr[:-2] + gr[2:]) / 2.0)
top2 = np.argsort(e2)[-12:]
print("most anomalous PROCESSED rows:", sorted(int(t) + 1 for t in top2))

# 3) dither row sums
d = np.load(r"C:\Users\Lenovo\Shivxng3010\banner_build\portrait_dark_dither.npy")
ds = d.sum(axis=1).astype(float)
f2 = np.abs(ds[1:-1] - (ds[:-2] + ds[2:]) / 2.0)
top3 = np.argsort(f2)[-12:]
print("most anomalous DITHER rows:", sorted(int(t) + 1 for t in top3))
for t in sorted(top3):
    r = int(t) + 1
    print(f"  dither row {r}: dots={int(ds[r])} neighbors={int(ds[r-1])},{int(ds[r+1])}")
