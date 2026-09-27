import numpy as np
from PIL import Image, ImageOps

img = Image.open(r"D:\Downloads\nccc.jpg")
cropped = img.crop((20, 25, 638, 725))
resized = cropped.resize((300, 340), Image.Resampling.LANCZOS)
g = np.array(ImageOps.grayscale(resized)).astype(np.float32)

def band_stat(a):
    m = np.median(a, axis=1)
    dev = np.zeros_like(m)
    for r in range(a.shape[0]):
        lo, hi = max(0, r - 4), min(a.shape[0], r + 5)
        nb = list(m[lo:r]) + list(m[r + 1:hi])
        dev[r] = abs(m[r] - np.median(nb))
    return m, dev

m0, dev0 = band_stat(g)
print(f"BEFORE: mean dev={dev0.mean():.2f} max dev={dev0.max():.2f} rows>4: {int((dev0>4).sum())}")

def destripe(a, thresh=6.0):
    out = a.copy()
    m = np.median(a, axis=1)
    for r in range(a.shape[0]):
        lo, hi = max(0, r - 4), min(a.shape[0], r + 5)
        nb = list(m[lo:r]) + list(m[r + 1:hi])
        corr = float(np.median(nb) - m[r])
        if abs(corr) <= thresh:
            out[r] += corr
    return np.clip(out, 0, 255)

g2 = destripe(g)
m1, dev1 = band_stat(g2)
print(f"AFTER:  mean dev={dev1.mean():.2f} max dev={dev1.max():.2f} rows>4: {int((dev1>4).sum())}")

# save side-by-side face strips
face0 = g[90:200, :]
face1 = g2[90:200, :]
Image.fromarray(face0.astype(np.uint8)).resize((600, 220), Image.NEAREST).save(
    r"C:\Users\Lenovo\Shivxng3010\banner_build\debug_face_before.png")
Image.fromarray(face1.astype(np.uint8)).resize((600, 220), Image.NEAREST).save(
    r"C:\Users\Lenovo\Shivxng3010\banner_build\debug_face_after.png")
jk0 = g[240:320, :]
jk1 = g2[240:320, :]
Image.fromarray(jk0.astype(np.uint8)).resize((600, 160), Image.NEAREST).save(
    r"C:\Users\Lenovo\Shivxng3010\banner_build\debug_jacket_before.png")
Image.fromarray(jk1.astype(np.uint8)).resize((600, 160), Image.NEAREST).save(
    r"C:\Users\Lenovo\Shivxng3010\banner_build\debug_jacket_after.png")
print("saved before/after strips")
