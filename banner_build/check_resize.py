import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from scipy.ndimage import binary_closing, binary_fill_holes, label

img = Image.open(r"D:\Downloads\nccc.jpg")
crop = img.crop((20, 25, 638, 725))

variants = {
    "lanczos": crop.resize((300, 340), Image.Resampling.LANCZOS),
    "bicubic": crop.resize((300, 340), Image.Resampling.BICUBIC),
    "blur07": crop.filter(ImageFilter.GaussianBlur(0.7)).resize((300, 340), Image.Resampling.LANCZOS),
    "blur10": crop.filter(ImageFilter.GaussianBlur(1.0)).resize((300, 340), Image.Resampling.LANCZOS),
}

arr0 = np.array(variants["lanczos"]).astype(np.float32)
dist = np.linalg.norm(arr0 - np.array([252.0, 253.0, 254.0]), axis=-1)
raw = dist > 14.0
closed = binary_closing(raw, structure=np.ones((3, 3)))
filled = binary_fill_holes(closed)
lbl, num = label(filled)
sizes = [np.sum(lbl == i) for i in range(1, num + 1)]
mask = (lbl == (np.argmax(sizes) + 1))

def run(resized):
    gray = ImageOps.grayscale(resized)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.3)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    a = np.array(gray).astype(np.float32) / 255.0
    a[~mask] = 0.0
    buf = a.copy()
    h, w = buf.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for r in range(h):
        even = (r % 2 == 0)
        cols = range(w) if even else range(w - 1, -1, -1)
        for c in cols:
            if not mask[r, c]:
                buf[r, c] = 0.0
                continue
            old = buf[r, c]
            nv = 1.0 if old >= 0.5 else 0.0
            out[r, c] = int(nv)
            err = old - nv
            if even:
                if c + 1 < w and mask[r, c + 1]: buf[r, c + 1] += err * 7 / 16
                if r + 1 < h:
                    if c - 1 >= 0 and mask[r + 1, c - 1]: buf[r + 1, c - 1] += err * 3 / 16
                    if mask[r + 1, c]: buf[r + 1, c] += err * 5 / 16
                    if c + 1 < w and mask[r + 1, c + 1]: buf[r + 1, c + 1] += err * 1 / 16
            else:
                if c - 1 >= 0 and mask[r, c - 1]: buf[r, c - 1] += err * 7 / 16
                if r + 1 < h:
                    if c + 1 < w and mask[r + 1, c + 1]: buf[r + 1, c + 1] += err * 3 / 16
                    if mask[r + 1, c]: buf[r + 1, c] += err * 5 / 16
                    if c - 1 >= 0 and mask[r + 1, c - 1]: buf[r + 1, c - 1] += err * 1 / 16
    return out

for name, rs in variants.items():
    d = run(rs)
    print(f"{name}: dots={int(d.sum())}", flush=True)
    vv = np.ones((60, 300, 3), dtype=np.uint8) * 10
    vv[d[250:310] == 1] = [167, 139, 250]
    Image.fromarray(vv).resize((900, 180), Image.NEAREST).save(
        rf"C:\Users\Lenovo\Shivxng3010\banner_build\debug_jk_{name}.png")
print("done")
