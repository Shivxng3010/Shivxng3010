import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from scipy.ndimage import binary_closing, binary_fill_holes, label

img = Image.open(r"D:\Downloads\nccc.jpg")
cropped = img.crop((20, 25, 638, 725))
resized = cropped.resize((300, 340), Image.Resampling.LANCZOS)
base = np.array(ImageOps.grayscale(resized)).astype(np.float32)

arr_rgb = np.array(resized).astype(np.float32)
dist = np.linalg.norm(arr_rgb - np.array([252.0, 253.0, 254.0]), axis=-1)
raw = dist > 14.0
closed = binary_closing(raw, structure=np.ones((3, 3)))
filled = binary_fill_holes(closed)
lbl, num = label(filled)
sizes = [np.sum(lbl == i) for i in range(1, num + 1)]
mask = (lbl == (np.argmax(sizes) + 1))

def destripe(a, thresh):
    out = a.copy()
    m = np.median(a, axis=1)
    for r in range(a.shape[0]):
        lo, hi = max(0, r - 4), min(a.shape[0], r + 5)
        nb = list(m[lo:r]) + list(m[r + 1:hi])
        corr = float(np.median(nb) - m[r])
        if abs(corr) <= thresh:
            out[r] += corr
    return np.clip(out, 0, 255)

def full_dither(gray_arr):
    g = Image.fromarray(gray_arr.astype(np.uint8))
    g = ImageOps.autocontrast(g, cutoff=1)
    g = ImageEnhance.Contrast(g).enhance(1.3)
    g = g.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    a = np.array(g).astype(np.float32) / 255.0
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

def rowband_metric(d):
    s = d.sum(axis=1).astype(float)
    dev = np.zeros_like(s)
    for r in range(len(s)):
        lo, hi = max(0, r - 3), min(len(s), r + 4)
        nb = list(s[lo:r]) + list(s[r + 1:hi])
        dev[r] = abs(s[r] - np.median(nb))
    m = mask.sum(axis=1) > 100
    return float(dev[m].mean()), float(dev[m].max()), int((dev[m] > 12).sum())

for th in [None, 6.0, 4.0, 2.5]:
    g = destripe(base, th) if th else base.copy()
    d = full_dither(g)
    mean, mx, n = rowband_metric(d)
    print(f"thresh={th}: dither rowband mean={mean:.2f} max={mx:.1f} rows>12: {n} dots={int(d.sum())}")
    if th in (None, 2.5):
        tag = "none" if th is None else "t25"
        dv = np.ones((110, 300, 3), dtype=np.uint8) * 10
        dv[d[90:200] == 1] = [167, 139, 250]
        Image.fromarray(dv).resize((900, 330), Image.NEAREST).save(
            rf"C:\Users\Lenovo\Shivxng3010\banner_build\debug_face_{tag}.png")
        jv = np.ones((80, 300, 3), dtype=np.uint8) * 10
        jv[d[240:320] == 1] = [167, 139, 250]
        Image.fromarray(jv).resize((900, 240), Image.NEAREST).save(
            rf"C:\Users\Lenovo\Shivxng3010\banner_build\debug_jacket_{tag}.png")
print("done")
