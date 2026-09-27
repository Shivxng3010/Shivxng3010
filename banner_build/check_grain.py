import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from scipy.ndimage import binary_closing, binary_fill_holes, label

img = Image.open(r"D:\Downloads\nccc.jpg")
cropped = img.crop((20, 25, 638, 725))
resized = cropped.resize((300, 340), Image.Resampling.LANCZOS)
gray = ImageOps.grayscale(resized)
gray = ImageOps.autocontrast(gray, cutoff=1)
gray = ImageEnhance.Contrast(gray).enhance(1.3)
gray = gray.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
proc = np.array(gray).astype(np.float32)

arr_rgb = np.array(resized).astype(np.float32)
dist = np.linalg.norm(arr_rgb - np.array([252.0, 253.0, 254.0]), axis=-1)
raw = dist > 14.0
closed = binary_closing(raw, structure=np.ones((3, 3)))
filled = binary_fill_holes(closed)
lbl, num = label(filled)
sizes = [np.sum(lbl == i) for i in range(1, num + 1)]
mask = (lbl == (np.argmax(sizes) + 1))

def run(gray_in):
    a = gray_in / 255.0
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

def bandpower(d, r0, r1, c0, c1, lo=2.5, hi=14.0):
    sub = d[r0:r1, c0:c1].astype(float)
    v = sub.mean(axis=1)
    v = v - v.mean()
    F = np.abs(np.fft.rfft(v))
    fr = np.fft.rfftfreq(len(v), d=1.0)
    tot = F[1:].sum() + 1e-9
    with np.errstate(divide="ignore"):
        pers = np.where(fr > 0, 1.0 / np.maximum(fr, 1e-9), 0)
    sel = (fr > 0) & (pers >= lo) & (pers <= hi)
    return F[sel].sum() / tot

rng = np.random.default_rng(7)
for sig in [0.0, 2.0, 3.0, 4.0]:
    g = proc + (rng.normal(0, sig, proc.shape) if sig > 0 else 0)
    g = np.clip(g, 0, 255)
    d = run(g)
    j = bandpower(d, 250, 310, 60, 240)
    f = bandpower(d, 150, 190, 150, 250)
    print(f"sigma={sig}: jacket bp={j:.3f} cheek bp={f:.3f} dots={int(d.sum())}", flush=True)
    if sig in (0.0, 3.0):
        tag = "nograin" if sig == 0.0 else "grain3"
        for region, r0, r1, hh in (("face", 90, 200, 330), ("jacket", 250, 310, 180)):
            vv = np.ones((r1 - r0, 300, 3), dtype=np.uint8) * 10
            vv[d[r0:r1] == 1] = [167, 139, 250]
            Image.fromarray(vv).resize((900, hh), Image.NEAREST).save(
                rf"C:\Users\Lenovo\Shivxng3010\banner_build\debug_{region}_{tag}.png")
print("done")
