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

def notch_destripe(a, lo_per, hi_per, damp=0.8, protect=12.0):
    H = a.shape[0]
    m = np.median(a, axis=1)
    wide = np.array([np.median(m[max(0, r - 20):min(H, r + 21)]) for r in range(H)])
    F = np.fft.rfft(m)
    freqs = np.fft.rfftfreq(H, d=1.0)
    bp = np.zeros_like(F)
    with np.errstate(divide="ignore"):
        pers = np.where(freqs > 0, 1.0 / np.maximum(freqs, 1e-9), 0)
    sel = (freqs > 0) & (pers >= lo_per) & (pers <= hi_per)
    bp[sel] = F[sel]
    wave = np.fft.irfft(bp, n=H)
    out = a.copy()
    for r in range(H):
        if abs(m[r] - wide[r]) <= protect:
            out[r] -= damp * wave[r]
    return np.clip(out, 0, 255)

def run(gray_in):
    g = Image.fromarray(np.clip(gray_in, 0, 255).astype(np.uint8))
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

variants = {"wide614": (6.0, 14.0), "wide2514": (2.5, 14.0)}
for name, (lo, hi) in variants.items():
    g = notch_destripe(base, lo, hi)
    d = run(g)
    j = bandpower(d, 250, 310, 60, 240)
    f = bandpower(d, 150, 190, 150, 250)
    print(f"{name}: jacket bp={j:.3f} cheek bp={f:.3f} dots={int(d.sum())}", flush=True)
    vv = np.ones((60, 300, 3), dtype=np.uint8) * 10
    vv[d[250:310] == 1] = [167, 139, 250]
    Image.fromarray(vv).resize((900, 180), Image.NEAREST).save(
        rf"C:\Users\Lenovo\Shivxng3010\banner_build\debug_jacket_{name}.png")
print("done")
