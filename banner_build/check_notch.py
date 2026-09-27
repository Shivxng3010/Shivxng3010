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

def notch_destripe(a, lo_per=6.0, hi_per=14.0, damp=0.75, protect=12.0):
    H = a.shape[0]
    m = np.median(a, axis=1)
    wide = np.zeros_like(m)
    for r in range(H):
        x0, x1 = max(0, r - 20), min(H, r + 21)
        wide[r] = np.median(m[x0:x1])
    F = np.fft.rfft(m)
    freqs = np.fft.rfftfreq(H, d=1.0)
    bp = np.zeros_like(F)
    sel = (freqs > 0) & ((1.0 / freqs) >= lo_per) & ((1.0 / freqs) <= hi_per)
    bp[sel] = F[sel]
    wave = np.fft.irfft(bp, n=H)
    out = a.copy()
    for r in range(H):
        if abs(m[r] - wide[r]) <= protect:
            out[r] -= damp * wave[r]
    return np.clip(out, 0, 255), wave

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

def bandpower(d, r0, r1, c0, c1, lo=6.0, hi=14.0):
    sub = d[r0:r1, c0:c1].astype(float)
    v = sub.mean(axis=1)
    v = v - v.mean()
    F = np.abs(np.fft.rfft(v))
    fr = np.fft.rfftfreq(len(v), d=1.0)
    tot = F[1:].sum() + 1e-9
    sel = (fr > 0) & ((1.0 / fr) >= lo) & ((1.0 / fr) <= hi)
    return F[sel].sum() / tot

g_notch, wave = notch_destripe(base)
print("wave amplitude std:", round(float(wave.std()), 2), "max:", round(float(np.abs(wave).max()), 2))
for name, g in [("base", base), ("notched", g_notch)]:
    d = run(g)
    j = bandpower(d, 250, 310, 60, 240)
    f = bandpower(d, 150, 190, 150, 250)
    print(f"{name}: jacket 6-14bp={j:.3f} cheek 6-14bp={f:.3f} dots={int(d.sum())}", flush=True)
    for tag, r0, r1, hh in (("face", 90, 200, 330), ("jacket", 250, 310, 180)):
        vv = np.ones((r1 - r0, 300, 3), dtype=np.uint8) * 10
        vv[d[r0:r1] == 1] = [167, 139, 250]
        Image.fromarray(vv).resize((900, hh), Image.NEAREST).save(
            rf"C:\Users\Lenovo\Shivxng3010\banner_build\debug_{tag}_{name}.png")
print("done")
