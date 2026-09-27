import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from scipy.ndimage import binary_closing, binary_fill_holes, label, gaussian_filter1d

img = Image.open(r"D:\Downloads\nccc.jpg")
cropped = img.crop((20, 25, 638, 725))
resized = cropped.resize((300, 340), Image.Resampling.LANCZOS)
base_gray = np.array(ImageOps.grayscale(resized)).astype(np.float32)

arr_rgb = np.array(resized).astype(np.float32)
dist = np.linalg.norm(arr_rgb - np.array([252.0, 253.0, 254.0]), axis=-1)
raw = dist > 14.0
closed = binary_closing(raw, structure=np.ones((3, 3)))
filled = binary_fill_holes(closed)
lbl, num = label(filled)
sizes = [np.sum(lbl == i) for i in range(1, num + 1)]
mask = (lbl == (np.argmax(sizes) + 1))

def run(gray_in, serpentine=True):
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

def bandpower(d, r0, r1, c0, c1):
    sub = d[r0:r1, c0:c1].astype(float)
    rs = sub.mean(axis=1)
    rs = rs - rs.mean()
    F = np.abs(np.fft.rfft(rs))
    tot = F.sum() + 1e-9
    H = len(rs)
    band = 0.0
    for per in (2, 3, 4, 5, 6):
        k = int(round(H / per))
        if 0 < k < len(F):
            band += F[k]
    return band / tot

variants = {
    "base": base_gray,
    "vblur06": gaussian_filter1d(base_gray, 0.6, axis=0),
    "vblur08": gaussian_filter1d(base_gray, 0.8, axis=0),
}
for name, g in variants.items():
    d = run(g)
    j = bandpower(d, 250, 310, 60, 240)
    f = bandpower(d, 150, 190, 150, 250)
    print(f"{name}: jacket bp={j:.3f} cheek bp={f:.3f} dots={int(d.sum())}", flush=True)
    if name in ("base", "vblur06"):
        for tag, r0, r1, hh in (("face", 90, 200, 330), ("jacket", 250, 310, 180)):
            vv = np.ones((r1 - r0, 300, 3), dtype=np.uint8) * 10
            vv[d[r0:r1] == 1] = [167, 139, 250]
            Image.fromarray(vv).resize((900, hh), Image.NEAREST).save(
                rf"C:\Users\Lenovo\Shivxng3010\banner_build\debug_{tag}_{name}.png")
print("done")
