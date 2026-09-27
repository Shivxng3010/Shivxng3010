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

arr_rgb = np.array(resized).astype(np.float32)
dist = np.linalg.norm(arr_rgb - np.array([252.0, 253.0, 254.0]), axis=-1)
raw = dist > 14.0
closed = binary_closing(raw, structure=np.ones((3, 3)))
filled = binary_fill_holes(closed)
lbl, num = label(filled)
sizes = [np.sum(lbl == i) for i in range(1, num + 1)]
mask = (lbl == (np.argmax(sizes) + 1))

base = np.array(gray).astype(np.float32) / 255.0
base[~mask] = 0.0

def dither(buf_in, serpentine):
    buf = buf_in.copy()
    h, w = buf.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for r in range(h):
        ltr = True if not serpentine else (r % 2 == 0)
        cols = range(w) if ltr else range(w - 1, -1, -1)
        for c in cols:
            if not mask[r, c]:
                buf[r, c] = 0.0
                continue
            old = buf[r, c]
            nv = 1.0 if old >= 0.5 else 0.0
            out[r, c] = int(nv)
            err = old - nv
            if ltr:
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

d_serp = dither(base, True)
d_ltr = dither(base, False)

def bandpower(d, r0, r1, c0, c1):
    sub = d[r0:r1, c0:c1].astype(float)
    rs = sub.mean(axis=1)
    rs = rs - rs.mean()
    F = np.abs(np.fft.rfft(rs))
    tot = F.sum() + 1e-9
    # power at period 2..6 rows (freq H/k)
    H = len(rs)
    band = 0.0
    for per in (2, 3, 4, 5, 6):
        k = int(round(H / per))
        if 0 < k < len(F):
            band += F[k]
    return band / tot

for name, d in [("serpentine", d_serp), ("left-to-right", d_ltr)]:
    j = bandpower(d, 250, 310, 60, 240)   # jacket smooth area
    f = bandpower(d, 150, 190, 150, 250)  # cheek smooth area
    print(f"{name}: jacket 2-6px band-power={j:.3f} cheek band-power={f:.3f} dots={int(d.sum())}")

for name, d in [("serp", d_serp), ("ltr", d_ltr)]:
    jv = np.ones((60, 300, 3), dtype=np.uint8) * 10
    jv[d[250:310] == 1] = [167, 139, 250]
    Image.fromarray(jv).resize((900, 180), Image.NEAREST).save(
        rf"C:\Users\Lenovo\Shivxng3010\banner_build\debug_scan_{name}.png")
print("saved")
