import numpy as np
from PIL import Image, ImageOps

img = Image.open(r"D:\Downloads\nccc.jpg")
crop = img.crop((20, 25, 638, 725))
g = np.array(ImageOps.grayscale(crop)).astype(np.float32)
H, W = g.shape
print("full-res crop:", W, H)

med = np.median(g, axis=1)
# detrend with wide median (window 61) to remove slow shading, keep periodic
trend = np.zeros_like(med)
for r in range(H):
    lo, hi = max(0, r - 30), min(H, r + 31)
    trend[r] = np.median(med[lo:hi])
resid = med - trend
print("residual std:", round(float(resid.std()), 2), "max abs:", round(float(np.abs(resid).max()), 2))

F = np.abs(np.fft.rfft(resid))
freqs = np.fft.rfftfreq(H, d=1.0)
# ignore very low freq (period > 200px) and DC
F[:max(1, int(H / 200))] = 0
peaks = np.argsort(F)[-6:]
for p in sorted(peaks):
    per = 1.0 / freqs[p] if freqs[p] > 0 else 0
    print(f"period {per:6.1f}px  amplitude {2*F[p]/H:5.2f}")
