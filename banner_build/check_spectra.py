import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

img = Image.open(r"D:\Downloads\nccc.jpg")
cropped = img.crop((20, 25, 638, 725))
resized = cropped.resize((300, 340), Image.Resampling.LANCZOS)
gray = ImageOps.grayscale(resized)
gray = ImageOps.autocontrast(gray, cutoff=1)
gray = ImageEnhance.Contrast(gray).enhance(1.3)
gray = gray.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
g = np.array(gray).astype(np.float32)
d = np.load(r"C:\Users\Lenovo\Shivxng3010\banner_build\portrait_dark_dither.npy")

def spectrum(v, name):
    v = v - v.mean()
    F = np.abs(np.fft.rfft(v))
    H = len(v)
    tot = F[1:].sum() + 1e-9
    print(f"--- {name} (N={H}) top periods:")
    order = np.argsort(F)[-7:]
    for k in sorted(order):
        if k == 0:
            continue
        print(f"  period {H/k:6.1f} rows  share {F[k]/tot:.3f}")

gj = g[250:310, 60:240].mean(axis=1)
dj = d[250:310, 60:240].astype(float).mean(axis=1)
spectrum(gj, "INPUT gray jacket rows")
spectrum(dj, "DITHER jacket rows")

gf = g[150:190, 150:250].mean(axis=1)
df = d[150:190, 150:250].astype(float).mean(axis=1)
spectrum(gf, "INPUT gray cheek rows")
spectrum(df, "DITHER cheek rows")
