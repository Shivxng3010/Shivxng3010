import numpy as np
from PIL import Image

img = Image.open(r"D:\Downloads\nccc.jpg")
cropped = img.crop((20, 25, 638, 725))
resized = cropped.resize((300, 340), Image.Resampling.LANCZOS)
arr = np.array(resized).astype(np.float32)
lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

print("dark-fraction (lum<100) per column, rows 10-330:")
for c in list(range(0, 14)) + list(range(280, 300)):
    frac = float((lum[10:330, c] < 100).mean())
    print(f"  col {c}: {frac:.2f}")

# true subject extent excluding candidate frame cols (drop cols 0-4 and 284-299, recompute)
print("subject x-extent per row band (frame cols excluded):")
for r0, r1 in [(30, 60), (60, 120), (120, 200), (200, 260), (260, 340)]:
    sub = arr[r0:r1, 5:284]
    dd = np.linalg.norm(sub - np.array([252.0, 253.0, 254.0]), axis=-1)
    xs = np.where((dd > 14).any(axis=0))[0]
    if len(xs):
        print(f"  rows {r0}-{r1}: content xmin={xs.min()+5} xmax={xs.max()+5}")
    else:
        print(f"  rows {r0}-{r1}: no content")
