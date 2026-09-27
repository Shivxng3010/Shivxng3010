import numpy as np
from PIL import Image

img = Image.open(r"D:\Downloads\nccc.jpg")
crop = img.crop((20, 25, 638, 725)).convert("L")
g = np.array(crop).astype(np.float32)
H, W = g.shape
print("full-res crop:", W, H)

dark = g < 140
# bright-white neighbors left and right (median over 6px windows, offset 2)
leftw = np.zeros_like(dark, dtype=bool)
rightw = np.zeros_like(dark, dtype=bool)
for c in range(W):
    l0, l1 = max(0, c - 8), max(0, c - 2)
    r0, r1 = min(W, c + 3), min(W, c + 9)
    if l1 > l0:
        leftw[:, c] = np.median(g[:, l0:l1], axis=1) > 215
    if r1 > r0:
        rightw[:, c] = np.median(g[:, r0:r1], axis=1) > 215

cand = dark & leftw & rightw
# vertical run length per column (longest consecutive run)
for c in range(W):
    col = cand[:, c]
    if not col.any():
        continue
    d = np.diff(np.concatenate([[0], col.view(np.int8), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    longest = int((ends - starts).max()) if len(starts) else 0
    total = int(col.sum())
    if longest >= 30:
        print(f"col {c}: longest run={longest}px total={total}px")

# top rows: check for horizontal dark band (border remnant)
for r in [0, 1, 2, 3, 4, 5, 8, 12]:
    print(f"row {r}: darkfrac={(g[r] < 140).mean():.2f} meanlum={g[r].mean():.0f}")
