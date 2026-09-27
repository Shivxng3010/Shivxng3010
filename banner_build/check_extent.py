import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_fill_holes, label

img = Image.open(r"D:\Downloads\nccc.jpg")
cropped = img.crop((20, 25, 638, 725))
resized = cropped.resize((300, 340), Image.Resampling.LANCZOS)
arr = np.array(resized).astype(np.float32)
dist = np.linalg.norm(arr - np.array([252.0, 253.0, 254.0]), axis=-1)
raw = dist > 14.0
closed = binary_closing(raw, structure=np.ones((3, 3)))
filled = binary_fill_holes(closed)
lbl, num = label(filled)
sizes = [np.sum(lbl == i) for i in range(1, num + 1)]
mask = (lbl == (np.argmax(sizes) + 1))

# per-row subject extent
print("rows where subject touches x<=11 or x>=290:")
for r in range(0, 340, 1):
    xs = np.where(mask[r])[0]
    if len(xs) == 0:
        continue
    if xs.min() <= 11 or xs.max() >= 290:
        print(f"  row {r}: xmin={xs.min()} xmax={xs.max()}")

# what do cols 291-295 look like (dark line or subject)?
print("mean RGB of cols 291-295 by row band:")
for r0, r1 in [(0, 40), (40, 120), (120, 200), (200, 280), (280, 340)]:
    print(f"  rows {r0}-{r1}: {arr[r0:r1, 291:296].mean(axis=(0,1)).round(1)}")
print("backdrop white ref rows 0-40 cols 100-200:", arr[0:40, 100:200].mean(axis=(0,1)).round(1))
