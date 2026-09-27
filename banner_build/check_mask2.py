import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_fill_holes, label

img = Image.open(r"D:\Downloads\nccc.jpg")
cropped = img.crop((20, 25, 638, 725))
resized = cropped.resize((300, 340), Image.Resampling.LANCZOS)
print("new crop size:", cropped.size, "aspect:", cropped.size[0] / cropped.size[1])
arr = np.array(resized).astype(np.float32)

for thresh, k in [(25, 5), (14, 3), (14, 5)]:
    dist = np.linalg.norm(arr - np.array([252.0, 253.0, 254.0]), axis=-1)
    raw = dist > thresh
    closed = binary_closing(raw, structure=np.ones((k, k)))
    filled = binary_fill_holes(closed)
    lbl, num = label(filled)
    sizes = [np.sum(lbl == i) for i in range(1, num + 1)]
    big = np.argmax(sizes) + 1
    print(f"thresh={thresh} k={k}: comps={num} biggest={sizes[big-1]} ratio={sizes[big-1] / (300*340):.3f}")

# save overlay for best setting
dist = np.linalg.norm(arr - np.array([252.0, 253.0, 254.0]), axis=-1)
raw = dist > 14.0
closed = binary_closing(raw, structure=np.ones((3, 3)))
filled = binary_fill_holes(closed)
lbl, num = label(filled)
sizes = [np.sum(lbl == i) for i in range(1, num + 1)]
mask = (lbl == (np.argmax(sizes) + 1))
ov = np.array(resized).copy()
ov[~mask] = [255, 0, 0]
Image.fromarray(ov.astype(np.uint8)).save(r"C:\Users\Lenovo\Shivxng3010\banner_build\debug_overlay2.png")
mvis = np.zeros((340, 300, 3), dtype=np.uint8)
mvis[mask] = [0, 200, 0]
mvis[~mask] = [255, 0, 0]
Image.fromarray(mvis[160:280, 160:280]).resize((480, 480), Image.NEAREST).save(r"C:\Users\Lenovo\Shivxng3010\banner_build\debug_jaw_mask2.png")
resized.save(r"C:\Users\Lenovo\Shivxng3010\banner_build\debug_resized2.png")
print("saved")
