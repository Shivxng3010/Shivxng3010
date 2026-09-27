import numpy as np
from PIL import Image

d = np.load(r"C:\Users\Lenovo\Shivxng3010\banner_build\portrait_dark_dither.npy")
print("LEFT edge dots per column (rows 0-339):")
for c in range(0, 12):
    print(f"  col {c}: {int(d[:, c].sum())} dots")
print("RIGHT edge dots per column (rows 0-339):")
for c in range(288, 300):
    print(f"  col {c}: {int(d[:, c].sum())} dots")

# where vertically? row ranges for the busy edge columns
for c in [0, 1, 2, 293, 294, 295, 296]:
    rows = np.where(d[:, c] == 1)[0]
    if len(rows):
        print(f"col {c}: {len(rows)} dots rows {rows.min()}-{rows.max()}")

# light dither edges too
l = np.load(r"C:\Users\Lenovo\Shivxng3010\banner_build\portrait_light_dither.npy")
print("LIGHT left cols 0-5:", [int(l[:, c].sum()) for c in range(0, 6)])
print("LIGHT right cols 294-299:", [int(l[:, c].sum()) for c in range(294, 300)])

# visualize edge strips of current dark dither
for tag, c0, c1 in (("left", 0, 20), ("right", 280, 300)):
    vv = np.ones((340, c1 - c0, 3), dtype=np.uint8) * 10
    vv[d[:, c0:c1] == 1] = [167, 139, 250]
    Image.fromarray(vv).resize(((c1 - c0) * 24, 680), Image.NEAREST).save(
        rf"C:\Users\Lenovo\Shivxng3010\banner_build\debug_edge_{tag}.png")
print("saved edge strips")
