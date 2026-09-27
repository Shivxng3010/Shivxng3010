import numpy as np
from PIL import Image

d = np.load(r"C:\Users\Lenovo\Shivxng3010\banner_build\portrait_dark_dither.npy")
for tag, c0, c1 in (("left", 0, 24), ("right", 276, 300)):
    vv = np.ones((340, c1 - c0, 3), dtype=np.uint8) * 10
    vv[d[:, c0:c1] == 1] = [167, 139, 250]
    Image.fromarray(vv).resize(((c1 - c0) * 20, 680), Image.NEAREST).save(
        rf"C:\Users\Lenovo\Shivxng3010\banner_build\debug_edge2_{tag}.png")
# row distribution of edge dots: which rows?
for c0, c1, name in [(1, 7, "left"), (293, 299, "right")]:
    rows = np.where(d[:, c0:c1].sum(axis=1) > 0)[0]
    print(name, "rows with edge dots:", len(rows), "ranges:", end=" ")
    if len(rows):
        print(f"{rows.min()}-{rows.max()}", "first10:", rows[:10].tolist(), "last10:", rows[-10:].tolist())
print("saved")
