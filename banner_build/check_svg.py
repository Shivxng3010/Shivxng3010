import re
import numpy as np

with open(r"C:\Users\Lenovo\Shivxng3010\assets\dark.svg", encoding="utf-8") as f:
    svg = f.read()

loop = svg.split("portrait-loop")[1].split("travellers")[0]
ds = re.findall(r"<path d=([^/]*)/>", loop)
print("paths in loop layer:", len(ds))

grid = np.zeros((340, 300), dtype=np.uint8)
s, x0, y0 = 1.18, 45.0, 105.4
pat = re.compile(r"M([0-9.]+),([0-9.]+)h([0-9.]+)")
for d in ds:
    for m in pat.finditer(d):
        x, y, w = float(m.group(1)), float(m.group(2)), float(m.group(3))
        c0 = int(round((x - x0) / s))
        r = int(round((y - s / 2 - y0) / s))
        n = int(round(w / s))
        if 0 <= r < 340:
            grid[r, max(c0, 0):min(c0 + n, 300)] = 1

ref = np.load(r"C:\Users\Lenovo\Shivxng3010\banner_build\portrait_dark_dither.npy")
print("dots in SVG:", int(grid.sum()), "dots in npy:", int(ref.sum()))
print("correlation:", float((grid == ref).mean()))
print("missing in SVG:", int(((ref == 1) & (grid == 0)).sum()))
print("extra in SVG:", int(((ref == 0) & (grid == 1)).sum()))
