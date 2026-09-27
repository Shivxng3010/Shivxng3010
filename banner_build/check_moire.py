import re
import numpy as np
from PIL import Image

with open(r"C:\Users\Lenovo\Shivxng3010\assets\dark.svg", encoding="utf-8") as f:
    svg = f.read()

loop = svg.split("portrait-loop")[1].split("travellers")[0]
ds = re.findall(r"<path d=([^/]*)/>", loop)

# rasterize runs at 4x supersample: grid 300x340 -> each cell 4px; run = filled bar
SS = 4
W, H = 300 * SS, 340 * SS
canvas = np.zeros((H, W), dtype=np.float32)
s, x0, y0 = 1.18, 45.0, 105.4
pat = re.compile(r"M([0-9.]+),([0-9.]+)h([0-9.]+)")
for d in ds:
    for m in pat.finditer(d):
        x, y, w = float(m.group(1)), float(m.group(2)), float(m.group(3))
        # run covers grid cols [c0, c0+n), row r -> pixel box
        c0 = (x - x0) / s
        r = (y - s / 2 - y0) / s
        n = w / s
        x_p0, x_p1 = int(round(c0 * SS)), int(round((c0 + n) * SS))
        y_p0, y_p1 = int(round(r * SS)), int(round((r + 1) * SS))
        canvas[y_p0:y_p1, x_p0:x_p1] = 1.0

full = Image.fromarray((canvas * 255).astype(np.uint8))
# jacket rows 250..310 -> pixels
for scale, tag in [(1.0, "s100"), (0.76, "s076"), (0.6, "s060")]:
    nw = int(round(1180 * scale))
    # emulate browser: scale whole-canvas equivalent - just scale the strip region proportionally
    strip = full.crop((0, 250 * SS, W, 310 * SS))
    w2 = int(round(strip.width * scale))
    h2 = int(round(strip.height * scale))
    small = strip.resize((w2, h2), Image.BILINEAR)
    small = small.resize((900, 180), Image.NEAREST)
    small.save(rf"C:\Users\Lenovo\Shivxng3010\banner_build\debug_moire_{tag}.png")
    a = np.array(small).astype(float)
    rs = a.mean(axis=1)
    dev = np.abs(rs[1:-1] - (rs[:-2] + rs[2:]) / 2).mean()
    print(tag, "strip row-jitter:", round(float(dev), 2))
print("done")
