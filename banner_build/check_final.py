import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

d = np.load(r"C:\Users\Lenovo\Shivxng3010\banner_build\portrait_dark_dither.npy")
l = np.load(r"C:\Users\Lenovo\Shivxng3010\banner_build\portrait_light_dither.npy")
print("DARK left cols 0-6 dots:", [int(d[:, c].sum()) for c in range(0, 7)])
print("DARK right cols 293-299 dots:", [int(d[:, c].sum()) for c in range(293, 300)])
print("LIGHT left cols 0-6 dots:", [int(l[:, c].sum()) for c in range(0, 7)])
print("LIGHT right cols 293-299 dots:", [int(l[:, c].sum()) for c in range(293, 300)])

for p in ["C:/Users/Lenovo/Shivxng3010/assets/dark.svg",
          "C:/Users/Lenovo/Shivxng3010/assets/light.svg"]:
    ET.parse(p)
    print("Valid XML:", p.split("/")[-1])

vis = np.ones((340, 300, 3), dtype=np.uint8) * 10
vis[d == 1] = [167, 139, 250]
Image.fromarray(vis).save(r"C:\Users\Lenovo\Shivxng3010\banner_build\debug_final_dark.png")
print("total dark dots:", int(d.sum()), "light dots:", int(l.sum()))
