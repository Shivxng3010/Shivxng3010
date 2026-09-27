import numpy as np
from PIL import Image

img = Image.open(r"D:\Downloads\nccc.jpg")
arr = np.array(img)
print("full-res left cols 38-58, rows 100-700: col means:")
for c in range(38, 59, 2):
    print(f"  x={c}: RGB={arr[100:700, c].mean(axis=0).round(0)}")
print("full-res right cols 592-616, rows 100-700: col means:")
for c in range(592, 617, 2):
    print(f"  x={c}: RGB={arr[100:700, c].mean(axis=0).round(0)}")

Image.fromarray(arr[25:675, 38:70]).resize((256, 5200 // 10), Image.NEAREST).save(
    r"C:\Users\Lenovo\Shivxng3010\banner_build\debug_fleft.png")
Image.fromarray(arr[25:675, 590:620]).resize((240, 5200 // 10), Image.NEAREST).save(
    r"C:\Users\Lenovo\Shivxng3010\banner_build\debug_fright.png")
print("saved")
