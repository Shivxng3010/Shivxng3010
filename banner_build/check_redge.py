import numpy as np
from PIL import Image

img = Image.open(r"D:\Downloads\nccc.jpg")
cropped = img.crop((20, 25, 638, 725))
resized = cropped.resize((300, 340), Image.Resampling.LANCZOS)
arr = np.array(resized)
Image.fromarray(arr[30:210, 255:300]).resize((360, 1440), Image.NEAREST).save(
    r"C:\Users\Lenovo\Shivxng3010\banner_build\debug_redge_photo.png")
Image.fromarray(arr[30:210, 0:40]).resize((320, 1440), Image.NEAREST).save(
    r"C:\Users\Lenovo\Shivxng3010\banner_build\debug_ledge_photo.png")
print("saved")
