import numpy as np
from PIL import Image, ImageOps

resized = Image.open(r"C:\Users\Lenovo\Shivxng3010\banner_build\debug_resized2.png")
d = np.load(r"C:\Users\Lenovo\Shivxng3010\banner_build\portrait_dark_dither.npy")

# face band region rows 90-200: upscale 3x strips
face = np.array(resized.convert("L"))[90:200, :]
Image.fromarray(face).resize((900, 330), Image.NEAREST).save(
    r"C:\Users\Lenovo\Shivxng3010\banner_build\debug_face_gray.png")
dv = np.ones((110, 300, 3), dtype=np.uint8) * 10
dv[d[90:200] == 1] = [167, 139, 250]
Image.fromarray(dv).resize((900, 330), Image.NEAREST).save(
    r"C:\Users\Lenovo\Shivxng3010\banner_build\debug_face_dither.png")

# jacket band region rows 240-320
jk = np.array(resized.convert("L"))[240:320, :]
Image.fromarray(jk).resize((900, 240), Image.NEAREST).save(
    r"C:\Users\Lenovo\Shivxng3010\banner_build\debug_jacket_gray.png")
jv = np.ones((80, 300, 3), dtype=np.uint8) * 10
jv[d[240:320] == 1] = [167, 139, 250]
Image.fromarray(jv).resize((900, 240), Image.NEAREST).save(
    r"C:\Users\Lenovo\Shivxng3010\banner_build\debug_jacket_dither.png")
print("saved strips")
