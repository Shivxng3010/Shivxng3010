import numpy as np
d = np.load(r"C:\Users\Lenovo\Shivxng3010\banner_build\portrait_dark_dither.npy")
sub = d[90:200, 40:260].astype(float)
rs = sub.mean(axis=1)
print("face dither row density rows 90-199 (cols 40-260):")
prev = None
for i, v in enumerate(rs):
    flag = ""
    if prev is not None and abs(v - prev) > 0.09:
        flag = "  <-- JUMP"
    print(f"{90+i}: {v:.2f}{flag}")
    prev = v
