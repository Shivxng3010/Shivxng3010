from PIL import Image
img = Image.open(r"D:\Downloads\nccc.jpg")
print("quantization tables:", img.quantization)
