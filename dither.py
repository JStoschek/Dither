from PIL import Image

file_path = "cat.jpg"
img = Image.open(file_path)
print(f"Opened: {file_path}")
print(f"Size: {img.size[0]}x{img.size[1]}")
print(f"Mode: {img.mode}")
