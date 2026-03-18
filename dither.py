from PIL import Image

file_path = "cat.jpg"
img = Image.open(file_path)
print(f"Opened: {file_path}")
print(f"Size: {img.size[0]}x{img.size[1]}")
print(f"Mode: {img.mode}")

scale = 0.5

gray = img.convert("L")
if scale != 1.0:
    new_size = (int(img.width * scale), int(img.height * scale))
    gray = gray.resize(new_size, Image.LANCZOS)
    print(f"Scaled to: {gray.size[0]}x{gray.size[1]}")
gray.save("cat_gray.png")
print("Saved cat_gray.png")

# Floyd-Steinberg dithering
pixels = list(gray.get_flattened_data())
width, height = gray.size
# Work with floats so error accumulates accurately
buf = [float(p) for p in pixels]

for y in range(height):
    for x in range(width):
        old = buf[y * width + x]
        new = 255.0 if old >= 128 else 0.0
        buf[y * width + x] = new
        err = old - new

        if x + 1 < width:
            buf[y * width + (x + 1)] += err * 7 / 16
        if y + 1 < height:
            if x - 1 >= 0:
                buf[(y + 1) * width + (x - 1)] += err * 3 / 16
            buf[(y + 1) * width + x] += err * 5 / 16
            if x + 1 < width:
                buf[(y + 1) * width + (x + 1)] += err * 1 / 16

dithered = Image.new("L", (width, height))
dithered.putdata([int(max(0, min(255, p))) for p in buf])
dithered.save("cat_dithered.png")
print("Saved cat_dithered.png")
