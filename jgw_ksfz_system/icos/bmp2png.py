from PIL import Image, ImageTk
img = Image.open("./ico.jpeg")
print(img.mode)
img = img.convert("RGBA")
img.save('./ico.png')

