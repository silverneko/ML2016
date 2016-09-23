from sys import argv
from PIL import Image

imageFilename = argv[1]

image = Image.open(imageFilename)
image = image.rotate(180)
image.save("ans2.png", "PNG")
