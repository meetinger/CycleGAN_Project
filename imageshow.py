from PIL import Image
import numpy as np
import matplotlib.pyplot as plt1

def load_image(imgname) :
    img = Image.open(imgname)
    img.load()
    return img
def imshow(image, title=None):
    plt1.imshow(image)
    if title is not None:
        plt1.title(title)
    plt1.pause(0.001)
def show(imname, title):
    imshow(load_image(imname), title)
    plt1.show()