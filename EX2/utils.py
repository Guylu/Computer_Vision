import os

import numpy as np
import skimage.color
from PIL import Image
from IPython.display import display, Markdown


__all__ = ['imread', 'imwrite', 'imshow', 'rgb2grey']


def imread(path):
    image = Image.open(path).convert(mode='RGB')
    image = np.array(image).astype(np.float32) / 255
    return image


def imwrite(path, image):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    Image.fromarray(image).save(path)


def imshow(image, title=None):
    if title:
        display(Markdown("### %s" % title))
    display(Image.fromarray(image))

def rgb2grey(images):
    return skimage.color.rgb2grey(images)
