import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import sobel, gaussian
from skimage.io import imread, imshow
from skimage.transform import rotate


def rotate_image(image, rotation_angle):
    return rotate(image, rotation_angle, cval=1)


def image_to_grayscale(image):
    if image.shape[-1] == 3:
        return rgb2gray(image)
    else:
        return image


def invert_image(image):
    return ~image


def apply_sobel(image):
    return sobel(image_to_grayscale(image))


def apply_gaussian(image):
    return gaussian(image_to_grayscale(image))


def read_from_path(path):
    return imread(path)


def show_image(image):
    imshow(image)
    plt.axis('off')
    plt.show()
