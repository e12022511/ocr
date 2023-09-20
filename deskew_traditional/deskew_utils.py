import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.io import imread, imshow
from skimage.transform import rotate


def rotate_image(image, rotation_angle):
    return rotate(image, rotation_angle, cval=1)


def image_to_grayscale(image):
    return rgb2gray(image)


def apply_sobel(image):
    return sobel(image_to_grayscale(image))


def read_from_path(path):
    return imread(path)


def show_image(image):
    imshow(image)
    plt.axis('off')
    plt.show()
