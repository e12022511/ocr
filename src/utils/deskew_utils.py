import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel, gaussian
from skimage.io import imread, imshow, imsave
from skimage.transform import rotate


def rotate_image(image, rotation_angle):
        rotated_image = rotate(image, rotation_angle, cval=1)
        return (rotated_image * 255).astype(np.uint8)



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


def save_image(image_data, file_path):
    """
    Save an image to a specified file path.

    Parameters:
    - image_data: Image data as a PIL Image object.
    - file_path: Path where the image will be saved.

    Returns:
    - None
    """
    try:
        imsave(file_path, image_data)
        print(f"Image saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving image: {e}")
