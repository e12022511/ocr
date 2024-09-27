import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageOps, Image
from skimage.color import rgb2gray
from skimage.filters import sobel, gaussian
from skimage.io import imshow, imsave
from skimage.transform import rotate


def rotate_image(image, rotation_angle):
    """
       Rotates the given image by the specified angle.

       Args:
           image (numpy.ndarray): Input image in NumPy array format.
           rotation_angle (float): The angle by which to rotate the image, in degrees.

       Returns:
           numpy.ndarray: The rotated image as an 8-bit unsigned integer array.
       """
    rotated_image = rotate(image, rotation_angle, cval=1)
    return (rotated_image * 255).astype(np.uint8)


def image_to_grayscale(image):
    """
       Converts an RGB image to grayscale. If the image is already single-channel, it returns the image as-is.

       Args:
           image (numpy.ndarray): Input image in NumPy array format.

       Returns:
           numpy.ndarray: Grayscale image.
       """
    if image.shape[-1] == 3:
        return rgb2gray(image)
    else:
        return image


def invert_image(image):
    """
        Inverts the given image by flipping pixel values.

        Args:
            image (numpy.ndarray): Input image in NumPy array format.

        Returns:
            numpy.ndarray: Inverted image.
        """
    return ~image


def apply_sobel(image):
    """
        Applies the Sobel filter to the given image to detect edges.

        Args:
            image (numpy.ndarray): Input image in NumPy array format.

        Returns:
            numpy.ndarray: Image with Sobel edge detection applied.
        """
    return sobel(image_to_grayscale(image))


def apply_gaussian(image):
    """
        Applies a Gaussian filter to the given image to smooth it.

        Args:
            image (numpy.ndarray): Input image in NumPy array format.

        Returns:
            numpy.ndarray: Smoothed image with the Gaussian filter applied.
        """
    return gaussian(image_to_grayscale(image))


def read_from_path(path):
    """
       Reads an image from the specified file path, automatically handling EXIF data orientation.

       Args:
           path (str): Path to the image file.

       Returns:
           numpy.ndarray: Image read and converted into NumPy array format.
       """
    image = Image.open(path)

    # Correct orientation using EXIF data (if available)
    image = ImageOps.exif_transpose(image)

    # Convert to NumPy array for use with skimage
    return np.array(image)


def show_image(image):
    """
       Displays the given image using matplotlib.

       Args:
           image (numpy.ndarray): Input image in NumPy array format.
       """
    imshow(image)
    plt.axis('off')
    plt.show()


def zoom_in_image(image_path, crop_percentage):
    """
        Crops the image by zooming in according to the specified crop percentage.

        Args:
            image_path (str): Path to the image file.
            crop_percentage (float): Percentage to crop from each side (between 0 and 0.5).

        Returns:
            numpy.ndarray: Cropped image.
        """
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)

    width, height = image.size

    left = int(width * crop_percentage)
    upper = int(height * crop_percentage)
    right = width - left
    lower = height - upper

    # Crop the image using the defined zoom_area
    cropped_image = image.crop((left, upper, right, lower))

    return np.array(cropped_image)


def save_image(image_data, file_path):
    """
        Saves the given image data to the specified file path.

        Args:
            image_data (numpy.ndarray): Image data to be saved.
            file_path (str): Path where the image should be saved.

        Raises:
            Exception: If there is an error during saving.
        """
    try:
        imsave(file_path, image_data)
        print(f"Image saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving image: {e}")
