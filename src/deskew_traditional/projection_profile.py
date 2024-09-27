from src.deskew_traditional.deskew_method import DeskewMethod
from src.utils.deskew_utils import *


def calculate_pixel_per_line(image):
    """
        Calculates the sum of pixel values along each row of the image. This can be used for
        histogram analysis in the Projection Profile method for deskewing.

        Args:
            image (numpy.ndarray): Input grayscale or binary image as a NumPy array.

        Returns:
            numpy.ndarray: A 1D array representing the sum of pixel values for each row.
        """
    return np.sum(image, axis=1)


class ProjectionProfile(DeskewMethod):
    """
       Implements the Projection Profile method for deskewing an image. It rotates the image
       at various angles and calculates a histogram of pixel values to find the best angle.

       Attributes:
           histogram (dict): A dictionary that stores the median pixel sum for each rotation angle.

       Methods:
           deskew(image_path): Deskews the image located at the given file path using the
                               Projection Profile method.
       """

    histogram = {}

    def deskew(self, image_path):
        """
        Deskew the input image using the Projection Profile method
        Parameters:
        - image_path (str): The file path of the input image
        Returns:
        - float: The estimated deskew angle in degrees
        Raises:
        - Any exceptions raised during image reading or processing.
        """
        image = read_from_path(image_path)
        grey_image = image_to_grayscale(image)

        for angle in range(-10, 10):
            rotated_image = rotate_image(grey_image, angle)
            pixel_array = calculate_pixel_per_line(rotated_image)
            self.histogram[angle] = np.median(pixel_array)

        max_key_value_pair = max(self.histogram.items(), key=lambda x: x[1])
        max_angle = max_key_value_pair[0]
        return max_angle

