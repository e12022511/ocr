from collections import Counter

import skimage.transform

from src.deskew_traditional.deskew_method import DeskewMethod
from src.utils.deskew_utils import *


class HoughTransform(DeskewMethod):

    def deskew(self, image_path):
        """
        Deskew the input image using the Hough Transform method
        Parameters:
        - image_path (str): The file path of the input image
        Returns:
        - float: The estimated deskew angle in degrees
        Raises:
        - Any exceptions raised during image reading or processing.
        """
        image = read_from_path(image_path)
        sobel_image = apply_sobel(image)
        show_image(sobel_image)
        theta_range = np.deg2rad(np.arange(-5, 5, 0.1))
        hspace, angles, distances = skimage.transform.hough_line(sobel_image, theta_range)
        _, angles, _ = skimage.transform.hough_line_peaks(hspace, angles, distances)

        angles_in_degree = [np.rad2deg(angle) for angle in angles]
        counter = Counter(angles_in_degree)
        max_value = max(counter.values())
        most_common_values = []
        for value, count in counter.items():
            if count == max_value:
                most_common_values.append(value)
        angle = np.mean(most_common_values)
        if angle > 0 and 90 < angle < 180:
            angle = angle - 90
        if 180 > angle > 270:
            angle = angle - 180
        if angle > 270:
            angle = angle - 270
        return angle
