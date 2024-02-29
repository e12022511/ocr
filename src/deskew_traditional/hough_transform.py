from collections import Counter

import skimage.transform

from src.deskew_traditional.deskew_method import DeskewMethod
from src.utils.deskew_utils import *


class HoughTransform(DeskewMethod):

    def deskew(self, image_path):
        image = read_from_path(image_path)
        sobel_image = apply_sobel(image)
        show_image(sobel_image)
        theta_range = np.deg2rad(np.arange(0.1, 180.0))
        hspace, angles, distances = skimage.transform.hough_line(sobel_image, theta_range)
        _, angles, _ = skimage.transform.hough_line_peaks(hspace, angles, distances)

        angles_in_degree = [np.rad2deg(angle) for angle in angles]

        counter = Counter(angles_in_degree)
        max_value = max(counter.values())
        most_common_values = []
        for value, count in counter.items():
            if count == max_value:
                most_common_values.append(value)
        return np.mean(most_common_values) - 90
