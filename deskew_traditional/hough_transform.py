import numpy as np
import skimage.transform

from deskew_traditional.deskew_method import DeskewMethod
from deskew_traditional.deskew_utils import *


class HoughTransform(DeskewMethod):
    def deskew(self, image):
        sobel_image = apply_sobel(image)
        theta_range = np.linspace(0, np.pi, 180)
        hspace, angles, distances = skimage.transform.hough_line(sobel_image, theta_range)
        _, angles, _ = skimage.transform.hough_line_peaks(hspace, angles, distances)

        median_angle = np.median(angles)
        rotation_angle = np.rad2deg(median_angle) - 90
        print(rotation_angle)
        return rotate_image(image, rotation_angle)
