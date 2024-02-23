from deskew_traditional.deskew_method import DeskewMethod
from utils.deskew_utils import *


def calculate_pixel_per_line(image):
    return np.sum(image, axis=1)


class ProjectionProfile(DeskewMethod):
    histogram = {}

    def deskew(self, image_path):
        image = read_from_path(image_path)
        grey_image = image_to_grayscale(image)

        for angle in range(-10, 10):
            rotated_image = rotate_image(grey_image, angle)
            pixel_array = calculate_pixel_per_line(rotated_image)
            self.histogram[angle] = np.median(pixel_array)

        max_key_value_pair = max(self.histogram.items(), key=lambda x: x[1])
        max_angle = max_key_value_pair[0]
        return max_angle

