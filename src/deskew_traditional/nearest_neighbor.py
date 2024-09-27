import math

import cv2

from src.deskew_traditional.deskew_method import DeskewMethod
from src.utils.deskew_utils import *

# Global accumulator for connected components
accumulator = []


class NearestNeighbor(DeskewMethod):
    """
       Implements the Nearest Neighbor method for deskewing an image. It calculates the skew angle
       based on the positions of connected components and their nearest neighbors.

       Methods:
           deskew(image_path): Deskews the image located at the given file path.
       """
    def deskew(self, image_path):
        """
        Deskew the input image using the Nearest Neighbor method
        Parameters:
        - image_path (str): The file path of the input image
        Returns:
        - float: The estimated deskew angle in degrees
        Raises:
        - Any exceptions raised during image reading or processing.
        """
        image = read_from_path(image_path)
        find_connected_components(image)
        highest_angle_values = find_nearest_neighbors(15)
        rotation_angle = filter_outliers(highest_angle_values)
        return rotation_angle


def find_connected_components(image):
    """
       Identifies connected components in the given image using erosion and connected components analysis.

       Args:
           image (numpy.ndarray): Input image for which connected components are to be found.
       """
    eroded_image = erode_image(image)
    # invert image since cv2.connectedComponentsWithStats works better with inverted Images
    inverted_image = invert_image(eroded_image)
    number_of_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted_image, connectivity=8)
    color_connected_components(number_of_labels, labels, stats, True)
    fill_accumulator_with_components(number_of_labels, centroids)


def erode_image(image):
    """
       Applies erosion to the given image to enhance the connected components.

       Args:
           image (numpy.ndarray): Input image for erosion.

       Returns:
           numpy.ndarray: Eroded binary image.
       """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(binary, kernel, iterations=1)


def color_connected_components(number_of_labels, labels, stats, show_img):
    """
       Colors the connected components in the image for visualization and optionally displays it.

       Args:
           number_of_labels (int): Number of connected components found.
           labels (numpy.ndarray): Label matrix where each connected component has a unique label.
           stats (numpy.ndarray): Statistics about each component.
           show_img (bool): Whether to display the colored components.
       """
    if not show_img:
        return
    colors = np.zeros((number_of_labels, 3), dtype=np.uint8)
    # filter elements bigger than 1000
    small_component_mask = (stats[:, 4] <= 1000)
    colors[small_component_mask] = np.random.randint(0, 255, size=(np.sum(small_component_mask), 3), dtype=np.uint8)
    colored_components = colors[labels]
    show_image(colored_components)


def fill_accumulator_with_components(number_of_labels, centroids):
    """
        Fills the global accumulator with connected component centroids.

        Args:
            number_of_labels (int): The number of connected components.
            centroids (numpy.ndarray): The centroids of each connected component.
        """
    for label in range(1, number_of_labels):  # Skip background (label 0)
        centroid_x, centroid_y = centroids[label]
        accumulator.append(ConnectedComponent(label, centroid_x, centroid_y))


# performs the nearest neighbor algorithm and returns an array of
# highest angles found in the document between neighbors
def find_nearest_neighbors(number_of_highest_values):
    """
      Performs the nearest neighbor algorithm and returns the highest angles found between neighbors.

      Args:
          number_of_highest_values (int): The number of highest angle differences to return.

      Returns:
          list: A list of the highest angle differences between neighbors.
      """
    highest_values = []
    for component in accumulator:
        distances = []
        for potential_neighbor in accumulator:
            if component.label == potential_neighbor.label:
                continue
            if (math.fabs(component.y - potential_neighbor.y) < 0.75) and (
                    math.fabs(component.x - potential_neighbor.x) < 40):
                dist = euclidean_distance(component.x, component.y, potential_neighbor.x, potential_neighbor.y)
                distances.append(EuclidianDistance(potential_neighbor.label, dist))

        smallest_distances = find_smallest_distances(5, distances)
        smallest_diff_values = [math.fabs(angle_difference(component, accumulator[x.label - 1])) for x in
                                smallest_distances]
        highest_values.extend(smallest_diff_values)

    highest_values.sort(reverse=True)
    return highest_values[:number_of_highest_values]


def euclidean_distance(x1, y1, x2, y2):
    """
        Calculates the Euclidean distance between two points.

        Args:
            x1, y1 (float): Coordinates of the first point.
            x2, y2 (float): Coordinates of the second point.

        Returns:
            float: The Euclidean distance between the two points.
        """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def find_smallest_distances(number, distances):
    """
       Finds the smallest distances from the given list of distances.

       Args:
           number (int): The number of smallest distances to find.
           distances (list): List of EuclideanDistance objects.

       Returns:
           list: The smallest Euclidean distances.
       """
    sorted_array = sorted(distances, key=lambda x: x.distance)
    return sorted_array[:number]


# computes the angle between two points, if a line would be drawn through both
def angle_difference(component1, component2):
    """
        Computes the angle between two connected components.

        Args:
            component1 (ConnectedComponent): The first component.
            component2 (ConnectedComponent): The second component.

        Returns:
            float: The angle between the two components in degrees.
        """
    delta_x = component2.x - component1.x
    delta_y = component2.y - component1.y
    m = delta_y / delta_x
    rho = math.atan(m)
    return math.degrees(rho)


def filter_outliers(elements):
    """
        Filters out outliers from a list of elements by finding elements with similar values.

        Args:
            elements (list): List of values to filter.

        Returns:
            float: The filtered value.
        """
    for element in elements:
        count = 0
        for other_element in elements:
            if element == other_element:
                continue  # Skip the element itself
            if math.fabs(element - other_element) <= 0.5:
                count += 1
        if count > 1:
            return element
    return elements[0]


class ConnectedComponent:
    """
       Represents a connected component with a label and coordinates.

       Attributes:
           label (int): The label of the component.
           x (float): The x-coordinate of the centroid.
           y (float): The y-coordinate of the centroid.
       """
    def __init__(self, label, x, y):
        self.label = label
        self.x = x
        self.y = y

    def __str__(self):
        return f"ConnectedComponent(label= {self.label}, centroid_x= {self.x}, centroid_y= {self.y})"


class EuclidianDistance:
    """
       Represents the Euclidean distance between two components.

       Attributes:
           label (int): The label of the component.
           distance (float): The Euclidean distance.
       """
    def __init__(self, label, distance):
        self.label = label
        self.distance = distance

    def __str__(self):
        return f"EuclidianDistance(label= {self.label}, distance= {self.distance})"
