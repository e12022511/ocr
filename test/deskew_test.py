import unittest

from src.deskew_traditional.hough_transform import HoughTransform
from src.deskew_traditional.nearest_neighbor import NearestNeighbor
from src.deskew_traditional.projection_profile import ProjectionProfile


class TestProjectionProfile(unittest.TestCase):

    def test_pp(self):
        image_path = "../resources/testdata/doc.png"

        deskew_method = ProjectionProfile()
        result = deskew_method.deskew(image_path)

        self.assertGreaterEqual(result, 1.5, msg="Value should be greater than or equal to 1.5")
        self.assertLessEqual(result, 3, msg="Value should be less than or equal to 2.5")

    def test_ht(self):
        image_path = "../resources/testdata/doc.png"

        deskew_method = HoughTransform()
        result = deskew_method.deskew(image_path)

        self.assertGreaterEqual(result, 1.5, msg="Value should be greater than or equal to 1.5")
        self.assertLessEqual(result, 3, msg="Value should be less than or equal to 2.5")

    def test_knn(self):
        image_path = "../resources/testdata/doc.png"

        deskew_method = NearestNeighbor()
        result = deskew_method.deskew(image_path)

        self.assertGreaterEqual(result, 1.5, msg="Value should be greater than or equal to 1.5")
        self.assertLessEqual(result, 3, msg="Value should be less than or equal to 2.5")


if __name__ == '__main__':
    unittest.main()
