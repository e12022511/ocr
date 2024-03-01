import unittest

from src.utils.ocr_measure import calculate_ocr_accuracy


class TestOCRMeasure(unittest.TestCase):
    def test_same_strings_LEV(self):
        result = calculate_ocr_accuracy(["kitten"], ["kitten"], "LEV")
        self.assertEqual(result, 0, "Levenshtein distance between identical strings should be 0")

    def test_insertion_LEV(self):
        result = calculate_ocr_accuracy(["kitten"], ["kittten"], "LEV")
        self.assertEqual(result, 1, "Levenshtein distance for insertion should be 1")

    def test_deletion_LEV(self):
        result = calculate_ocr_accuracy(["kitten"], ["kiten"], "LEV")
        self.assertEqual(result, 1, "Levenshtein distance for deletion should be 1")

    def test_substitution_LEV(self):
        result = calculate_ocr_accuracy(["kitten"], ["kotten"], "LEV")
        self.assertEqual(result, 1, "Levenshtein distance for substitution should be 1")

    def test_complex_case_LEV(self):
        result = calculate_ocr_accuracy(["sitting"], ["kitten"], "LEV")
        self.assertEqual(result, 3, "Levenshtein distance for a complex case should be 3")

    def test_multiple_lines_LEV(self):
        result = calculate_ocr_accuracy(["sitting", "kitten", "sitting"], ["kitten", "kotten", "sitting"], "LEV")
        self.assertEqual(result, 4, "Levenshtein distance for a complex case should be 3")

    def test_same_strings_CER(self):
        result = calculate_ocr_accuracy(["kitten"], ["kitten"], "CER")
        self.assertEqual(round(float(result), 4), 0, "Levenshtein distance between identical strings should be 0")

    def test_insertion_CER(self):
        result = calculate_ocr_accuracy(["kitten"], ["kittten"], "CER")
        self.assertEqual(round(float(result), 4), 0.1429, "Levenshtein distance for insertion should be 1")

    def test_deletion_CER(self):
        result = calculate_ocr_accuracy(["kitten"], ["kiten"], "CER")
        self.assertEqual(round(float(result), 4), 0.2, "Levenshtein distance for deletion should be 1")

    def test_substitution_CER(self):
        result = calculate_ocr_accuracy(["kitten"], ["kotten"], "CER")
        self.assertEqual(round(float(result), 4), 0.1667, "Levenshtein distance for substitution should be 1")

    def test_complex_case_CER(self):
        result = calculate_ocr_accuracy(["sitting"], ["kitten"], "CER")
        self.assertEqual(round(float(result), 4), 0.5, "Levenshtein distance for a complex case should be 3")

    def test_multiple_lines_CER(self):
        result = calculate_ocr_accuracy(["sitting", "kitten", "sitting"], ["kitten", "kotten", "sitting"], "CER")
        self.assertEqual(round(float(result), 4), 0.1905, "Levenshtein distance for a complex case should be 3")


if __name__ == '__main__':
    unittest.main()
