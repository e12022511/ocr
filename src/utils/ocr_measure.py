import Levenshtein
from torchmetrics.functional.text import char_error_rate


def calculate_ocr_accuracy(ground_truth, ocr_output, method):
    """
    Calculate OCR accuracy based on the specified method.
    Parameters:
    - ground_truth (list): List of strings representing the ground truth.
    - ocr_output (list): List of strings representing the OCR output.
    - method (str): The method to calculate accuracy. Supported values: "LEV" or "CER".
    Returns:
    - float: The calculated OCR accuracy.
    Raises:
    - ValueError: If an invalid method is provided.
    Note:
    - The "LEV" method calculates accuracy using Levenshtein distance.
    - The "CER" method calculates accuracy using Character Error Rate.
     """
    if method == "LEV":
        return Levenshtein.distance(' '.join(ground_truth), ' '.join(ocr_output))
    if method == "CER":
        return char_error_rate(ground_truth, ocr_output)
    else:
        raise ValueError(f"Invalid method '{method}'. Supported methods are 'Levenshtein' and 'CER'.")


def txt_to_string(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines() if line.strip()]


if __name__ == '__main__':
    test_number = 4
    ground_truth_string = txt_to_string(f"../resources/ocr_tests/{test_number}/ground_truth.txt")
    ht_string = txt_to_string(f"../resources/ocr_tests/{test_number}/ht.txt")
    pp_string = txt_to_string(f"../resources/ocr_tests/{test_number}/pp.txt")
    knn_string = txt_to_string(f"../resources/ocr_tests/{test_number}/knn.txt")
    ml_string = txt_to_string(f"../resources/ocr_tests/{test_number}/ml.txt")
    skewed_string = txt_to_string(f"../resources/ocr_tests/{test_number}/skewed.txt")

    compare_method = "CER"

    ht = calculate_ocr_accuracy(ground_truth_string, ht_string, compare_method)
    pp = calculate_ocr_accuracy(ground_truth_string, pp_string, compare_method)
    knn = calculate_ocr_accuracy(ground_truth_string, knn_string, compare_method)
    ml = calculate_ocr_accuracy(ground_truth_string, ml_string, compare_method)
    skewed = calculate_ocr_accuracy(ground_truth_string, skewed_string, compare_method)

    print(f"ht: {ht}")
    print(f"pp: {pp}")
    print(f"knn: {knn}")
    print(f"ml: {ml}")
    print(f"skewed: {skewed}")
