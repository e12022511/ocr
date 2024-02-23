import Levenshtein
from torchmetrics.functional.text import char_error_rate


def calculate_ocr_accuracy(ground_truth, ocr_output, method):
    if method == "Levenshtein":
        return Levenshtein.distance(ground_truth, ocr_output)
    if method == "CER":
        return char_error_rate(ground_truth, ocr_output)
    else:
        raise ValueError(f"Invalid method '{method}'. Supported methods are 'Levenshtein' and 'CER'.")


def txt_to_string(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]
        return ' '.join(lines)


if __name__ == '__main__':
    ground_truth_string = txt_to_string("../resources/ocr_tests/1/ground_truth.txt")
    ht_string = txt_to_string("../resources/ocr_tests/1/ht.txt")
    pp_string = txt_to_string("../resources/ocr_tests/1/pp.txt")
    knn_string = txt_to_string("../resources/ocr_tests/1/knn.txt")
    ml_string = txt_to_string("../resources/ocr_tests/1/ml.txt")
    skewed_string = txt_to_string("../resources/ocr_tests/1/skewed.txt")

    compare_method = "Levenshtein"

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
