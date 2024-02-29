import filetype
import pytesseract
from pdf2image import convert_from_path

from src.deskew_ml.load_model import MlMethod
from src.deskew_traditional.hough_transform import HoughTransform
from src.deskew_traditional.nearest_neighbor import NearestNeighbor
from src.deskew_traditional.projection_profile import ProjectionProfile
from src.utils.deskew_utils import *

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def extract_text(file_path):
    file_type = filetype.guess(file_path).extension
    if file_type == 'pdf':
        pdf_to_text(file_path)
    elif file_type == 'png' or file_type == 'jpeg':
        images = [read_from_path(file_path)]
        image_to_text(images, 'output.txt')
    else:
        print("File can only be of Type pdf/png/jpeg")


def image_to_text(images, output_file_path):
    save = True
    if save:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for i in range(len(images)):
                extracted_text = pytesseract.image_to_string(images[i])
                output_file.write(extracted_text + '\n')


def pdf_to_text(pdf_path):
    images = convert_from_path(pdf_path)
    image_to_text(images, 'output.txt')


if __name__ == '__main__':
    test_number = 5

    path = f'resources/ocr_tests/{test_number}/no_skews.png'
    img = read_from_path(path)
    image_to_text([img], f'resources/ocr_tests/{test_number}/ground_truth.txt')

    path = f'resources/ocr_tests/{test_number}/skewed.png'
    img = read_from_path(path)
    image_to_text([img], f'resources/ocr_tests/{test_number}/skewed.txt')

    ht = HoughTransform()
    angle = ht.deskew(path)
    print(f"Hough: {angle}")
    corrected_image = rotate_image(img, angle)
    show_image(corrected_image)
    image_to_text([corrected_image], f'resources/ocr_tests/{test_number}/ht.txt')

    pp = ProjectionProfile()
    angle = pp.deskew(path)
    print(f"PP: {angle}")
    corrected_image = rotate_image(img, angle)
    show_image(corrected_image)
    image_to_text([corrected_image], f'resources/ocr_tests/{test_number}/pp.txt')

    knn = NearestNeighbor()
    angle = knn.deskew(path)
    print(f"KNN: {angle}")
    corrected_image = rotate_image(img, angle)
    show_image(corrected_image)
    image_to_text([corrected_image], f'resources/ocr_tests/{test_number}/knn.txt')

    ml = MlMethod()
    angle = ml.deskew(path)
    print(f"ML: {angle}")
    corrected_image = rotate_image(img, angle)
    show_image(corrected_image)
    image_to_text([corrected_image], f'resources/ocr_tests/{test_number}/ml.txt')
