import filetype
import pytesseract
from pdf2image import convert_from_path

from deskew_ml.load_model import MlMethod
from deskew_traditional.hough_transform import HoughTransform
from deskew_traditional.nearest_neighbor import NearestNeighbor
from deskew_traditional.projection_profile import ProjectionProfile
from utils.deskew_utils import *

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
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for i in range(len(images)):
            extracted_text = pytesseract.image_to_string(images[i])
            output_file.write(extracted_text + '\n')



def pdf_to_text(pdf_path):
    images = convert_from_path(pdf_path)
    image_to_text(images, 'output.txt')


if __name__ == '__main__':
    path = 'resources/ocr_tests/1/skewed.png'
    img = read_from_path(path)
    image_to_text([img], 'resources/ocr_tests/1/skewed.txt')

    ht = HoughTransform()
    angle = ht.deskew(path)
    print(f"Hough: {angle}")
    corrected_image = rotate_image(img, angle)
    image_to_text([corrected_image], 'resources/ocr_tests/1/ht.txt')

    pp = ProjectionProfile()
    angle = pp.deskew(path)
    print(f"PP: {angle}")
    corrected_image = rotate_image(img, angle)
    image_to_text([corrected_image], 'resources/ocr_tests/1/pp.txt')

    knn = NearestNeighbor()
    angle = knn.deskew(path)
    print(f"KNN: {angle}")
    corrected_image = rotate_image(img, angle)
    image_to_text([corrected_image], 'resources/ocr_tests/1/knn.txt')

    ml = MlMethod()
    angle = ml.deskew(path)
    print(f"ML: {angle}")
    corrected_image = rotate_image(img, angle)
    image_to_text([corrected_image], 'resources/ocr_tests/1/ml.txt')

