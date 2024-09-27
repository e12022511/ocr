import os

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
    """
        Extract text from a file based on its type (PDF or image).

        Args:
            file_path (str): The path to the input file (PDF or image).

        Raises:
            ValueError: If the file type is not supported.
        """
    file_type = filetype.guess(file_path).extension
    if file_type == 'pdf':
        pdf_to_text(file_path)
    elif file_type == 'png' or file_type == 'jpeg':
        images = [read_from_path(file_path)]
        image_to_text(images, 'output.txt')
    else:
        print("File can only be of Type pdf/png/jpeg")


def image_to_text_no_write(images):
    """
      Extract text from a list of images without writing to a file.

      Args:
          images (list): List of images as NumPy arrays.

      Returns:
          list: List of extracted text lines.
      """
    for i in range(len(images)):
        extracted_text = pytesseract.image_to_string(images[i])
        result_string = '\n'.join(line for line in extracted_text.splitlines() if line.strip())
        # output_file.write(result_string)
        return result_string.splitlines()


def image_to_text(images, output_file_path):
    """
        Extract text from a list of images and write the result to a specified output file.

        Args:
            images (list): List of images as NumPy arrays.
            output_file_path (str): The file path where the extracted text will be saved.
        """
    with open(output_file_path, 'w', encoding='utf-8', newline=os.linesep) as output_file:
        for i in range(len(images)):
            extracted_text = pytesseract.image_to_string(images[i])
            result_string = '\n'.join(line for line in extracted_text.splitlines() if line.strip())
            # output_file.write(result_string)
            return result_string


def pdf_to_text(pdf_path):
    """
       Convert a PDF file to text by first converting it to images.

       Args:
           pdf_path (str): The path to the PDF file to be converted.
       """
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
