import filetype
import pytesseract
from pdf2image import convert_from_path

from deskew_traditional.deskew_utils import *
from deskew_traditional.hough_transform import HoughTransform
from deskew_traditional.projection_profile import ProjectionProfile
from deskew_traditional.nearest_neighbor import NearestNeighbor

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def extract_text(file_path):
    file_type = filetype.guess(file_path).extension
    if file_type == 'pdf':
        pdf_to_text(file_path)
    elif file_type == 'png' or file_type == 'jpeg':
        images = [read_from_path(file_path)]
        image_to_text(images)
    else:
        print("File can only be of Type pdf/png/jpeg")


def image_to_text(images):
    for i in range(len(images)):
        extracted_text = pytesseract.image_to_string(images[i])
        print(extracted_text)


def pdf_to_text(pdf_path):
    images = convert_from_path(pdf_path)
    image_to_text(images)


if __name__ == '__main__':
    path = 'resources/PDFs/doc.png'
    img = read_from_path(path)

    ht = HoughTransform()
    pp = ProjectionProfile()
    knn = NearestNeighbor()

    corrected_image = knn.deskew(img)
    show_image(corrected_image)
