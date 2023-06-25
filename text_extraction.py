import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from pdf2image import convert_from_path
import filetype

def extract_text(file_path):
    file_type = filetype.guess(file_path).extension
    if (file_type == 'pdf'):
        pdf_to_text(file_path)
    elif (file_type == 'png' or file_type == 'jpeg'):
        images = []
        images.append(get_image_from_path(file_path))
        image_to_text(images)

def image_to_text(images):
    for i in range(len(images)):
        extracted_text = pytesseract.image_to_string(images[i])
        print(extracted_text)

def pdf_to_text(pdf_path):
    images = convert_from_path(pdf_path)
    image_to_text(images)

def get_image_from_path(file_path):
    try:
        image = Image.open(file_path)
        return image
    except IOError:
        print("Unable to open image file:", file_path)


if __name__ == '__main__':
    path = 'resources/PDFs/sample.pdf'
    extract_text(path)

