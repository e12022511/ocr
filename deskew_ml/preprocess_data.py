import os

import re
from shutil import copyfile


def preprocess(path):
    for filename in os.listdir(path):
        image_name = os.path.join(path, filename)
        angle = float(extract_angle(image_name))
        rounded_angle = round_angle(angle)
        path_including_angle = os.path.join(r'..\resources\sorted_data', f'{rounded_angle}')

        destination_directory = os.path.join(os.path.join(os.path.dirname(__file__), path_including_angle))
        os.makedirs(destination_directory, exist_ok=True)
        copyfile(os.path.join(path, filename), os.path.join(destination_directory, filename))


def extract_angle(filename):
    pattern = r'\[(.*?)\]'
    match = re.search(pattern, filename)
    if match:
        extracted_value = match.group(1)
        return extracted_value
    else:
        return 0


def round_angle(number):
    intnumber = int(number)
    decimal = number - intnumber
    if number > 0:
        if decimal < 0.5:
            if decimal < 0.25:
                return intnumber
            else:
                return intnumber + 0.5
        else:
            if decimal > 0.75:
                return intnumber + 1
            else:
                return intnumber + 0.5
    else:
        decimal = decimal * -1
        if decimal < 0.5:
            if decimal < 0.25:
                return intnumber
            else:
                return intnumber - 0.5
        else:
            if decimal > 0.75:
                return intnumber - 1
            else:
                return intnumber - 0.5


if __name__ == '__main__':
    preprocess(os.path.abspath(os.path.join(os.path.dirname(__file__), r'..\resources\dataset')))
