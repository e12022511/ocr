from enum import Enum
from typing import Optional

import typer
import logging

from src.deskew_ml.load_model import MlMethod
from src.deskew_traditional.hough_transform import HoughTransform
from src.deskew_traditional.nearest_neighbor import NearestNeighbor
from src.deskew_traditional.projection_profile import ProjectionProfile
from src.utils.deskew_utils import *
from src.utils.text_extraction import image_to_text

app = typer.Typer(help="OCR Improvement by deskewing")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DeskewVariant(str, Enum):
    ht = "ht"
    knn = "knn"
    pp = "pp"
    ml = "ml"


@app.command()
def generate(input: str = None,
             output: str = None,
             deskew: DeskewVariant = None,
             ocrresult: Optional[str] = None):
    logging.info(f"Deskewing document with ${deskew}")
    if deskew == DeskewVariant.ht:
        ht = HoughTransform()
        angle = ht.deskew(input)
    elif deskew == DeskewVariant.knn:
        knn = NearestNeighbor()
        angle = knn.deskew(input)
    elif deskew == DeskewVariant.pp:
        pp = ProjectionProfile()
        angle = pp.deskew(input)
    elif deskew == DeskewVariant.ml:
        ml = MlMethod()
        angle = ml.deskew(input)
    else:
        angle = 0
    img = read_from_path(input)
    rotated_image = rotate_image(img, angle)
    save_image(rotated_image, output)

    if ocrresult:
        image_to_text([rotated_image], ocrresult)


if __name__ == "__main__":
    app()
