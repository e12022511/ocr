# Deskew OCR 

## Overview

This project focuses on improving OCR (Optical Character Recognition) accuracy by deskewing documents. 
Deskewing helps correct the tilt or skew in scanned or photographed documents, making text recognition more accurate.
The project has been done as a bachelor thesis at TU-Wien. 

## Features

- Automatic deskewing of input images/documents.
- Include basic traditional approaches such as Hough-Transform, Nearest Neighbor Classification or Projection Profile
- Deep learning model that deskews the document. 
- Metrics to measure ocr accuracy.
- Improved OCR accuracy for deskewed documents.

## Usage

### Docker Image

You can use the provided Docker image to run the deskew OCR tool easily.

#### Pull the Docker Image

To pull the Docker image, use the following command:

```bash
docker pull e12022511/deskew-ocr:version2
```

#### Run the docker image
```bash
docker run -v <path> deskew-ocr python ./main.py --input <path_to_img> --output <output_path> --deskew ht --ocrresult <optional:output_txt_of_ocrresult>
```

Options for --deskew flag: ht, knn, pp, ml


### Machine Learning Model download: https://drive.google.com/file/d/1mKIjtSZzSZhKrlNXVCZaJqWTroDHMZJz/view?usp=sharing
