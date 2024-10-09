# OCR

## Installing Required Packages:
'''!apt-get install tesseract-ocr -y
!pip install pytesseract opencv-python pillow'''

_apt-get install tesseract-ocr -y: Installs the Tesseract OCR engine on the system.
pip install pytesseract opencv-python pillow: Installs Python libraries:
pytesseract: A wrapper for Tesseract that allows you to call Tesseract from Python.
opencv-python: OpenCV library for image processing.
Pillow: A fork of the Python Imaging Library (PIL) for image manipulation._

## Importing Libraries:

'''import cv2
import pytesseract
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files'''

_Imports necessary libraries for image processing, OCR, and visualization._

## Preprocessing the Image:

'''def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    processed_image = cv2.medianBlur(binary_image, 3)
    return processed_image'''

_cv2.cvtColor: Converts the input image to grayscale, which simplifies the data for further processing._
_cv2.threshold: Applies a binary threshold to create a black-and-white image, enhancing text visibility._

## Extracting Text:

'''def extract_text(image):
    text = pytesseract.image_to_string(image)
    return text'''
    
_Uses Tesseract to extract text from the preprocessed image and returns the extracted text._

## Running the OCR Pipeline:

'''def ocr_pipeline(image_path):
    image = cv2.imread(image_path)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    
    preprocessed_image = preprocess_image(image)
    extracted_text = extract_text(preprocessed_image)
    print("Extracted Text:")
    print(extracted_text)'''

_cv2.imread: Reads the image from the specified path.
Displays the original image using Matplotlib.
Preprocesses the image and then extracts and prints the text._

## Setting Tesseract Command Path:
    
'''pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
!sudo apt install tesseract-ocr'''

Specifies the path to the Tesseract executable, which may vary depending on the operating system.
This command is repeated and might be unnecessary since Tesseract should already be installed.

## Running the Pipeline on Uploaded Image:

'''image_path = list(uploaded.keys())[0]  # Using the uploaded image's path
ocr_pipeline(image_path)'''

_This part assumes that an image file has been uploaded (in a Colab environment, for example). It retrieves the first uploaded image's path and runs the OCR pipeline._



