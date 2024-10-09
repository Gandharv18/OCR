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


## Information about Packages:

cv2 - cv2 is a Python module from OpenCV (Open Source Computer Vision Library), which is widely used for computer vision and image processing tasks. OpenCV provides tools for analyzing images and videos, including tasks like object detection, image transformations, edge detection, video analysis, and more. The cv2 module specifically refers to OpenCV's Python bindings.

pytesseract - pytessreact is a Python wrapper for Tesseract-OCR, an optical character recognition (OCR) engine that extracts text from images. It allows Python developers to easily use Tesseract OCR functionalities for detecting and extracting text from images.
Tesseract was initially developed by Hewlett-Packard and is now maintained by Google. It supports a wide range of languages and works well with printed text, handwritten notes, and text from various image formats.

PIT - PIT (also known as PIT Mutation Testing or PITest) is a tool used for mutation testing in Java. Mutation testing is a method to evaluate the quality of unit tests by introducing small changes (mutations) to the code, then checking whether the existing tests are able to detect and fail because of these changes. This process helps measure how effective the tests are at catching defects in the codebase.

NumPy - NumPy (short for Numerical Python) is a powerful open-source library in Python that is primarily used for performing mathematical, scientific, and engineering computations. It provides support for large, multi-dimensional arrays and matrices, along with a wide variety of mathematical functions to operate on these arrays. NumPy is widely used in data science, machine learning, and computational tasks.

matplotlib - matplotlib.pyplot is a submodule of the Matplotlib library in Python, widely used for creating static, animated, and interactive visualizations. It provides an interface similar to MATLAB, making it simple to create plots, charts, and graphs in Python. pyplot is often the starting point for most data visualizations and provides a convenient way to control the figure, axes, and plot styles.


## Summary 

Programming Languages: Python
Libraries: OpenCV (for image processing), pytesseract (for OCR), Pillow (for image manipulation), Matplotlib (for visualization)
OCR Engine: Tesseract OCR

This OCR system can accurately extract text from images and handle various text styles and conditions.
It also extract from Handwritten notes . printed documents etc.
Save the extracted text in Plain text format.
Provide visualization of the original image alongside the extracted text for verification.

## Limitations 
Image should be clear and bold then and only then it will extract information from image or photo.


