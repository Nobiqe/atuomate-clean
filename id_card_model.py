# Import libraries for image processing, OCR, logging, and text normalization
import os
import cv2  # OpenCV for image processing
import logging  # For logging messages
import re  # For regular expressions
import numpy as np  # For numerical operations
from ultralytics import YOLO  # YOLO models for object detection
import easyocr  # OCR library for text extraction
from hezar.models import Model  # Hezar model for Persian OCR
from PIL import Image  # For image conversion
from hazm import Normalizer  # For Persian text normalization

# Configuration constants
OUTPUT_DIR = "output_images"  # Directory for output images
LOG_FILE = "text_log.log"  # Log file for extracted text
CROP_RATIO_FIRST = 0.20  # Crop ratio for first image
SHIFT_PERCENTAGE_RIGHT = 0.15  # Right shift percentage for cropping
SHIFT_PERCENTAGE_LEFT = 0.5  # Left shift percentage for cropping
OCR_LANGUAGES = ['fa']  # OCR language (Persian)
OCR_PARAMS = {  # Parameters for EasyOCR
    "decoder": "beamsearch", "beamWidth": 15, "batch_size": 2, "workers": 2,
    "contrast_ths": 0.1, "adjust_contrast": 0.7, "text_threshold": 0.8,
    "low_text": 0.2, "link_threshold": 0.1, "width_ths": 6.0,
    "ycenter_ths": 0.5, "mag_ratio": 10.0, "slope_ths": 0.3, "detail": 1
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLO models once at startup
print("Loading models...")
try:
    MODEL_POLO = YOLO("polov11.pt")  # Load Polo YOLO model
    MODEL_YOLO = YOLO("yolov8x.pt")  # Load YOLOv8x model
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    MODEL_POLO = MODEL_YOLO = None  # Set models to None if loading fails

# Set up logging to file
def setup_logging(log_file):
    logger = logging.getLogger('IDCardProcessor')  # Create logger
    logger.setLevel(logging.INFO)  # Set logging level
    logger.handlers = []  # Clear existing handlers
    handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')  # File handler
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')  # Simple log format
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def process_id_card(first_image_data, second_image_data,job_output_dir,log_file):
    logger = setup_logging(log_file) #Initialize logger 
    if not MODEL_YOLO or MODEL_POLO:
        return {"status": "error", "message": "No ID card detected", "data": None}
    
    # Load images
    first_image = load_image(first_image_data, True)
    second_image = load_image(second_image_data, False)
    if first_image is None or second_image is None:
        return {"status": "error", "message": "Image loading failed", "data": None}
    
    first_cropped = polo_result["detections"][0]["cropped_object"]  # Get cropped ID card

