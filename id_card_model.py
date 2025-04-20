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

# Process image with YOLO model to detect objects
def process_image(image, model):
    result = model(image)  # Run YOLO detection
    if len(result[0].boxes) == 0:  # Check for detections
        print("No objects detected")
        return None
    detections = []
    for box in result[0].boxes.data:  # Iterate through detected boxes
        x1, y1, x2, y2, conf, cls = box.tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert coordinates to integers

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert coordinates to integers
        cropped_object = image[y1:y2, x1:x2]  # Crop detected object
        class_name = model.names[int(cls)]  # Get class name
        detections.append({"class": class_name, "cropped_object": cropped_object})
    return {"detections": detections}

# Load and preprocess image from bytes
def load_image(image_data, is_first_image=True):
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)  # Decode image
    if image is None:
        print("Failed to decode image")
        return None
    height, width = image.shape[:2]
    # Rotate image if needed based on orientation
    if is_first_image and height > width:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif not is_first_image and width > height:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image

def detect_and_extract_text(image, is_first_image=True):
    H, W = image.shape[:2]  # Get image dimensions
    # Crop image based on whether it's the first or second image
    if is_first_image:
        crop_y_start = int(H * CROP_RATIO_FIRST)
        crop_y_end = int(W * SHIFT_PERCENTAGE_RIGHT)
        crop_x_start = int(W * SHIFT_PERCENTAGE_LEFT)
        cropped_image = image[crop_y_start:H, crop_x_start:W - crop_y_end]
    else:
        crop_y_start = 0
        crop_y_end = int(H * SHIFT_PERCENTAGE_RIGHT)
        crop_x_start = int(W * 0.5)
        cropped_image = image[crop_y_start:crop_y_end, crop_x_start:W]
    image_with_boxes = cropped_image.copy()  # Copy for drawing boxes
    H_crop, W_crop = cropped_image.shape[:2]  # Get cropped dimensions

    # Preprocess image for OCR
    def preprocess_image(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        if is_first_image:
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)  # Apply bilateral filter
            denoised = cv2.fastNlMeansDenoising(bilateral, None, 10, 7, 21)  # Denoise
            contrasted = cv2.convertScaleAbs(denoised, alpha=1.2, beta=10)  # Adjust contrast
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # Sharpening kernel
            sharpened = cv2.filter2D(contrasted, -1, kernel)  # Apply sharpening
            _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Threshold
          #  kernel_close = np.ones((2, 2), np.uint8)
           # closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)  # Morphological close
           # kernel_open = np.ones((1, 1), np.uint8)
            #cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)  # Morphological open
           # kernel_dilate = np.ones((1, 1), np.uint8)
            #final = cv2.dilate(cleaned, kernel_dilate, iterations=1)  # Dilate
            return thresh
        return cv2.GaussianBlur(gray, (3, 3), 0)  # Apply Gaussian blur for second image
    
    
def process_id_card(first_image_data, second_image_data,job_output_dir,log_file):
    logger = setup_logging(log_file) #Initialize logger 
    if not MODEL_YOLO or MODEL_POLO:
        return {"status": "error", "message": "No ID card detected", "data": None}
    
    # Load images
    first_image = load_image(first_image_data, True)
    second_image = load_image(second_image_data, False)
    if first_image is None or second_image is None:
        return {"status": "error", "message": "Image loading failed", "data": None}
    
    # Detect ID card in first image
    polo_result = process_image(first_image, MODEL_POLO)
    if not polo_result:
        return {"status": "error", "message": "No ID card detected", "data": None}
    #croped id card from image 
    first_cropped = polo_result["detections"][0]["cropped_object"]  # Get cropped ID card

    # Detect person in cropped image
    person_result = process_image(first_cropped, MODEL_YOLO)
    person_path = None
    if person_result:
        for detection in person_result["detections"]:
            if detection["class"] == "person":
                person_path = os.path.join(job_output_dir, "person.jpg")
                cv2.imwrite(person_path, detection["cropped_object"])  # Save person image
                break

    # Extract text from images
    all_texts = []
    text_result_first = detect_and_extract_text(first_cropped, True)
    for detection in text_result_first["text_detections"]:
        text = convert_persian_to_english_numbers(detection["text"])
        all_texts.append(text)
        print(f"Debug: First image text: {text}")

    text_result_second = detect_and_extract_text(second_image, False)
    for detection in text_result_second["text_detections"]:
        text = convert_persian_to_english_numbers(detection["text"])
        all_texts.append(text)
        print(f"Debug: Second image text: {text}")