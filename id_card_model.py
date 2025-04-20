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
# Normalize Persian text (e.g., replace 'آ' with 'ا')
def normalize_text(text):
    if isinstance(text, str):
        text = text.replace('آ', 'ا')  # Replace specific Persian character
        normalizer = Normalizer()
        return normalizer.normalize(text)  # Normalize text
    return text

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

    preprocessed = preprocess_image(cropped_image)  # Preprocess image
    reader = easyocr.Reader(OCR_LANGUAGES if is_first_image else ['fa', 'en'], gpu=False, verbose=False)  # Initialize OCR
    results = reader.readtext(preprocessed, **OCR_PARAMS)  # Run OCR
    print(f"Debug: Detected {len(results)} text boxes")

    # Sort and merge text boxes
    sorted_results = sorted(results, key=lambda x: (x[0][0][1] + x[0][2][1]) / 2)  # Sort by y-coordinate
    merged_boxes, merged_texts, used = [], [], [False] * len(sorted_results)
    X_THRESHOLD, Y_THRESHOLD = 50, 10  # Thresholds for merging

    for i in range(len(sorted_results)):
        if used[i]: continue
        group, group_texts = [sorted_results[i][0]], [sorted_results[i][1]]  # Start group
        y_center_i, x_right_i = (group[0][0][1] + group[0][2][1]) / 2, group[0][1][0]
        for j in range(len(sorted_results)):
            if used[j] or i == j: continue
            bbox_j = sorted_results[j][0]
            y_center_j, x_left_j = (bbox_j[0][1] + bbox_j[2][1]) / 2, bbox_j[0][0]
            # Merge if close enough
            if abs(y_center_i - y_center_j) < Y_THRESHOLD and abs(x_right_i - x_left_j) < X_THRESHOLD:
                group.append(bbox_j)
                group_texts.append(sorted_results[j][1])
                used[j] = True
        merged_bbox = group[0]
        for bbox in group[1:]:  # Merge bounding boxes
            merged_bbox = [
                [min(merged_bbox[0][0], bbox[0][0]), min(merged_bbox[0][1], bbox[0][1])],
                [max(merged_bbox[1][0], bbox[1][0]), min(merged_bbox[1][1], bbox[1][1])],
                [max(merged_bbox[2][0], bbox[2][0]), max(merged_bbox[2][1], bbox[2][1])],
                [min(merged_bbox[3][0], bbox[3][0]), max(merged_bbox[3][1], bbox[3][1])]
            ]
        merged_boxes.append(merged_bbox)
        merged_texts.append(' '.join(group_texts))  # Join texts
        used[i] = True
    text_detections = []
    for idx, bbox in enumerate(merged_boxes):
        top_left = tuple(map(int, bbox[0]))  # Top-left corner
        bottom_right = tuple(map(int, bbox[2]))  # Bottom-right corner
        padding_x, padding_y = 20, max(2, int((bottom_right[1] - top_left[1]) / 4))  # Padding
        top_left_padded = (max(0, top_left[0] - padding_x), max(0, top_left[1] - padding_y))
        bottom_right_padded = (min(W_crop, bottom_right[0] + padding_x), min(H_crop, bottom_right[1] + padding_y))

        # Crop region for OCR
        cropped_region = cropped_image[top_left_padded[1]:bottom_right_padded[1], top_left_padded[0]:bottom_right_padded[0]]
        preprocessed_region = preprocess_image(cropped_region)  # Preprocess region

        # Use Hezar model for first image, EasyOCR for second
        if is_first_image:
            ocr_model = Model.load("hezarai/crnn-fa-printed-96-long")  # Load Hezar OCR model
            pil_image = Image.fromarray(cv2.cvtColor(preprocessed_region, cv2.COLOR_GRAY2RGB))
            text = ocr_model.predict(pil_image)[0]["text"]  # Run OCR
        else:
            text = reader.readtext(preprocessed_region, **OCR_PARAMS)[0][1] if reader.readtext(preprocessed_region, **OCR_PARAMS) else merged_texts[idx]

        text = normalize_text(text.strip())  # Normalize extracted text
        cv2.rectangle(image_with_boxes, top_left_padded, bottom_right_padded, (0, 255, 0), 2)  # Draw box
        text_detections.append({"text": text})

    # Save annotated image
    output_path = os.path.join(OUTPUT_DIR, f"annotated_{'first' if is_first_image else 'second'}.jpg")
    cv2.imwrite(output_path, image_with_boxes)
    return {"text_detections": text_detections, "annotated_path": output_path}            

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