import cv2
import easyocr
import csv
import re
from difflib import SequenceMatcher
from datetime import datetime

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# CSV file to store detected plates and their entry/exit times
csv_file = 'license_plate_log.csv'

# Load video
video_path = 'mycarplate.mp4'
cap = cv2.VideoCapture(video_path)

# Common OCR misread character mapping
char_mapping = {
    '0': ['O', 'Q', 'D'],
    '1': ['I', 'L'],
    '2': ['Z'],
    '5': ['S'],
    '8': ['B'],
    'O': ['0', 'Q'],
    'Z': ['2'],
    'Q': ['O', '0']
}

# Initialize variables to store plate data
detected_plates = {}

# Function to correct common OCR character misreads
def correct_plate_chars(plate):
    corrected_plate = ""
    for char in plate:
        if char in char_mapping:
            corrected_plate += char_mapping[char][0]  # Pick the most likely match
        else:
            corrected_plate += char
    return corrected_plate

# Validate license plate format with regex
def validate_plate_format(plate):
    pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$'
    return re.match(pattern, plate)

# Post-process the detected plate to correct OCR misreads and validate the format
def post_process_plate(plate):
    plate = plate.upper()
    corrected_plate = correct_plate_chars(plate)
    
    if validate_plate_format(corrected_plate):
        return corrected_plate
    else:
        return plate  # If invalid, return the original uncorrected plate

# Check if two plates are similar using a similarity threshold (50%)
def is_similar_plate(plate1, plate2, threshold=0.5):
    return SequenceMatcher(None, plate1, plate2).ratio() > threshold

# Function to log license plate with entry and exit times
def log_plate(plate, current_time, mode="entry"):
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if mode == "entry":
            writer.writerow([plate, current_time, ""])
        elif mode == "exit":
            writer.writerow([plate, "", current_time])

# Read frames from video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection (assuming you have YOLO code integrated here)
    # Detected bounding box around license plate is cropped as `plate_image`
    
    # Simulate a detected bounding box for license plate (replace with YOLO's output)
    plate_image = frame

    # Use EasyOCR to recognize text in the detected license plate
    result = reader.readtext(plate_image, detail=0)
    
    if result:
        detected_plate = result[0].replace(" ", "")  # Clean up extra spaces
        processed_plate = post_process_plate(detected_plate)  # Correct misreads

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Check if the plate is already detected (within threshold)
        similar_found = False
        for existing_plate in detected_plates.keys():
            if is_similar_plate(processed_plate, existing_plate):
                similar_found = True
                # Update exit time if plate exits after some time
                detected_plates[existing_plate]['exit'] = current_time
                log_plate(existing_plate, current_time, mode="exit")
                break
        
        if not similar_found:
            # Add the new plate to the dictionary and log its entry
            detected_plates[processed_plate] = {'entry': current_time, 'exit': None}
            log_plate(processed_plate, current_time, mode="entry")

# Release the video capture object
cap.release()

# Closing any open windows (if using OpenCV windows)
cv2.destroyAllWindows()
