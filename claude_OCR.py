import hydra
import torch
import easyocr
import cv2
import pandas as pd
import os
from datetime import datetime
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from difflib import SequenceMatcher
import re

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Initialize DataFrame for logging
df = pd.DataFrame(columns=['License_Plate', 'Entry_Time'])
recorded_plates = {}  # Dictionary to store plates and their timestamps

def similar(a, b):
    """Check if two plate numbers are similar based on sequence matching."""
    return SequenceMatcher(None, a, b).ratio() > 0.8

def clean_plate(plate):
    """Clean and validate the license plate text."""
    plate = ''.join(c for c in plate if c.isalnum()).upper()
    
    # Basic Indian license plate format validation
    pattern = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z0-9]{4}$')
    if not pattern.match(plate):
        return None
    return plate

def is_duplicate(plate, current_time):
    """Check if this is a duplicate detection within a short time window."""
    if plate in recorded_plates:
        last_time = datetime.strptime(recorded_plates[plate], '%Y-%m-%d %H:%M:%S')
        time_diff = (current_time - last_time).total_seconds()
        return time_diff < 5  # Ignore detections within 5 seconds
    return False

def process_plate(plate_text):
    """Process the detected license plate."""
    try:
        # Clean and validate plate
        cleaned_plate = clean_plate(plate_text)
        if not cleaned_plate:
            return False
        
        current_time = datetime.now()
        time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Check for similar existing plates
        for existing_plate in list(recorded_plates.keys()):
            if similar(cleaned_plate, existing_plate):
                if is_duplicate(existing_plate, current_time):
                    return False
                cleaned_plate = existing_plate  # Use the existing plate number
                break
        
        # Only add if it's not a duplicate detection
        if not is_duplicate(cleaned_plate, current_time):
            new_entry = pd.DataFrame({
                'License_Plate': [cleaned_plate],
                'Entry_Time': [time_str]
            })
            global df
            df = pd.concat([df, new_entry], ignore_index=True)
            recorded_plates[cleaned_plate] = time_str
            df.to_csv('vehicle_log.csv', index=False)
            print(f"Recorded new plate: {cleaned_plate}")
            return True
            
    except Exception as e:
        print(f"Error processing plate: {e}")
        
    return False

def perform_ocr(img, bbox):
    """Perform OCR on the detected region."""
    try:
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add padding to the region
        padding = 5
        h, w = img.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        plate_img = img[y1:y2, x1:x2]
        
        # Preprocessing for OCR
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=0)  # Increase contrast
        
        # OCR the image
        results = reader.readtext(gray)
        
        if results:
            # Get the result with highest confidence
            best_result = max(results, key=lambda x: x[2])
            if best_result[2] > 0.45:  # Confidence threshold
                return best_result[1]
            
    except Exception as e:
        print(f"OCR Error: {e}")
    return ""

class DetectionPredictor(BasePredictor):
    def get_annotator(self, img):
        """Returns the annotator for the image."""
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        """Preprocess image for model inference."""
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()
        img /= 255
        return img

    def postprocess(self, preds, img, orig_img):
        """Postprocess predictions from the model."""
        preds = ops.non_max_suppression(preds,
                                      self.args.conf,
                                      self.args.iou,
                                      agnostic=self.args.agnostic_nms,
                                      max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        """Write results after detection and processing."""
        p, im, im0 = batch
        log_string = ""
        
        if len(im.shape) == 3:
            im = im[None]
        
        self.seen += 1
        im0 = im0.copy()
        
        if self.webcam:
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        
        if len(det) == 0:
            return log_string

        for *xyxy, conf, cls in reversed(det):
            if conf < 0.3:  # Confidence threshold
                continue
                
            # Perform OCR on the detected region
            plate_text = perform_ocr(im0, xyxy)
            
            if plate_text:
                if process_plate(plate_text):
                    # Draw the box and label on the image
                    cleaned_plate = clean_plate(plate_text)
                    if cleaned_plate:
                        label = f"{cleaned_plate}"
                        self.annotator.box_label(xyxy, label, color=colors(int(cls), True))

            if self.args.save_crop:
                save_one_box(xyxy, im0.copy(), 
                           file=self.save_dir / 'crops' / self.model.names[int(cls)] / f'{p.stem}.jpg',
                           BGR=True)

        return log_string

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    """Main function to run the prediction."""
    # Set confidence threshold
    cfg.conf = 0.3
    
    # Set model path - REPLACE THIS WITH YOUR MODEL PATH
    cfg.model = "best.pt"  # Use your trained model path
    
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    
    predictor = DetectionPredictor(cfg)
    predictor()

if __name__ == "__main__":
    predict()
