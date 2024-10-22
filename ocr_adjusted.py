import hydra
import torch
import easyocr
import cv2
import pandas as pd
from datetime import datetime, timedelta
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from difflib import SequenceMatcher  # For fuzzy matching
import re  # For regex-based plate validation

# Initialize DataFrame
vehicle_data = pd.DataFrame(columns=['License Plate', 'Entry Time', 'Exit Time'])
recorded_plates = set()  # Global set to track recorded plates (avoid duplicates)
cooldown_time = timedelta(seconds=10)  # Cooldown period to prevent logging duplicate entries

# Helper function to validate plate format (Indian format as an example)
def validate_plate_format(plate):
    pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$'
    return re.match(pattern, plate) is not None

def getOCR(im, coors):
    """Extracts text (license plate) using EasyOCR from the region of interest."""
    x, y, w, h = int(coors[0]), int(coors[1]), int(coors[2]), int(coors[3])
    im = im[y:h, x:w]
    conf = 0.4  # Increased confidence threshold

    # Image preprocessing for better OCR accuracy
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    try:
        results = reader.readtext(thresh)
        ocr = ""

        for result in results:
            if len(result[1]) > 6 and result[2] > conf:  # Filter results by length and confidence
                ocr = result[1]
        return str(ocr)
    except Exception as e:
        print(f"Error applying OCR: {e}")
        return ""

def normalize_plate(plate):
    """Normalize the license plate by removing spaces, special characters, and converting to uppercase."""
    plate = ''.join(e for e in plate if e.isalnum())  # Keep only alphanumeric characters
    return plate.upper()

def is_similar_plate(plate1, plate2, threshold=0.85):
    """Check if two plates are similar using a higher similarity threshold."""
    return SequenceMatcher(None, plate1, plate2).ratio() > threshold

class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def log_entry(self, plate):
        global vehicle_data, recorded_plates

        # Current time in milliseconds and day-month-year
        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")[:-3]
        normalized_plate = normalize_plate(plate)

        # Check cooldown for re-logging similar plates
        last_recorded_time = vehicle_data.loc[vehicle_data['License Plate'] == normalized_plate, 'Entry Time']
        if not last_recorded_time.empty:
            last_entry_time = datetime.strptime(last_recorded_time.values[0], "%d-%m-%Y %H:%M:%S.%f")
            if datetime.now() - last_entry_time < cooldown_time:
                return  # Skip logging if it's within the cooldown period

        # Ensure plate format is valid before logging
        if not validate_plate_format(normalized_plate):
            return

        # Add new vehicle entry
        new_entry = pd.DataFrame({'License Plate': [normalized_plate], 'Entry Time': [current_time], 'Exit Time': [None]})
        vehicle_data = pd.concat([vehicle_data, new_entry], ignore_index=True)
        recorded_plates.add(normalized_plate)

    def log_exit(self, plate):
        global vehicle_data

        # Current time for exit
        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")[:-3]
        normalized_plate = normalize_plate(plate)

        if normalized_plate in vehicle_data['License Plate'].values:
            vehicle_data.loc[vehicle_data['License Plate'] == normalized_plate, 'Exit Time'] = current_time

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()

        det = preds[idx]
        if len(det) == 0:
            return

        for *xyxy, conf, cls in reversed(det):
            ocr = getOCR(im0, xyxy)
            if ocr != "":
                if ocr in vehicle_data['License Plate'].values:
                    self.log_exit(ocr)
                else:
                    self.log_entry(ocr)
            self.annotator.box_label(xyxy, ocr, color=colors(cls, True))

        # Save vehicle log every 100 frames to reduce file I/O overhead
        if self.seen % 100 == 0:
            vehicle_data.to_csv('vehicle_entry_exit_log.csv', index=False)
            print("Vehicle data saved to 'vehicle_entry_exit_log.csv'")

        return

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()

if __name__ == "__main__":
    reader = easyocr.Reader(['en'])
    predict()
