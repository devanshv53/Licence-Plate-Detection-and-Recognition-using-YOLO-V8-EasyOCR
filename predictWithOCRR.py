import hydra
import torch
import easyocr
import cv2
import pandas as pd
from datetime import datetime
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from difflib import SequenceMatcher  # For fuzzy matching
import os  # To handle directories

# Initialize OCR Reader
reader = easyocr.Reader(['en'])

# Global set to track recorded plates (to avoid duplicates)
recorded_plates = set()

class DetectionPredictor(BasePredictor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.vehicle_data = pd.DataFrame(columns=['License Plate', 'Entry Time', 'Exit Time'])
        self.recorded_plates = set()

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
        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")[:-3]
        normalized_plate = normalize_plate(plate)
        
        for rec_plate in self.recorded_plates:
            if is_similar_plate(rec_plate, normalized_plate):
                return  # Skip if a similar plate exists

        new_entry = pd.DataFrame({'License Plate': [normalized_plate], 'Entry Time': [current_time], 'Exit Time': [None]})
        self.vehicle_data = pd.concat([self.vehicle_data, new_entry], ignore_index=True)
        self.recorded_plates.add(normalized_plate)

    def log_exit(self, plate):
        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")[:-3]
        normalized_plate = normalize_plate(plate)
        
        if normalized_plate in self.vehicle_data['License Plate'].values:
            self.vehicle_data.loc[self.vehicle_data['License Plate'] == normalized_plate, 'Exit Time'] = current_time

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        im0 = im0.copy()
        if len(im.shape) == 3:
            im = im[None]  # Expand for batch dim
        
        det = preds[idx]
        if len(det) == 0:
            return log_string
        
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)  # Integer class
            label = f'{self.model.names[c]} {conf:.2f}' if not self.args.hide_conf else self.model.names[c]
            ocr = getOCR(im0, xyxy)

            if ocr:
                label = ocr
                if ocr in self.vehicle_data['License Plate'].values:
                    self.log_exit(ocr)
                else:
                    self.log_entry(ocr)

            self.annotator.box_label(xyxy, label, color=colors(c, True))

        # Save CSV after processing
        csv_path = 'vehicle_entry_exit_log.csv'
        self.vehicle_data.to_csv(csv_path, index=False)
        print(f"Vehicle data saved to '{csv_path}'")

        return log_string


def getOCR(im, coors):
    x, y, w, h = int(coors[0]), int(coors[1]), int(coors[2]), int(coors[3])
    im = im[y:h, x:w]
    conf = 0.2

    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    try:
        results = reader.readtext(gray)
        ocr = ""

        for result in results:
            if len(results) == 1:
                ocr = result[1]
            elif len(results) > 1 and len(result[1]) > 6 and result[2] > conf:
                ocr = result[1]

        return str(ocr)
    except Exception as e:
        print(f"Error applying OCR: {e}")
        return ""


def normalize_plate(plate):
    """Normalize the license plate by removing spaces, special characters, and converting to uppercase."""
    plate = ''.join(e for e in plate if e.isalnum())  # Keep only alphanumeric characters
    return plate.upper()


def is_similar_plate(plate1, plate2, threshold=0.8):
    """Check if two plates are similar using a similarity threshold."""
    return SequenceMatcher(None, plate1, plate2).ratio() > threshold


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    predict()
