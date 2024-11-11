import hydra
import torch
import easyocr
import cv2
import pandas as pd
import re
from datetime import datetime
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from difflib import SequenceMatcher

class VehicleTracker:
    def __init__(self):
        self.vehicle_data = pd.DataFrame(columns=['License Plate', 'Entry Time', 'Exit Time', 'Confidence'])
        self.recorded_plates = {}  # Dict to store plate info with timestamps
        self.plate_pattern = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$")  # Adjust pattern based on your license plate format
        self.min_detection_confidence = 0.5
        self.plate_similarity_threshold = 0.85
        self.time_threshold = 60  # Minimum seconds between same plate detections
        
    def validate_plate(self, plate):
        """Validate if the detected plate matches expected format."""
        normalized = self.normalize_plate(plate)
        if len(normalized) < 6 or len(normalized) > 12:
            return False
        return bool(self.plate_pattern.match(normalized))
    
    def normalize_plate(self, plate):
        """Normalize the license plate text."""
        # Remove all non-alphanumeric characters and convert to uppercase
        normalized = ''.join(char for char in plate if char.isalnum()).upper()
        return normalized
    
    def find_similar_plate(self, plate):
        """Find if a similar plate exists in recorded plates."""
        normalized = self.normalize_plate(plate)
        for recorded in self.recorded_plates.keys():
            if SequenceMatcher(None, normalized, recorded).ratio() > self.plate_similarity_threshold:
                return recorded
        return None
    
    def should_update_plate(self, plate, current_time):
        """Determine if enough time has passed to update plate record."""
        if plate in self.recorded_plates:
            last_time = datetime.strptime(self.recorded_plates[plate]['last_seen'], 
                                        "%d-%m-%Y %H:%M:%S.%f")
            time_diff = (current_time - last_time).total_seconds()
            return time_diff > self.time_threshold
        return True
    
    def update_vehicle_record(self, plate, confidence):
        """Update vehicle entry/exit record."""
        current_time = datetime.now()
        time_str = current_time.strftime("%d-%m-%Y %H:%M:%S.%f")[:-3]
        
        if not self.validate_plate(plate):
            return False
            
        normalized_plate = self.normalize_plate(plate)
        similar_plate = self.find_similar_plate(normalized_plate)
        
        if similar_plate:
            normalized_plate = similar_plate
            
        if not self.should_update_plate(normalized_plate, current_time):
            return False
            
        if normalized_plate not in self.recorded_plates:
            # New entry
            self.recorded_plates[normalized_plate] = {
                'last_seen': time_str,
                'confidence': confidence
            }
            new_entry = pd.DataFrame({
                'License Plate': [normalized_plate],
                'Entry Time': [time_str],
                'Exit Time': [None],
                'Confidence': [confidence]
            })
            self.vehicle_data = pd.concat([self.vehicle_data, new_entry], ignore_index=True)
        else:
            # Update exit time
            self.recorded_plates[normalized_plate]['last_seen'] = time_str
            self.recorded_plates[normalized_plate]['confidence'] = max(
                confidence, 
                self.recorded_plates[normalized_plate]['confidence']
            )
            self.vehicle_data.loc[
                self.vehicle_data['License Plate'] == normalized_plate, 
                ['Exit Time', 'Confidence']
            ] = [time_str, self.recorded_plates[normalized_plate]['confidence']]
            
        return True

class ImprovedOCR:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])
        self.min_confidence = 0.4
        self.preprocessing_methods = [
            self._basic_preprocessing,
            self._adaptive_threshold,
            self._otsu_threshold
        ]
        
    def _basic_preprocessing(self, image):
        """Basic image preprocessing."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.equalizeHist(gray)
    
    def _adaptive_threshold(self, image):
        """Adaptive thresholding preprocessing."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    def _otsu_threshold(self, image):
        """Otsu's thresholding preprocessing."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    def read_plate(self, image, coords):
        """Extract and read license plate text using multiple preprocessing methods."""
        x, y, w, h = map(int, coords)
        plate_img = image[y:h, x:w]
        
        best_result = ("", 0)  # (text, confidence)
        
        for preprocess in self.preprocessing_methods:
            try:
                processed_img = preprocess(plate_img)
                results = self.reader.readtext(processed_img)
                
                for (_, text, conf) in results:
                    if conf > self.min_confidence and conf > best_result[1]:
                        best_result = (text, conf)
                        
            except Exception as e:
                print(f"Error in OCR preprocessing: {e}")
                continue
                
        return best_result

class DetectionPredictor(BasePredictor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.tracker = VehicleTracker()
        self.ocr = ImprovedOCR()
        
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))
        
    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()
        img /= 255
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
        
    def write_results(self, idx, preds, batch):
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
            
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
            
        for *xyxy, conf, cls in reversed(det):
            if conf < self.tracker.min_detection_confidence:
                continue
                
            if self.args.save_txt:
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / 
                       torch.tensor(im0.shape)[[1, 0, 1, 0]]).view(-1).tolist()
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    
            if self.args.save or self.args.save_crop or self.args.show:
                plate_text, ocr_conf = self.ocr.read_plate(im0, xyxy)
                
                if plate_text and self.tracker.update_vehicle_record(plate_text, ocr_conf):
                    label = f"{plate_text} ({ocr_conf:.2f})"
                    self.annotator.box_label(xyxy, label, color=colors(int(cls), True))
                    
                if self.args.save_crop:
                    save_one_box(xyxy, im0.copy(), 
                               file=self.save_dir / 'crops' / self.model.names[int(cls)] / f'{p.stem}.jpg',
                               BGR=True)
                               
        # Save the updated vehicle data
        self.tracker.vehicle_data.to_csv('vehicle_entry_exit_log.csv', index=False)
        
        return log_string

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()

if __name__ == "__main__":
    predict()
