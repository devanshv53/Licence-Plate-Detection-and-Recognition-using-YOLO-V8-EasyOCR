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
from difflib import SequenceMatcher  # For fuzzy matching

# Initialize DataFrame
vehicle_data = pd.DataFrame(columns=['License Plate', 'Entry Time', 'Exit Time'])

# Global set to track recorded plates (to avoid duplicates)
recorded_plates = set()

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Common OCR misread character mapping
char_mapping = {
    '0': ['O', 'Q', 'D'],
    '1': ['I', 'L'],
    '2': ['Z'],
    '3': ['E'],
    '4': ['A'],
    '5': ['S'],
    '6': ['G'],
    '7': ['T'],
    '8': ['B'],
    '9': ['P'],
    'O': ['0', 'Q'],
    'Z': ['2'],
    'Q': ['O', '0'],
    'E': ['3'],
    'A': ['4'],
    'S': ['5'],
    'G': ['6'],
    'T': ['7'],
    'P': ['9']
}

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

def correct_plate_chars(plate):
    """Correct common OCR character misreads."""
    corrected_plate = ""
    for char in plate:
        found = False
        for key, values in char_mapping.items():
            if char in values:
                corrected_plate += key
                found = True
                break
        if not found:
            corrected_plate += char
    return corrected_plate.upper()

def normalize_plate(plate):
    """Normalize the license plate by removing spaces, special characters, and converting to uppercase."""
    plate = ''.join(e for e in plate if e.isalnum())  # Keep only alphanumeric characters
    return plate.upper()

def is_similar_plate(plate1, plate2, threshold=0.5):
    """Check if two plates are similar using a similarity threshold (50% similarity by default)."""
    return SequenceMatcher(None, plate1, plate2).ratio() > threshold

def validate_plate(plate):
    """Validate license plate using regex pattern for Indian formats."""
    pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$'
    return re.match(pattern, plate) is not None

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
        # Change time format to include milliseconds and day-month-year
        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")[:-3]

        normalized_plate = normalize_plate(plate)
        
        # Check if the new plate is similar to an already recorded one with more than 80% similarity
        for rec_plate in recorded_plates:
            if is_similar_plate(rec_plate, normalized_plate):
                return  # If a similar plate exists, do not log a new entry

        # If no similar plate is found, log it as a new entry
        new_entry = pd.DataFrame({'License Plate': [normalized_plate], 'Entry Time': [current_time], 'Exit Time': [None]})
        vehicle_data = pd.concat([vehicle_data, new_entry], ignore_index=True)
        recorded_plates.add(normalized_plate)

    def log_exit(self, plate):
        global vehicle_data
        # Change time format to include milliseconds and day-month-year
        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")[:-3]
        
        normalized_plate = normalize_plate(plate)
        
        # Check if the plate exists in the log, and mark its exit time
        for index, row in vehicle_data.iterrows():
            if is_similar_plate(row['License Plate'], normalized_plate):
                if pd.isna(vehicle_data.at[index, 'Exit Time']):  # Only log exit if not already recorded
                    vehicle_data.at[index, 'Exit Time'] = current_time

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in reversed(det):
            if self.args.save_txt:  # Write to file
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                ocr = getOCR(im0, xyxy)
                if ocr != "":
                    corrected_ocr = correct_plate_chars(ocr)
                    if validate_plate(corrected_ocr):  # Validate using regex
                        label = corrected_ocr
                        # Log the entry only if it's a new plate
                        if corrected_ocr in vehicle_data['License Plate'].values:
                            self.log_exit(corrected_ocr)
                        else:
                            self.log_entry(corrected_ocr)
                    else:
                        print(f"Invalid plate format detected: {corrected_ocr}")
                self.annotator.box_label(xyxy, label, color=colors(c, True))
            if self.args.save_crop:
                imc = im0.copy()
                save_one_box(xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        # Save the DataFrame to a CSV file after processing all frames
        vehicle_data.to_csv('/content/vehicle_entry_exit_log.csv', index=False)
        print("Vehicle data saved to 'vehicle_entry_exit_log.csv'")

        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    predict()
