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
from difflib import SequenceMatcher
import re

# Initialize DataFrame
vehicle_data = pd.DataFrame(columns=['License Plate', 'Entry Time', 'Exit Time'])

# Global set to track recorded plates (to avoid duplicates)
recorded_plates = set()

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def preprocess_image_for_ocr(im):
    """Preprocess image for better OCR results."""
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.GaussianBlur(im, (5, 5), 0)  # Reduce noise
    _, im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Adaptive thresholding
    return im

def getOCR(im, coors):
    x, y, w, h = int(coors[0]), int(coors[1]), int(coors[2]), int(coors[3])
    im = im[y:h, x:w]
    im = preprocess_image_for_ocr(im)
    conf = 0.3  # Confidence threshold

    try:
        results = reader.readtext(im)
        ocr_results = [result[1] for result in results if result[2] > conf]
        ocr = " ".join(ocr_results)
        return str(ocr)
    except Exception as e:
        print(f"Error applying OCR: {e}")
        return ""

def normalize_plate(plate):
    """Normalize the license plate by removing spaces, special characters, and converting to uppercase."""
    plate = re.sub(r'\s+', '', plate)  # Remove all whitespaces
    plate = re.sub(r'[^\w]', '', plate)  # Remove non-alphanumeric characters
    return plate.upper()

def correct_plate(plate):
    """Correct common OCR errors."""
    corrections = {
        'O': '0',
        'I': '1',
        'L': '1',
        'Z': '2',
        'S': '5',
        'Q': '0'
    }
    plate = normalize_plate(plate)
    for incorrect, correct in corrections.items():
        plate = plate.replace(incorrect, correct)
    return plate

def is_valid_plate(plate):
    """Check if the plate is valid (length and format)."""
    # Define the expected format (e.g., 7 alphanumeric characters)
    return len(plate) >= 6 and len(plate) <= 10

def is_similar_plate(plate1, plate2, threshold=0.8):
    """Check if two plates are similar using a similarity threshold."""
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
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        normalized_plate = correct_plate(plate)
        if not is_valid_plate(normalized_plate):
            return  # Skip invalid plates

        # Check if any recorded plate is similar to the new plate
        for rec_plate in recorded_plates:
            if is_similar_plate(rec_plate, normalized_plate):
                return  # If similar plate exists, do not log the new one

        # Log the plate if it's unique
        new_entry = pd.DataFrame({'License Plate': [normalized_plate], 'Entry Time': [current_time], 'Exit Time': [None]})
        vehicle_data = pd.concat([vehicle_data, new_entry], ignore_index=True)
        recorded_plates.add(normalized_plate)

    def log_exit(self, plate):
        global vehicle_data
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        normalized_plate = correct_plate(plate)
        if normalized_plate in vehicle_data['License Plate'].values:
            vehicle_data.loc[vehicle_data['License Plate'] == normalized_plate, 'Exit Time'] = current_time

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
                    label = ocr
                    # Log the entry only if it's a new plate
                    if ocr in vehicle_data['License Plate'].values:
                        self.log_exit(ocr)
                    else:
                        self.log_entry(ocr)
                self.annotator.box_label(xyxy, label, color=colors(c, True))
            if self.args.save_crop:
                imc = im0.copy()
                save_one_box(xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        # Save the DataFrame to a CSV file after processing all frames
        vehicle_data.to_csv('vehicle_entry_exit_log.csv', index=False)
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
