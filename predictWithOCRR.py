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

# Initialize DataFrame
vehicle_data = pd.DataFrame(columns=['License Plate', 'Entry Time', 'Exit Time'])

# Global set to track recorded plates (to avoid duplicates)
recorded_plates = set()

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
        for rec_plate in recorded_plates:
            if is_similar_plate(rec_plate, normalized_plate):
                return  # If a similar plate exists, do not log the new one

        new_entry = pd.DataFrame({'License Plate': [normalized_plate], 'Entry Time': [current_time], 'Exit Time': [None]})
        vehicle_data = pd.concat([vehicle_data, new_entry], ignore_index=True)
        recorded_plates.add(normalized_plate)

    def log_exit(self, plate):
        global vehicle_data
        # Change time format to include milliseconds and day-month-year
        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")[:-3]
        
        normalized_plate = normalize_plate(plate)
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
import cv2
import torch
import re
import pandas as pd
from datetime import datetime

# Load models
detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 model for detection
recognition_model = ...  # Load your OCR model (EasyOCR, Tesseract, etc.)

# Define license plate regex pattern
plate_regex = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$')

# Initialize variables
vehicle_logs = []
entry_times = {}

# Process video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = detection_model(frame)  # Get detection results
        detections = results.pandas().xyxy[0]  # Convert to pandas dataframe

        for index, row in detections.iterrows():
            label = row['name']
            if label == 'license_plate':  # Filter for license plates
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                cropped_img = frame[y1:y2, x1:x2]

                # Run OCR on the cropped image
                plate_text = recognition_model(cropped_img)

                if plate_regex.match(plate_text):  # Validate the detected text
                    current_time = datetime.now()
                    if plate_text not in entry_times:  # First entry for this plate
                        entry_times[plate_text] = current_time
                        vehicle_logs.append([plate_text, current_time.strftime('%Y-%m-%d %H:%M:%S'), ''])  # Log entry time
                    else:  # Existing entry, log exit time
                        for log in vehicle_logs:
                            if log[0] == plate_text and log[2] == '':
                                log[2] = current_time.strftime('%Y-%m-%d %H:%M:%S')  # Update exit time
                                break

    cap.release()
    save_to_csv(vehicle_logs)

# Save logs to CSV
def save_to_csv(vehicle_logs):
    df = pd.DataFrame(vehicle_logs, columns=['License Plate', 'Entry Time', 'Exit Time'])
    df.to_csv('vehicle_logs.csv', index=False)
    print("Logs saved to vehicle_logs.csv")

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    # Set default values for model and source if not provided in the config
    cfg.model = cfg.model or "yolov8n.pt"  # Model name or path
    cfg.source = cfg.source or "/content/drive/My Drive/HackTU/Real_tiet_data.mp4"  # Default video file

    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    predictor = DetectionPredictor(cfg)
    predictor()

if __name__ == "__main__":
    reader = easyocr.Reader(['en'])
    predict()


