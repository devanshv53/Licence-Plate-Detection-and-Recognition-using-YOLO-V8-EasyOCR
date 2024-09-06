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
import re  # For pattern matching

# Initialize DataFrame
vehicle_data = pd.DataFrame(columns=['License Plate', 'Entry Time', 'Exit Time'])
unique_plates = set()  # Set to track unique license plates

def correct_ocr_mistakes(plate):
    """Fix common OCR misreadings in license plates."""
    corrections = {
        'O': '0',  # Replace 'O' with '0'
        'Q': '0',  # Replace 'Q' with '0'
        'I': '1',  # Replace 'I' with '1'
        'S': '5',  # Replace 'S' with '5'
        'Z': '2',  # Replace 'Z' with '2'
        'B': '8',  # Replace 'B' with '8'
        'l': '1',  # Replace lowercase 'l' with '1'
        '|': '1'   # Replace '|' with '1'
    }
    
    for wrong_char, correct_char in corrections.items():
        plate = plate.replace(wrong_char, correct_char)
    return plate

def validate_plate_format(plate):
    """Validate the format of the detected license plate based on common patterns."""
    # General format for Indian license plates: 2 letters, 1-2 digits, 1-2 letters, 1-4 digits
    pattern = r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{1,4}$"
    return bool(re.match(pattern, plate))

def preprocess_image_for_ocr(im, coors):
    """Preprocess the image to enhance OCR accuracy."""
    x, y, w, h = int(coors[0]), int(coors[1]), int(coors[2]), int(coors[3])
    cropped_im = im[y:h, x:w]

    # Convert to grayscale
    gray = cv2.cvtColor(cropped_im, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply thresholding for better contrast
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Resize to improve OCR readability
    resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    return resized

def getOCR(im, coors):
    """Extract text (license plate) from image using OCR."""
    try:
        processed_im = preprocess_image_for_ocr(im, coors)
        results = reader.readtext(processed_im)
        conf = 0.2
        ocr = ""

        for result in results:
            if len(results) == 1:
                ocr = result[1]
            elif len(results) > 1 and len(result[1]) > 6 and result[2] > conf:
                ocr = result[1]

        # Normalize, correct mistakes, and validate the detected plate
        normalized_plate = correct_ocr_mistakes(ocr)
        corrected_plate = correct_ocr_mistakes(normalized_plate)

        if validate_plate_format(corrected_plate):
            return corrected_plate
        else:
            return ""
    except Exception as e:
        print(f"Error applying OCR: {e}")
        return ""

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
        global vehicle_data, unique_plates
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if plate not in unique_plates:  # Avoid duplicates
            unique_plates.add(plate)  # Add to set of unique plates
            new_entry = pd.DataFrame({'License Plate': [plate], 'Entry Time': [current_time], 'Exit Time': [None]})
            vehicle_data = pd.concat([vehicle_data, new_entry], ignore_index=True)

    def log_exit(self, plate):
        global vehicle_data
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if plate in vehicle_data['License Plate'].values:
            vehicle_data.loc[vehicle_data['License Plate'] == plate, 'Exit Time'] = current_time

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
                    # Log the entry and exit times
                    if ocr in unique_plates:
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
    reader = easyocr.Reader(['en