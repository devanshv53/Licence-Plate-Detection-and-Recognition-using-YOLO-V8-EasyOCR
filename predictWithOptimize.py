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

# Initialize DataFrame
vehicle_data = pd.DataFrame(columns=['License Plate', 'Entry Time', 'Exit Time'])

# Global set to track recorded plates (to avoid duplicates)
recorded_plates = set()

# Character map for OCR correction
CHAR_MAP = {
    'S': '5', 'O': '0', 'I': '1', 'Z': '2', 'E': '3',
    'l': '1', 'Q': '0', 'B': '8', 'G': '6', 'T': '7'
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

def correct_plate(plate):
    """Apply character mapping and validate with regex."""
    corrected_plate = ''.join([CHAR_MAP.get(char, char) for char in plate])
    pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$'
    return corrected_plate if re.match(pattern, corrected_plate) else None

def normalize_plate(plate):
    plate = ''.join(e for e in plate if e.isalnum())  
    return plate.upper()

def is_similar_plate(plate1, plate2, threshold=0.8):
    return SequenceMatcher(None, plate1, plate2).ratio() > threshold

class DetectionPredictor(BasePredictor):

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

    def log_entry(self, plate):
        global vehicle_data, recorded_plates
        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")[:-3]

        normalized_plate = normalize_plate(plate)
        corrected_plate = correct_plate(normalized_plate)

        if corrected_plate and not any(is_similar_plate(rec_plate, corrected_plate) for rec_plate in recorded_plates):
            new_entry = pd.DataFrame({'License Plate': [corrected_plate], 'Entry Time': [current_time], 'Exit Time': [None]})
            vehicle_data = pd.concat([vehicle_data, new_entry], ignore_index=True)
            recorded_plates.add(corrected_plate)

    def log_exit(self, plate):
        global vehicle_data
        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")[:-3]
        
        normalized_plate = normalize_plate(plate)
        corrected_plate = correct_plate(normalized_plate)

        if corrected_plate and corrected_plate in vehicle_data['License Plate'].values:
            vehicle_data.loc[vehicle_data['License Plate'] == corrected_plate, 'Exit Time'] = current_time

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

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        for *xyxy, conf, cls in reversed(det):
            if self.args.save_txt:
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save or self.args.save_crop or self.args.show:
                c = int(cls)
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                ocr = getOCR(im0, xyxy)
                if ocr != "":
                    label = ocr
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

        vehicle_data.to_csv('vehicle_entry_exit_log.csv', index=False)
        print("Vehicle data saved to 'vehicle_entry_exit_log.csv'")

        return log_string

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()

if __name__ == "__main__":
    reader = easyocr.Reader(['en'])
    predict()
