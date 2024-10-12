import torch
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.general import cv2, check_img_size
from ultralytics.yolo.utils.torch_utils import select_device
from ultralytics.yolo.utils.datasets import LoadStreams, LoadImages
from ultralytics.yolo.utils.plots import Annotator

class YOLOv8_OCRPredictor(BasePredictor):

    def __init__(self, opt):
        super().__init__()
        self.model = torch.load(opt.model, map_location=self.device)['model'].float()
        self.device = select_device(opt.device)
        self.imgsz = check_img_size(opt.imgsz, s=self.model.stride.max())  # check image size
        self.vid_path, self.vid_writer = None, None
        self.opt = opt

    def get_annotator(self, img):
        """Initialize annotator for each frame."""
        if img is not None:
            return Annotator(img, line_width=self.opt.line_thickness, example=str(self.model.names))
        else:
            print("Error: Input image is None.")
            return None

    def save_preds(self, vid_cap, i, path):
        """Saves predictions with annotations for each frame."""
        # Ensure annotator is initialized
        if self.annotator is None:
            self.annotator = self.get_annotator(self.imgs[i])

        if self.annotator:
            im0 = self.annotator.result()  # Try to obtain the annotated result
            if im0 is not None:
                vid_cap.write(im0)  # Write the annotated frame to the video file
            else:
                print(f"Error: Annotator result is None for frame {i}.")
        else:
            print("Error: Annotator was not initialized for frame {i}.")

    def run(self, source):
        """Runs the prediction on the input source."""
        source = str(source)
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.model.stride.max(), auto=True)

        for path, img, im0s, vid_cap, s in dataset:
            self.annotator = self.get_annotator(im0s)  # Initialize the annotator for each frame

            # Inference
            img = torch.from_numpy(img).to(self.device)
            img = img.float()  # uint8 to fp32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim

            pred = self.model(img)[0]  # Get model predictions

            # Process predictions
            for i, det in enumerate(pred):  # detections per image
                if len(det):  # If detection is available
                    self.save_preds(vid_cap, i, path)

if __name__ == "__main__":
    # Example usage
    opt = {
        'model': '/content/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/best.pt',
        'source': '/content/drive/My Drive/HackTU/Dataset_TIET.mp4',
        'device': 'cuda',  # or 'cpu'
        'imgsz': 640,  # image size
        'line_thickness': 3  # line thickness for annotation
    }
    
    # Create an instance of the predictor
    predictor = YOLOv8_OCRPredictor(opt)

    # Run the predictor on the source
    predictor.run(opt['source'])
