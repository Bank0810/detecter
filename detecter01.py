import cv2
import cvzone
import math
from ultralytics import YOLO

class ObjectDetection:
    def __init__(self, model_path, class_names, video_source=0):
        self.model = YOLO(model_path)
        self.class_names = class_names
        self.cap = cv2.VideoCapture(video_source)
        self.cap.set(3, 1920)
        self.cap.set(4, 1080)

    def detect_objects(self):
        while True:
            success, img = self.cap.read()
            if not success:
                print("Failed to grab frame")
                break

            results = self.model(img, stream=True)
            self.draw_boxes(img, results)

            cv2.imshow('Image', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def draw_boxes(self, img, results):
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                # Class Name
                cvzone.putTextRect(img, f'{self.class_names[cls]} {conf}', (max(0, x1), max(35, y1)))

    def release_resources(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                   "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                   "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                   "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                   "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                   "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                   "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                   "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                   "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                   "teddy bear", "hair drier", "toothbrush"]

    detector = ObjectDetection('yolov8n.pt', class_names)
    try:
        detector.detect_objects()
    finally:
        detector.release_resources()

