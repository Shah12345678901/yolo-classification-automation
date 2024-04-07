from ultralytics import YOLO
import cv2
import torch

class GenderDetectionVideoProcessor:
    def __init__(self, yolo_model_path, gender_model_path, video_path, output_video_path):
        self.yolo_model_path = yolo_model_path
        self.gender_model_path = gender_model_path
        self.video_path = video_path
        self.output_video_path = output_video_path

        self.yolo_model = YOLO(self.yolo_model_path)
        self.gender_model = YOLO(self.gender_model_path)

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(5))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (frame_width, frame_height))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            crops, coordinates = self.gender_detection(frame)

            for crop, coordinate in zip(crops, coordinates):
                results = self.gender_model(crop)
                class_label = results[0].probs.top1
                #name = 'Female' if class_label == 0 else 'Male'
                x1, y1, x2, y2 = coordinate

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, str(class_label), (x1 + 2, y2 - 2), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)

            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def gender_detection(self, frame):
        results = self.yolo_model(frame, conf=0.30, cls=0)
        crops = []
        coordinates = []
        device = torch.device("cuda" if torch.cuda.is_available () else "cpu")

        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]
                c = b.to(device)
                c = c.cpu().detach().numpy()
                
                x1, y1, x2, y2 = c

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                crop = frame[y1:y2, x1:x2]
                crops.append(crop)
                coordinates.append(c)
                
        return crops, coordinates


