from ultralytics import YOLO
import cv2
import os
import torch

class YOLOObjectCropper:
    def __init__(self, model_path, source_folder, destination_folders, classes):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.source_folder = source_folder
        self.destination_folders = destination_folders
        self.classes = classes

    def crop_objects(self):
        #os.makedirs(list(self.destination_folders.values()), exist_ok=True)

        count = 0

        for image_filename in os.listdir(self.source_folder):
            image_path = os.path.join(self.source_folder, image_filename)
            image = cv2.imread(image_path)

            results = self.model(image, conf=0.60, classes=self.classes)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0]
                    c = b.to(self.device)
                    c = c.cpu().detach().numpy()
                    x1, y1, x2, y2 = c
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)

                    class_label = int(box.cls.cpu().numpy())

                    if class_label in self.classes:
                        # Create a subfolder within the destination folder based on the class label
                        class_subfolder = os.path.join(self.destination_folders, str(class_label))
                        os.makedirs(class_subfolder, exist_ok=True)

                        # Save the cropped region within the class-specific subfolder
                        cropped_region = image[y1:y2, x1:x2]
                        output_path = os.path.join(class_subfolder, str(count) + image_filename)
                        cv2.imwrite(output_path, cropped_region)
                        count += 1



      

