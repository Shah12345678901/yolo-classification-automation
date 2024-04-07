import cv2
import os
import uuid

class YOLOv8DataProcessor:
    def __init__(self, image_folder, label_folder, output_folder):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.output_folder = output_folder

    def yolov5_to_x1y1x2y2(self, class_id, x, y, w, h, img_width, img_height):
        x_c = x * img_width
        y_c = y * img_height
        x1 = int(x_c - w * img_width / 2)
        y1 = int(y_c - h * img_height / 2)
        x2 = int(x_c + w * img_width / 2)
        y2 = int(y_c + h * img_height / 2)
        return class_id, x1, y1, x2, y2

    def process_images(self):
        # List all image files in the image folder
        image_files = [f for f in os.listdir(self.image_folder) if f.endswith('.jpg')]

        for image_file in image_files:
            image_path = os.path.join(self.image_folder, image_file)
            label_file = os.path.join(self.label_folder, os.path.splitext(image_file)[0] + ".txt")

            if not os.path.exists(label_file):
                print(f"Label file does not exist for {image_file}")
                continue

            image = cv2.imread(image_path)
            height, width, _ = image.shape

            with open(label_file, 'r') as file:
                lines = file.readlines()

            for line in lines:
                data = line.strip().split()
                class_id, x, y, w, h = map(float, data[0:5])  # YOLO format

                class_id, x1, y1, x2, y2 = self.yolov5_to_x1y1x2y2(class_id, x, y, w, h, width, height)

                # Create a folder for the class if it doesn't exist
                class_folder = os.path.join(self.output_folder, f'class_{class_id}')
                os.makedirs(class_folder, exist_ok=True)  # Automatically creates the folder if it doesn't exist

                # Generate a unique filename for the cropped image within the class folder
                unique_filename = str(uuid.uuid4())
                save_image_path = os.path.join(class_folder, f"{unique_filename}.jpg")
                cv2.imwrite(save_image_path, image[y1:y2, x1:x2])

                print(f"Saved image for {image_file} in {class_folder}")


