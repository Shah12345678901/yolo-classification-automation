# yolo-classification-automation

## Description:
This project facilitates YOLOv8 classification on custom datasets, covering the entire workflow from data preparation to inference. It handles data preparation for the classification training by using the detector model and also if the data is prepared for the detection, then it will automatically prepare for the classification. The project streamlines the process, enabling seamless transition between data preparation for detection and classification tasks. After data preparation, the project proceeds to train the classification model and perform inference on test data.

## Features:
- Automated data preparation for classification from both object detection and classification data formats.
- Training YOLOv8 model for classification.
- Performing inference on test data.

## Folder Structure:
- `Classification-Automation`
  - `cls-data`
  - `Data`
  - `detection_data`
  - `model_folder`
  - `output_video`
  - `test_video` 

## Step 1: Logging into the Server
1. Open Visual Studio Code (VS Code).
2. Open the Command Palette with "Ctrl+Shift+P."
3. Search for "Remote-SSH: Connect to Host" and select it.
4. Add the SSH hostname provided.
5. Select the SSH configuration file and connect.
6. Enter your password when prompted.

## Step 2: Creating a Conda Environment
1. Open a terminal within VS Code.
2. Create a Conda environment with `conda create -n myenv`.
3. Activate the Conda environment with `conda activate myenv`.

## Step 3: Running YOLOv8 Classification
```bash
### If you have detection format data and detector model trained on it:
python3 main.py --start_data_preparation --crop_by_detector --det_model_path /home/shahzeb/classifiaction-Automation/model_folder/best.pt --image_folder /home/shahzeb/classifiaction-Automation/detection_data/images --destination_folder /home/shahzeb/classifiaction-Automation/Data --classes 0 1 --split_data --train_folder /home/shahzeb/classifiaction-Automation/cls-data/train --val_folder /home/shahzeb/classifiaction-Automation/cls-data/val --split_ratio 0.7 --train_yolo --data_dir /home/shahzeb/classifiaction-Automation/cls-data --epochs 1 --model yolov8n-cls.pt --inference --video_path /home/shahzeb/classifiaction-Automation/test_video/test.mp4 --output_video_path /home/shahzeb/classifiaction-Automation/output_video/1.mp4 --data_folder /home/shahzeb/classifiaction-Automation/Data


### If you have detection format data and want to crop by using the labels file:
```bash
python3 main.py --start_data_preparation --crop_by_labels --image_folder /home/shahzeb/classifiaction-Automation/detection_data/images --label_folder /home/shahzeb/classifiaction-Automation/detection_data/labels --output_folder /home/shahzeb/classifiaction-Automation/Data --split_data --data_folder /home/shahzeb/classifiaction-Automation/Data --train_folder /home/shahzeb/classifiaction-Automation/cls-data/train --val_folder /home/shahzeb/classifiaction-Automation/cls-data/val --split_ratio 0.7 --train_yolo --data_dir /home/shahzeb/classifiaction-Automation/cls-data --epochs 1 --model yolov8n-cls.pt --inference --video_path /home/shahzeb/classifiaction-Automation/test_video/test.mp4 --output_video_path /home/shahzeb/classifiaction-Automation/output_video --det_model_path /home/shahzeb/classifiaction-Automation/model_folder/best.pt


### If you already have prepared data for classification:
```bash
python3 main.py --train_yolo --data_dir /home/shahzeb/classifiaction-Automation/cls-data --epochs 10 --model yolov8n-cls.pt --inference --video_path /home/shahzeb/classifiaction-Automation/test_video --output_video_path /home/shahzeb/classifiaction-Automation/output_video --det_model_path /home/shahzeb/classifiaction-Automation/model_folder/best.pt




This way, all the commands for data preparation and inference are included under Step 3, making it more convenient for users to follow the instructions.

