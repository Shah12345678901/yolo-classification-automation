import argparse
import os
from detection_to_crop import YOLOObjectCropper
from train_test_split import DataSplitter
from labele_to_crop import YOLOv8DataProcessor
from inference import GenderDetectionVideoProcessor
from latest_best_weight import find_latest_weights_folder

def main(args):
    if args.start_data_preparation:
        if args.crop_by_detector:
            det_model_path = args.det_model_path
            image_folder = args.image_folder
            destination_folder = args.destination_folder
            classes = args.classes
            cropper = YOLOObjectCropper(det_model_path, image_folder, destination_folder, classes)
            cropper.crop_objects()

        if args.crop_by_labels:
            image_folder = args.image_folder
            label_folder = args.label_folder
            output_folder = args.output_folder
            data_processor = YOLOv8DataProcessor(image_folder, label_folder, output_folder)
            data_processor.process_images()

        if args.split_data:
            data_folder = args.data_folder
            train_folder = args.train_folder
            val_folder = args.val_folder
            split_ratio = args.split_ratio
            splitter = DataSplitter(data_folder, train_folder, val_folder, split_ratio)
            splitter.split_data()

    if args.train_yolo:
        data_dir = args.data_dir
        epochs = args.epochs
        model = args.model
        yolo_task_cmd = f'yolo task=classify mode=train model={model} data="{data_dir}" epochs={epochs}'
        print(f"Running YOLO task command: {os.system(yolo_task_cmd)}")

    if args.inference:
        det_model_path = args.det_model_path
        cls_model_folder = find_latest_weights_folder()
        cls_model_path = cls_model_folder + "/best.pt"
        video_path = args.video_path
        output_video_path = args.output_video_path
        video_processor = GenderDetectionVideoProcessor(det_model_path, cls_model_path, video_path, output_video_path)
        video_processor.process_video()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data and perform YOLO training and inference.")
    
    parser.add_argument('--start_data_preparation', action='store_true', help="Start data preparation")
    parser.add_argument('--crop_by_detector', action='store_true', help="Crop by detector model")
    parser.add_argument('--det_model_path', type=str, help="det model path")
    parser.add_argument('--image_folder', type=str, help="Image folder path")
    parser.add_argument('--destination_folder', type=str, help="Destination folder path")
    parser.add_argument('--classes', type=int, nargs='+', help="List of classes")




    parser.add_argument('--crop_by_labels', action='store_true', help="Crop by using labels file")
    parser.add_argument('--label_folder', type=str, help="Label folder path")
    parser.add_argument('--output_folder', type=str, help="Output folder path")


    parser.add_argument('--split_data', action='store_true', help="Perform data splitting")
    parser.add_argument('--data_folder', type=str, help="Data folder path")
    parser.add_argument('--train_folder', type=str, help="Train folder path")
    parser.add_argument('--val_folder', type=str, help="Validation folder path")
    parser.add_argument('--split_ratio', type=float, help="Train:val ratio")


    parser.add_argument('--train_yolo', action='store_true', help="Start YOLO training")
    parser.add_argument('--data_dir', type=str, help="Data directory for training")
    parser.add_argument('--epochs', type=int, help="Number of epochs")
    parser.add_argument('--model', type=str, help="YOLO model config")


    parser.add_argument('--inference', action='store_true', help="Start the Inference part")
    parser.add_argument('--video_path', type=str, help="Video file path")
    parser.add_argument('--output_video_path', type=str, help="Output video file path")






    # parser.add_argument('--crop_by_detector', action='store_true', help="Crop by detector model")
    # parser.add_argument('--det_model_path', type=str, help="det model path")
    # parser.add_argument('--image_folder', type=str, help="Image folder path")
    # parser.add_argument('--destination_folder', type=str, help="Destination folder path")
    # parser.add_argument('--classes', type=int, nargs='+', help="List of classes")
    # parser.add_argument('--crop_by_labels', action='store_true', help="Crop by using labels file")
    # parser.add_argument('--label_folder', type=str, help="Label folder path")
    # parser.add_argument('--output_folder', type=str, help="Output folder path")
    # parser.add_argument('--split_data', action='store_true', help="Perform data splitting")
    # parser.add_argument('--data_folder', type=str, help="Data folder path")
    # parser.add_argument('--train_folder', type=str, help="Train folder path")
    # parser.add_argument('--val_folder', type=str, help="Validation folder path")
    # parser.add_argument('--split_ratio', type=float, help="Train:val ratio")
    # parser.add_argument('--train_yolo', action='store_true', help="Start YOLO training")
    # parser.add_argument('--data_dir', type=str, help="Data directory for training")
    # parser.add_argument('--epochs', type=int, help="Number of epochs")
    # parser.add_argument('--model', type=str, help="YOLO model config")
    # parser.add_argument('--inference', action='store_true', help="Start the Inference part")
    # #parser.add_argument('--gender_model_path', type=str, help="Gender classification model file path")
    # parser.add_argument('--video_path', type=str, help="Video file path")
    # parser.add_argument('--output_video_path', type=str, help="Output video file path")

    args = parser.parse_args()
    main(args)
