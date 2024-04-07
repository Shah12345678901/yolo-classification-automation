import os
import shutil
import random

class DataSplitter:
    def __init__(self, data_folder, train_folder, val_folder, split_ratio):
        self.data_folder = data_folder
        self.train_folder = train_folder
        self.val_folder = val_folder
        self.split_ratio = split_ratio

    def split_data(self):
        os.makedirs(self.train_folder, exist_ok=True)
        os.makedirs(self.val_folder, exist_ok=True)

        subfolders = os.listdir(self.data_folder)

        for subfolder in subfolders:
            subfolder_path = os.path.join(self.data_folder, subfolder)
            train_subfolder_path = os.path.join(self.train_folder, subfolder)
            val_subfolder_path = os.path.join(self.val_folder, subfolder)

            os.makedirs(train_subfolder_path, exist_ok=True)
            os.makedirs(val_subfolder_path, exist_ok=True)

            image_files = [f for f in os.listdir(subfolder_path) if f.endswith(('.jpg', '.png', '.jpeg', '.gif'))]

            random.shuffle(image_files)
            split_index = int(len(image_files) * self.split_ratio)

            train_files = image_files[:split_index]
            val_files = image_files[split_index:]

            for file in train_files:
                src_file = os.path.join(subfolder_path, file)
                dest_file = os.path.join(train_subfolder_path, file)
                shutil.copy(src_file, dest_file)

            for file in val_files:
                src_file = os.path.join(subfolder_path, file)
                dest_file = os.path.join(val_subfolder_path, file)
                shutil.copy(src_file, dest_file)


