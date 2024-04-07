import os

def find_latest_weights_folder():
    runsdetect_path = "/home/shahzeb/classifiaction-Automation/runs/classify/"

    def custom_sort_key(item):
        # Extract the numeric part attached to "train" and convert it to an integer
        try:
            return int(item[len('train'):])
        except ValueError:
            # If there's no numeric part, return a large negative value
            return -1

    train_folders_path = runsdetect_path
    train_folders = [folder for folder in os.listdir(train_folders_path) if folder.startswith('train')]

    train_folder = sorted(train_folders, key=custom_sort_key, reverse=True)
    print(train_folder)
    train_folder_curr = train_folder[0]
    weights_folder = os.path.join(runsdetect_path, train_folder_curr, 'weights')
    return weights_folder

