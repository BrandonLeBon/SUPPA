import os
from PIL import Image
from torch.utils.data import Dataset

def directory_filelist(target_directory):
    file_list = [f for f in os.listdir(target_directory) if os.path.isfile(os.path.join(target_directory, f)) and not f.startswith('.')]
    return file_list

def load_img(file_name):
    with open(file_name,'rb') as f:
        img = Image.open(f).convert("RGB")
    return img

class TrainingDirectoryDataset(Dataset):
    def __init__(self, input_directory, groundtruth_directory, transform):
        filelist = directory_filelist(input_directory)
        self.input_list = [input_directory + single_file for single_file in filelist]
        self.groundtruth_list = [groundtruth_directory + single_file for single_file in filelist]
        self.transform = transform
        self.nb_images = len(filelist)

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, item):
        input_image = load_img(self.input_list[item])
        groundtruth_image = load_img(self.groundtruth_list[item])
        input_image = self.transform(input_image)
        groundtruth_image = self.transform(groundtruth_image)
        return input_image, groundtruth_image
         
class InferenceDirectoryDataset(Dataset):
    def __init__(self, input_directory, transform):
        filelist = directory_filelist(input_directory)
        self.input_list = [input_directory + single_file for single_file in filelist]
        self.transform = transform
        self.nb_images = len(filelist)

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, item):
        input_image = load_img(self.input_list[item])
        input_image = self.transform(input_image)
        return input_image