import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')
from operators.downsampling import Downsampling
from operators.operator import OperatorPlusNoise
from degradations.utils.arguments import downsampling_read_arguments
from utils.dataloaders import directory_filelist, load_img
import torch
from torchvision.utils import save_image
from torchvision import transforms
import progressbar

def crop_img(img_ori, scale):
    img_size = img_ori.size()
    mod_size_x = img_size[1] % scale
    mod_size_y = img_size[2] % scale
    resized_ori = img_ori[:,0:img_size[1] - mod_size_x,0:img_size[2] - mod_size_y]
    return resized_ori

#####################LOADING VARIABLES##########################
input_dataset, output_dataset, nb_channels, noise_sigma, scale, kernel_type, antialiasing = downsampling_read_arguments()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#####################LOADING OPERATOR##########################
forward_operator = Downsampling(scale=scale, kernel_type=kernel_type, antialiasing=antialiasing).to(device=device)
forward_operator = OperatorPlusNoise(operator=forward_operator, noise_sigma=noise_sigma).to(device=device)

transform = transforms.Compose([transforms.ToTensor()])

#####################APPLYING OPERATOR##########################
os.mkdir(os.path.join(output_dataset, "groundtruth"))
os.mkdir(os.path.join(output_dataset, "low_resolution"))

file_list = directory_filelist(input_dataset)
nb_images_processed = 0
nb_images = len(file_list)
bar = progressbar.ProgressBar(maxval=nb_images, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
for file_name in file_list:
    img = load_img(os.path.join(input_dataset, file_name))
    img = transform(img).to(device=device)
    img = crop_img(img, scale)
    img = img.unsqueeze(0)

    degraded_img = forward_operator(img)

    save_image(img, os.path.join(output_dataset, "groundtruth", file_name))
    save_image(degraded_img, os.path.join(output_dataset, "low_resolution", file_name))
    
    nb_images_processed += 1
    bar.update(nb_images_processed)
bar.finish()