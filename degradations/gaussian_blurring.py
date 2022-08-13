import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')
from operators.blurring import GaussianBlur
from operators.operator import OperatorPlusNoise
from degradations.utils.arguments import gaussian_blur_read_arguments
from utils.dataloaders import directory_filelist, load_img
import torch
from torchvision.utils import save_image
from torchvision import transforms
import progressbar

#####################LOADING VARIABLES##########################
input_dataset, output_dataset, nb_channels, kernel_sigma, noise_sigma = gaussian_blur_read_arguments()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#####################LOADING OPERATOR##########################
forward_operator = GaussianBlur(kernel_sigma=kernel_sigma, nb_channels=nb_channels, device=device).to(device=device)
forward_operator = OperatorPlusNoise(operator=forward_operator, noise_sigma=noise_sigma).to(device=device)

transform = transforms.Compose([transforms.ToTensor()])

#####################APPLYING OPERATOR##########################
file_list = directory_filelist(input_dataset)
nb_images_processed = 0
nb_images = len(file_list)
bar = progressbar.ProgressBar(maxval=nb_images, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
for file_name in file_list:
    img = load_img(os.path.join(input_dataset, file_name))
    img = transform(img).to(device=device)
    img = img.unsqueeze(0)
    
    degraded_img = forward_operator(img)
    
    save_image(degraded_img, os.path.join(output_dataset, file_name))
    
    nb_images_processed += 1
    bar.update(nb_images_processed)
bar.finish()