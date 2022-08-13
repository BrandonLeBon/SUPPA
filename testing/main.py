import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from utils.dataloaders import TrainingDirectoryDataset
from testing.standard_testing import standard_testing
from testing.utils.arguments import read_arguments
from testing.utils.loaders import load_operator, load_solver

#####################LOADING VARIABLES##########################
args = read_arguments()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_location = os.path.dirname(os.path.realpath(__file__)) + '/../' + "ckpts/" + args.model_name + ".ckpt"

#####################LOADING DATASET##########################
transform = transforms.Compose([transforms.ToTensor()])

testing_dataset = TrainingDirectoryDataset(input_directory=args.input_testing_dataset,groundtruth_directory=args.groundtruth_testing_dataset,transform=transform)
testing_dataloader = torch.utils.data.DataLoader(dataset=testing_dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=8)

#####################LOADING SOLVER##########################
forward_operator = load_operator(args, device)
solver = load_solver(args, forward_operator, device)

#####################LOADING OPTIMIZER##########################
cpu_only = not torch.cuda.is_available()
if os.path.exists(save_location):
    if not cpu_only:
        saved_dict = torch.load(save_location)
    else:
        saved_dict = torch.load(save_location, map_location='cpu')
    solver.load_state_dict(saved_dict['solver_state_dict'])
else:
    print("JUST SO YOU KNOW, YOUR CHECKPOINT DOES NOT EXIST. CONTINUING WITH A RANDOMLY-INITIALIZED SOLVER.")

loss_function = torch.nn.MSELoss()

#####################LAUNCHING TRAINING##########################
standard_testing(solver, loss_function, testing_dataloader, testing_dataset.nb_images, args.ycbcr, args.output_folder, device)