import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data.dataloader import default_collate

from utils.dataloaders import TrainingDirectoryDataset
from training.standard_training import standard_training, SIUPPA_training
from training.utils.arguments import read_arguments
from training.utils.loaders import load_operator, load_solver

from random import randrange

def compute_patch(input_img, groundtruth_img, patch_size, scale):
    patch_low = input_img
    patch_gt = groundtruth_img
    if patch_size != 0:
        coordx = randrange(int(patch_size/2), input_img.size()[1]-int(patch_size/2)-1)
        coordy = randrange(int(patch_size/2), input_img.size()[2]-int(patch_size/2)-1)
        
        patch_low = input_img[:,int(coordx-patch_size/2):int(coordx+patch_size/2),int(coordy-patch_size/2):int(coordy+patch_size/2)]
        patch_gt = groundtruth_img[:,scale*int(coordx-patch_size/2):scale*int(coordx+patch_size/2),scale*int(coordy-patch_size/2):scale*int(coordy+patch_size/2)]
    return [patch_low, patch_gt]

def patch_collate(batch, patch_size, scale):
    patchs = [compute_patch(item[0], item[1], patch_size, scale) for item in batch]
    return default_collate(patchs)

#####################LOADING VARIABLES##########################
args = read_arguments()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_location = os.path.dirname(os.path.realpath(__file__)) + '/../' + "ckpts/" + args.model_name + ".ckpt"

#####################LOADING SOLVER##########################
forward_operator = load_operator(args, device)
solver = load_solver(args, forward_operator, device)
if args.solver == "SIUPPA":
    lambda_prox = torch.tensor(1.0, requires_grad=True, device=device)

#####################LOADING DATASET##########################
transform = transforms.Compose([transforms.ToTensor()])

training_dataset = TrainingDirectoryDataset(input_directory=args.input_training_dataset,groundtruth_directory=args.groundtruth_training_dataset,transform=transform)
training_dataloader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=int(args.batch_size), collate_fn=lambda b: patch_collate(b, int(args.patch_size), int(args.scale)), shuffle=True, drop_last=True, num_workers=8)

validation_dataset = TrainingDirectoryDataset(input_directory=args.input_testing_dataset,groundtruth_directory=args.groundtruth_testing_dataset,transform=transform)
validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=8)

#####################LOADING OPTIMIZER##########################
start_epoch = 0
if args.solver == "SIUPPA":
    optimizer = optim.Adam([{'params':solver.parameters(), 'lr':float(args.learning_rate)}, {'params':lambda_prox, 'lr':float(args.lambda_lr)}])
else:
    optimizer = optim.Adam(params=solver.parameters(), lr=float(args.learning_rate))
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=40, gamma=0.1)
cpu_only = not torch.cuda.is_available()
if os.path.exists(save_location):
    if not cpu_only:
        saved_dict = torch.load(save_location)
    else:
        saved_dict = torch.load(save_location, map_location='cpu')
    start_epoch = saved_dict['epoch']
    solver.load_state_dict(saved_dict['solver_state_dict'])
    optimizer.load_state_dict(saved_dict['optimizer_state_dict'])
    scheduler.load_state_dict(saved_dict['scheduler_state_dict'])

loss_function = torch.nn.MSELoss()

#####################LAUNCHING TRAINING##########################
if args.solver == "SIUPPA":
    SIUPPA_training(solver, lambda_prox, optimizer, scheduler, loss_function, training_dataloader, validation_dataloader, save_location, int(args.batch_size), int(args.epochs), start_epoch, int(args.nb_samples), training_dataset.nb_images, validation_dataset.nb_images, device)
else:
    standard_training(solver, optimizer, scheduler, loss_function, training_dataloader, validation_dataloader, save_location, int(args.batch_size), int(args.epochs), start_epoch, int(args.nb_samples), training_dataset.nb_images, validation_dataset.nb_images, device)
