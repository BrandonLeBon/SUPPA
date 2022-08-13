import os
import torch
import numpy as np
import progressbar
from math import log10
from torchvision.utils import save_image
from torch.autograd import Variable
import time
    
def rgb_to_ycbcr_wrong_dynamic(input):
  output = Variable(input.data.new(*input.size()))*255
  output[:, 0, :, :] = input[:, 0, :, :] * 65.481 + input[:, 1, :, :] * 128.553 + input[:, 2, :, :] * 24.966 + 16
  output[:, 1, :, :] = input[:, 0, :, :] * (-37.797) + input[:, 1, :, :] * (-74.203) + input[:, 2, :, :] * 112 + 128
  output[:, 2, :, :] = input[:, 0, :, :] * 112 + input[:, 1, :, :] * (-93.786) + input[:, 2, :, :] * (-18.214) + 128
  return output

def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta = .5
    y: torch.Tensor = .299 * r + .587 * g + .114 * b
    cb: torch.Tensor = (b - y) * .564 + delta
    cr: torch.Tensor = (r - y) * .713 + delta
    return torch.stack((y, cb, cr), -3)
    
def standard_testing(solver, loss_function, test_dataloader, nb_testing_img, ycbcr, output_folder, device):
    #####################TEST##########################
    nb_images_processed = 0
    nb_images = int(nb_testing_img)
    print("Testing:")
    bar = progressbar.ProgressBar(maxval=nb_images, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    
    if output_folder is not None:
        os.mkdir(os.path.join(output_folder, "groundtruth"))
        os.mkdir(os.path.join(output_folder, "results"))
        os.mkdir(os.path.join(output_folder, "degraded"))
    
    PSNR_accumulator = []
    MSE_accumulator = []
    compt = 0
    time_start = time.time() 
    
    for ii, sample_batch in enumerate(test_dataloader):
        input_batch = sample_batch[0].to(device=device)
        groundtruth_batch = sample_batch[1].to(device=device)
        
        reconstruction = solver(input_batch, groundtruth_batch).detach()
        
        if output_folder is not None:
            save_image(reconstruction[0], os.path.join(output_folder, "results", str(compt).zfill(6) + ".png"))
            save_image(groundtruth_batch[0], os.path.join(output_folder, "groundtruth", str(compt).zfill(6) + ".png"))
            save_image(input_batch[0], os.path.join(output_folder, "degraded", str(compt).zfill(6) + ".png"))
            compt += 1

        if ycbcr:
            reconstruction = rgb_to_ycbcr_wrong_dynamic(reconstruction)[:,0,:,:]
            groundtruth_batch = rgb_to_ycbcr_wrong_dynamic(groundtruth_batch)[:,0,:,:]

        loss_value = loss_function(reconstruction, groundtruth_batch)
            
        if ycbcr:
            psnr = 10 * log10(255**2/loss_value.item())
        else:
            psnr = 10 * log10(1/loss_value.item())
            
        PSNR_accumulator.append(psnr)
        MSE_accumulator.append(loss_value.item())
        nb_images_processed += 1
        bar.update(nb_images_processed)
    bar.finish()
    computation_time = time.time()- time_start
    if ycbcr:
        average_mse_psnr = 10 * log10(255**2/np.mean(MSE_accumulator))
    else:
        average_mse_psnr = 10 * log10(1/np.mean(MSE_accumulator))
    print("Average PSNR: " + str(np.mean(PSNR_accumulator)))
    print("Average MSE as PSNR: " + str(average_mse_psnr))
    print("Computation time:", computation_time, "seconds.")
