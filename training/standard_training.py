import os
import torch
import numpy as np
import progressbar
import datetime
import time
from torchvision.utils import save_image

def create_logs(save_location):
    day = str(datetime.date.today().year) + "_" + str(datetime.date.today().month) + "_" + str(datetime.date.today().day)
    time = str(datetime.datetime.now().hour) + "_" + str(datetime.datetime.now().minute) + "_" + str(datetime.datetime.now().second)
    
    log_day = os.path.dirname(os.path.realpath(__file__)) + '/../' + "logs/" + day
    if not os.path.isdir(log_day):
        os.mkdir(log_day)
    log_time = log_day + "/" + time
    if not os.path.isdir(log_time):
        os.mkdir(log_time)
    log_training = log_time + "/" + save_location.split('/')[-1][:-5]
    if not os.path.isdir(log_training):
        os.mkdir(log_training)
        
    return log_training

def save_logs(log_folder, training_losses, testing_losses, time_spent, memory_used):
    mean_training_loss = np.mean(training_losses)
    mean_testing_loss = np.mean(testing_losses)
    
    f = ""
    if not os.path.isfile(log_folder+"/log.txt"):
        f = open(log_folder+"/log.txt","w+")
    else:
        f = open(log_folder+"/log.txt","a+")
    f.write(str(mean_training_loss)+";"+str(mean_testing_loss)+";"+str(time_spent)+";"+str(memory_used)+"\n")
    f.close()
    
def save_model(solver, optimizer, scheduler, epoch, save_location, is_best):
    torch.save({'solver_state_dict': solver.state_dict(), 'epoch': epoch, 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}, save_location)
    if is_best:
        torch.save({'solver_state_dict': solver.state_dict()}, save_location[:-5]+"_best.ckpt")

def standard_training(solver, optimizer, scheduler, loss_function, train_dataloader, test_dataloader, save_location, batch_size, n_epochs, start_epoch, nb_samples_per_epoch, nb_training_img, nb_testing_img, device):    
    min_loss_validation = None
    log_folder = None
    for epoch in range(start_epoch, n_epochs):
        if device.type!='cpu':
            torch.cuda.reset_peak_memory_stats(device)
        time_start = time.time() 
        is_best = False
        
        #####################TRAIN##########################
        nb_batch_processed = 0
        nb_batchs = int(nb_training_img * nb_samples_per_epoch / float(batch_size))
        print("Training epoch " + str(epoch) + ":")
        bar = progressbar.ProgressBar(maxval=nb_batchs, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        
        training_losses = []               
        for i in range(nb_samples_per_epoch):
            for ii, sample_batch in enumerate(train_dataloader):
                input_batch = sample_batch[0].to(device=device)
                groundtruth_batch = sample_batch[1].to(device=device)
                optimizer.zero_grad()

                reconstruction = solver(input_batch)
    
                loss_value = loss_function(reconstruction, groundtruth_batch)
                loss_value.backward()
                optimizer.step()
                training_losses.append(loss_value.cpu().detach().numpy())
                
                nb_batch_processed += 1
                bar.update(nb_batch_processed)
        bar.finish()
        print("Mean training loss: " + str(np.mean(training_losses)))
        
        if scheduler is not None:
            scheduler.step(epoch)
        
        #####################TEST##########################
        nb_batch_processed = 0
        nb_batchs = int(nb_testing_img)
        print("Testing epoch:")
        bar = progressbar.ProgressBar(maxval=nb_batchs, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        
        testing_losses = []
        for ii, sample_batch in enumerate(test_dataloader):
            input_batch = sample_batch[0].to(device=device)
            groundtruth_batch = sample_batch[1].to(device=device)

            reconstruction = solver(input_batch).detach()

            loss_value = loss_function(reconstruction, groundtruth_batch)
            testing_losses.append(loss_value.cpu().detach().numpy())
            
            nb_batch_processed += 1
            bar.update(nb_batch_processed)
        bar.finish()
        print("Mean testing loss: " + str(np.mean(testing_losses)))

        validation_loss = np.mean(testing_losses)
        if min_loss_validation is None:
            min_loss_validation = validation_loss
            is_best = True
        else:
            if validation_loss < min_loss_validation:
                min_loss_validation = validation_loss
                is_best = True
                print("Best model!")
        
        if log_folder is None:        
            log_folder = create_logs(save_location)

        save_model(solver, optimizer, scheduler, epoch, save_location, is_best)
        training_computation_time = time.time()- time_start
        if device.type!='cpu':
            memory_used = torch.cuda.max_memory_allocated(device)
        else:
            memory_used = '-'
        save_logs(log_folder, training_losses, testing_losses, training_computation_time, memory_used)
        
def SIUPPA_training(solver, lambda_prox, optimizer, scheduler, loss_function, train_dataloader, test_dataloader, save_location, batch_size, n_epochs, start_epoch, nb_samples_per_epoch, nb_training_img, nb_testing_img, device):    
    min_loss_validation = None
    log_folder = None
    MSE = torch.nn.MSELoss()
    for epoch in range(start_epoch, n_epochs):
        if device.type!='cpu':
            torch.cuda.reset_peak_memory_stats(device)
        time_start = time.time() 
        is_best = False
        
        #####################TRAIN##########################
        nb_batch_processed = 0
        nb_batchs = int(nb_training_img * nb_samples_per_epoch / float(batch_size))
        print("Training epoch " + str(epoch) + ":")
        bar = progressbar.ProgressBar(maxval=nb_batchs, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        training_losses = []    
        inter_1 = []
        for i in range(nb_samples_per_epoch):
            for ii, sample_batch in enumerate(train_dataloader):
                input_batch = sample_batch[0].to(device=device)
                groundtruth_batch = sample_batch[1].to(device=device)

                optimizer.zero_grad()
                samples = solver(input_batch, True)

                total_loss = 0 
                for j in range(solver.nb_samples_per_inference):
                    intermediate_gt = (1+lambda_prox) * samples[j][1] - lambda_prox  * samples[j][0]
                    intermediate_loss = loss_function(intermediate_gt, groundtruth_batch)

                    total_loss = total_loss + intermediate_loss

                reconstruction_loss = loss_function(samples[-1], groundtruth_batch)
                total_loss = total_loss + reconstruction_loss
                
                total_loss.backward()
                optimizer.step()
                training_losses.append(reconstruction_loss.cpu().detach().numpy())
                
                nb_batch_processed += 1
                bar.update(nb_batch_processed)                    
        bar.finish()
        print("Mean training loss: " + str(np.mean(training_losses)), "Lambda: ", str(lambda_prox.item()))

        if scheduler is not None:
            scheduler.step(epoch)
        
        #####################TEST##########################
        nb_batch_processed = 0
        nb_batchs = int(nb_testing_img)
        print("Testing epoch:")
        bar = progressbar.ProgressBar(maxval=nb_batchs, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        
        testing_losses = []
        for ii, sample_batch in enumerate(test_dataloader):
            input_batch = sample_batch[0].to(device=device)
            groundtruth_batch = sample_batch[1].to(device=device)

            reconstruction = solver(input_batch, False).detach()

            loss_value = loss_function(reconstruction, groundtruth_batch)
            testing_losses.append(loss_value.cpu().detach().numpy())
            
            nb_batch_processed += 1
            bar.update(nb_batch_processed)
        bar.finish()
        print("Mean testing loss: " + str(np.mean(testing_losses)))
        
        validation_loss = np.mean(testing_losses)
        if min_loss_validation is None:
            min_loss_validation = validation_loss
            is_best = True
        else:
            if validation_loss < min_loss_validation:
                min_loss_validation = validation_loss
                is_best = True
                print("Best model!")
        
        if log_folder is None:        
            log_folder = create_logs(save_location)

        save_model(solver, optimizer, scheduler, epoch, save_location, is_best)
        training_computation_time = time.time()- time_start
        if device.type!='cpu':
            memory_used = torch.cuda.max_memory_allocated(device)
        else:
            memory_used = '-'
        save_logs(log_folder, training_losses, testing_losses, training_computation_time, memory_used)