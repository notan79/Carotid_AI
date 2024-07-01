import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from AutoEncoderCNN import AE_CNN
from GridSearch import GridSearch

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


class Trainer:
    def __init__(self, model, train_data, optimizer, loss_function, gpu_id, verbose = 2):
            
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.loss_function = loss_function
        print('here1')
        self.model = DDP(model, device_ids=[gpu_id])
        print('here2')

        self._verbose = verbose
        
        self.model.share_memory()
        
        
        
    def trainAE(EPOCHS):
            
        loss_arr = []
        min_loss = None
        outputs = []
        early_stop = False
        early_stop_depth = self._early_stop_depth
        
        loader = self.train_data.to(self.gpu_id)
        
        for epoch in range(EPOCHS):
            
            self.train_data.sampler.set_epoch(epoch) # allows shuffling to work properly
            
            if early_stop:
                if self._verbose != 0:
                    print(f'\n\n------EARLY STOP {min_loss}------\n\n')
                break

            count = 0

            model.train()
            for (image, _) in loader:
                image = image.to(self.gpu_id)
                #image = image.flatten(start_dim=1) # ignore the batch_size

                recon = model(image)
                loss = loss_function(recon, image)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                
                # UI
                if self._verbose == 2:
                    sys.stdout.write('\r')
                    sys.stdout.write("Epoch: {} [{:{}}] {:.1f}% | Loss: {}".format(epoch+1, "="*count, 
                                                                               len(loader)-1, 
                                                                               (100/(len(loader)-1)*count), 
                                                                               loss.item()))
                    sys.stdout.flush()

                count += 1
                
            loss_arr.append(loss.item())
            if not min_loss:
                min_loss = loss_arr[0]
            if early_stop_depth >= 1 and early_stop_depth < len(loss_arr[loss_arr.index(min_loss):]):
                early_stop = True
                for loss_item in loss_arr[loss_arr.index(min_loss):]:
                    if loss_item < min_loss:
                        min_loss = loss_item
                        early_stop = False
               
                    
            outputs.append((epoch, image[1], recon[1]))
            
            if self._verbose != 0:
                print(f'\nEpoch: {epoch + 1} | Loss: {loss.item():.4f}', end='\n'*2)
                
        torch.save(self.model.module.state_dict(), f'./models/Parallel')
    
    
    
    
