import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from AutoEncoderCNN import AE_CNN
from GridSearch import GridSearch
import socket

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

class Trainer:
    def __init__(self, model, train_data, optimizer, loss_function, gpu_id, verbose = 2, patience = 1000):
            
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.loss_function = loss_function
        print('here1')
        self.model = DDP(model, device_ids=[gpu_id], output_device = gpu_id)
        print('here2')

        self._verbose = verbose
        self._early_stop_depth = patience
        
        self.model.share_memory()
        
        
        
    def trainAE(self, EPOCHS):
            
        loss_arr = []
        min_loss = None
        outputs = []
        early_stop = False
        early_stop_depth = self._early_stop_depth
        
        loader = self.train_data
        
        for epoch in range(EPOCHS):
            
            self.train_data.sampler.set_epoch(epoch) # allows shuffling to work properly
            
            if early_stop:
                if self._verbose != 0:
                    print(f'\n\n------EARLY STOP {min_loss}------\n\n')
                break

            count = 0

            self.model.module.train()
            for (image, _) in loader:
                image = image.to(self.gpu_id)
                #image = image.flatten(start_dim=1) # ignore the batch_size

                recon = self.model(image)
                loss = self.loss_function(recon, image)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                
                # UI
                if self._verbose == 2:
                    sys.stdout.write('\r')
                    sys.stdout.write("Epoch: {} [{:{}}] {:.1f}% | Loss: {}\r".format(epoch+1, "="*count, 
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
                print(f'\nEpoch: {epoch + 1} | Loss: {loss.item():.4f} | GPU: {self.gpu_id}', end='\n'*2)
                
        torch.save(self.model.module.state_dict(), f'./models/Parallel')
        
        
        
def get_loaders(BATCH_SIZE):
    PATH = '/groups/francescavitali/eb2/subImages_slide299/H&E' # has 506 images

    tensor_transform = transforms.ToTensor()

    dataset = datasets.ImageFolder(PATH, 
                                  transform = tensor_transform) #loads the images

    train_set, val_set, test_set = torch.utils.data.random_split(dataset,
                                                       [404,51,51],# 70%, 30%
                                                       generator=torch.Generator(device='cpu'))

    train = torch.utils.data.DataLoader(dataset = train_set,
                                        batch_size = BATCH_SIZE,
                                        shuffle = False,
                                        sampler= DistributedSampler(train_set)) # copy paste for val and test
    
    
    return (train)
    
    
def ddp_setup(rank, world_size):
#     os.environ["MASTER_ADDR"] = str(socket.gethostbyname(socket.gethostname()))
#     os.environ["MASTER_PORT"] = os.environ["MASTER_PORT"] # some random port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
def main(rank, world_size, EPOCHS=10, BATCH_SIZE=4, LR = 0.0001, WD = 1e-5):
    torch.cuda.set_device(rank)
    print(1)
    ddp_setup(rank, world_size)
    loader = get_loaders(BATCH_SIZE)
    model = AE_CNN()
    
    optimizer = torch.optim.Adam(model.parameters(), lr = LR, weight_decay = WD)
    loss_function = torch.nn.MSELoss()

    print(2)

    trainer = Trainer(model, loader, optimizer, loss_function, rank, verbose=1)
    print(3)
    trainer.trainAE(EPOCHS)
    print(4)
    destroy_process_group()

if __name__ == '__main__':
    
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    
    print(f'Devices: {torch.cuda.device_count()}')
    
    
    EPOCHS = 5
    BATCH_SIZE = 2
    LR = 0.0001
    WD = 1e-5
    
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    
    main(local_rank,world_size, EPOCHS, BATCH_SIZE,LR,WD)
    
#     for i in range(world_size):
#         torch.cuda.set_device(i)
# #     main(0,1, EPOCHS, BATCH_SIZE,LR,WD)

#     processes = []
#     mp.set_start_method("spawn")
#     for rank in range(world_size):
#         p = mp.Process(target=main, args=(rank, world_size, EPOCHS, BATCH_SIZE,LR,WD))
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()
        
    #mp.spawn(main, args=(world_size, EPOCHS, BATCH_SIZE,LR,WD), nprocs=world_size)
