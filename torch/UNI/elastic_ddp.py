# Basic torch imports
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.nn.functional import binary_cross_entropy

# DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Utilities
import numpy as np
import sys
import os

# Model
from uni import get_encoder
from Classifier import Classifier

def prepare_dataloader(rank, world_size, transform, batch_size=16, pin_memory=False, num_workers=0):
    MAIN_PATH = '/groups/francescavitali/eb2/NewsubSubImages4/H&E'

    # Full dataset
    dataset = datasets.ImageFolder(MAIN_PATH, transform = transform)
    
    SPLIT = [55767, 6971, 6971] 
    
    train_set, _, _ = torch.utils.data.random_split(dataset,
                                                SPLIT)      # 80%, 10%, 10%
    
    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)

    train_loader = torch.utils.data.DataLoader(dataset = train_set,
                                            batch_size = batch_size,
                                            pin_memory=pin_memory,
                                            num_workers=num_workers,
                                            drop_last=False,
                                            shuffle = False,
                                            sampler=sampler)
    return train_loader


def main(LR=1e-4, GAMMA=0.02, EPOCHS=4, BATCH_SIZE=16):
    
    # initialize multiple processes
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    
    if rank == 0:
        # Empty the file
        with open('progress.txt', 'w') as file:
            file.write(f"DDP Training:\n\n")
            
        # Write to file
        write_out(f"Devices: {torch.cuda.device_count()}\n")
        
    # Get uni encoder and transform
    encoder = torch.load(f'models/encoder.pth').to(rank)
    transform = torch.load(f'models/transform.pth')
    
    # create model and move it to GPU with device id
    device_id = rank % torch.cuda.device_count()
    model = Classifier().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    
    # Get the loader
    train_loader = prepare_dataloader(rank, torch.cuda.device_count(), transform, batch_size=BATCH_SIZE)
    
    # Initialize the optimizer and scheduler
    optimizer = torch.optim.Adam([{'params': ddp_model.parameters(),'lr': LR}])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)
    
    # Set models to correct mode
    ddp_model.train()
    encoder.eval()
    
    if rank == 0:
        write_out(ddp_model.module)
    
    print(len(train_loader))
    # Train
    for epoch in range(EPOCHS):
        count = 0
        write_out(f"Epoch: {epoch+1}   |   Device: {device_id}\n")
        
        # Set the correct epoch
        train_loader.sampler.set_epoch(epoch)
        
        # Loop over all images 
        for img, label in train_loader: 
            optimizer.zero_grad()

            img = img.to(device_id)
            label = label.to(device_id)
            
            with torch.no_grad():
                features = encoder(img)

            pred = ddp_model(features)

            loss = binary_cross_entropy(pred, label.unsqueeze(dim=1).to(torch.float32))
            loss.backward()
            optimizer.step()
            
            if count % 100 == 0 or count == len(train_loader) - 1:
                print(f'Update: {rank}: {count}\n')
            count += 1
        
        # Update the scheduler
        scheduler.step()
        write_out(f'---Epoch: {epoch + 1} on device: {device_id} | Loss: {loss.item():.4f}---\n')
    
    write_out(f"Finished training on rank {rank}.")
    
    # Wait for all processes to complete
    dist.barrier()
    
    # Save on master
    if rank == 0:
        torch.save(ddp_model.module, f'./models/ddp.pth')
        write_out("Saved")
        
    # Cleanup
    dist.destroy_process_group()
    write_out(f"Destroyed process group: {rank}.")
    
    
def write_out(s):
    print(f'{s}\n')
    with open('progress.txt', 'a') as file:
        file.write(f"{s}\n")

if __name__ == "__main__":
    LR = 1e-10
    GAMMA = 0.1
    EPOCHS = 20
    BATCH_SIZE = 32
    main(LR, GAMMA, EPOCHS, BATCH_SIZE)