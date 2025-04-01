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

def prepare_dataloader(rank, world_size, transform, batch_size=12, pin_memory=False, num_workers=0):
    MAIN_PATH = '/groups/francescavitali/eb2/NewsubSubImages4/H&E'

    # Full dataset
    dataset = datasets.ImageFolder(MAIN_PATH, transform = transform)
    
    SPLIT = [101, 34804, 34804] # for testing
    
    train_set, _, _ = torch.utils.data.random_split(dataset,
                                                SPLIT)      # 80%, 10%, 10%
    print(f'Dataset len: {len(train_set)}')
    
    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)

    train_loader = torch.utils.data.DataLoader(dataset = train_set,
                                            batch_size = batch_size,
                                            pin_memory=pin_memory,
                                            num_workers=num_workers,
                                            drop_last=False,
                                            shuffle = False,
                                            sampler=sampler)
    return train_loader


def main(LR, GAMMA, EPOCHS):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    
    print(f"Start running basic DDP example on rank {rank}.")
    
    encoder = torch.load(f'models/encoder.pth').to(rank)
    transform = torch.load(f'models/transform.pth')
    
    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    model = Classifier().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    print("created ddp")
    
    train_loader = prepare_dataloader(rank, torch.cuda.device_count(), transform)
    print(f"Got loader for {rank}.")
    
    optimizer = torch.optim.Adam([{'params': ddp_model.parameters(),'lr': LR}])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)
    
    ddp_model.train()
    encoder.eval()
    
    print("Starting training")
    
    print(f"Len: {len(train_loader)}")
    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch+1}   |   Device: {device_id}")
        train_loader.sampler.set_epoch(epoch)
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
            
        print(f'---Epoch: {epoch + 1} on device: {device_id} | Loss: {loss.item():.4f}---')
        scheduler.step()
    
    if rank == 0:
        torch.save(ddp_model, f'./models/ddp.pth')
    dist.destroy_process_group()
    print(f"Finished running basic DDP example on rank {rank}.")

if __name__ == "__main__":
    print(f"Devices: {torch.cuda.device_count()}\n")
    LR = 4.3e-8
    GAMMA = 0.02
    EPOCHS = 4
    main(LR, GAMMA, EPOCHS)
    print("Done")