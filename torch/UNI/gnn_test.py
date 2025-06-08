print('Starting script: gnn_test.py')

import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as geo_nn
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader

from torchvision import datasets, transforms
from torch.nn import BCEWithLogitsLoss as BCE
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import gc
import psutil

from util.GCN_GAT import GCN_GAT
from util.CustomDatasets import PatientDataset
from util.train_datapoint import train_graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

MAIN_PATH = '/groups/francescavitali/eb2/NewsubSubImages4/H&E'
all_patients_dict = torch.load('./data/img_dict.pth')
all_edges = torch.load('./data/edges.pth')
keys_temp = list(all_edges.keys())[:10]

patient_max = None
max_edges = 0
for k in keys_temp:
    print(f'{k}: {all_edges[k][:10]}... total edges: {len(all_edges[k])}')
    if(len(all_edges[k]) > max_edges):
        max_edges = len(all_edges[k])
        patient_max = k
print(f'Amt keys: {len(all_patients_dict.keys())}')  
    
dataset = PatientDataset(all_patients_dict, all_edges)

# batch_size MUST BE 1
train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, generator=torch.Generator(device))

model = GCN_GAT(heads=3).to(device)


all_labels = [] # list of all labels s.t. x_i in {0,1}
for k, v in all_patients_dict.items():
    all_labels.append(v['label'].item())
    
    
    
binary_cross_entropy = BCE()

# Get the class weighting
class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels)

optimizer = torch.optim.Adam([{'params': model.parameters(),'lr': 0.005}])

i = 0
for train_batch in train_loader:
    i += 1
    print(f"\n----------\nMemory at start of loop (RSS): {psutil.Process().memory_info().rss / 1e6:.2f} MB")
    
    k = train_batch[0][0] # patient id
    print(f'Patient: {k} ({i}/{len(train_loader)})')
    v = train_batch[1] # dict with enc and label
    edges = train_batch[2] # edges for the graph
    
    optimizer.zero_grad()
    loss = train_graph(k,v,edges, model)

    if loss == None:
        continue
    
    loss.backward()
    optimizer.step()
    
    del loss, train_batch
    gc.collect()
    print("----------")
    
    
    
print('Finished running script')
