import torch
import numpy as np

def mse_loss(y_pred:torch.Tensor, y_exp: torch.Tensor) -> torch.tensor:
    return torch.mean( (y_pred - y_exp)**2)

def sparsity_loss(p:torch.Tensor) -> torch.tensor:
    return -torch.sum(torch.multiply(p, torch.log(p))) 

def dirichlet_loss(p:torch.Tensor) -> torch.tensor:
    # Negative sum of the log of the average of the columns within the batch
    return -torch.sum(torch.log(torch.mean(p, dim=0)))
