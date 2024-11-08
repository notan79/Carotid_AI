import torch

def get_totals(loader: torch.utils.data.DataLoader) -> dict:
    temp = {0:0, 1:0}
    
    for _, goal, _ in loader:
        temp[goal.item()] += 1
    
    return temp
