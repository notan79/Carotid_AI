import torch
import torchvision
from torchvision import datasets
import numpy as np
class ImageFolderWithPathsEncoder(datasets.ImageFolder):
    
    def __init__(self, path: str, transform: torchvision.transforms, encoder: torch.nn.Module, device: torch.device):
        super().__init__(path, transform=transform)
        self.encoder = encoder.eval()
        self.device = device
    
    def get_main_img_id(self, index: int) -> str:
        return self.imgs[index][0].split('/')[8]
    
    def get_encoded_img(self, index: int) -> torch.tensor:
        return self.encoder(super(ImageFolderWithPathsEncoder, self).__getitem__(index)[0].to(self.device).unsqueeze(0))
    
    def __getitem__(self, index: int) -> tuple:
        original_tuple = super(ImageFolderWithPathsEncoder, self).__getitem__(index)
        path, label = self.imgs[index]
        
        encoded = None
        with torch.no_grad():
            encoded = self.encoder(original_tuple[0].unsqueeze(0).to(self.device))
        
        tuple_with_path = (encoded, label, path)
        return tuple_with_path
    
class ImageFolderWithPaths(datasets.ImageFolder):
    
    def __init__(self, path: str, device: torch.device):
        super().__init__(path)
    
    def get_main_img_id(self, index: int) -> str:
        return self.imgs[index][0].split('/')[8]
    
    def __getitem__(self, index: int) -> tuple:
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path, label = self.imgs[index]
        
        tuple_with_path = (np.asarray(original_tuple[0]), label, path)
        return tuple_with_path
    
    
class PatientDatasetK(torch.utils.data.Dataset):
    def __init__(self, all_patients_dict, all_edges):
        self.patient_ids = list(all_patients_dict.keys())
        self.all_patients_dict = all_patients_dict
        self.all_edges = all_edges
        
    def __len__(self):
        return len(self.patient_ids)
    
    # Returns: (str, dict = {'enc': ..., 'label': ...}, list of edges)
    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        data = self.all_patients_dict[pid]
        edges = self.all_edges[pid]
        return pid, data, edges

class PatientDataset(torch.utils.data.Dataset):
    def __init__(self, all_patients_dict):
        self.patient_ids = list(all_patients_dict.keys())
        self.all_patients_dict = all_patients_dict
        
    def __len__(self):
        return len(self.patient_ids)
    
    # Returns: (str, dict = {'enc': ..., 'label': ..., 'edge_list': ...})
    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        data = self.all_patients_dict[pid]
        return pid, data