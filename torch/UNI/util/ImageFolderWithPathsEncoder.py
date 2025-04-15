import torch
import torchvision
from torchvision import datasets
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