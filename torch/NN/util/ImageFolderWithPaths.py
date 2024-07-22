import torch
from torchvision import datasets

class ImageFolderWithPaths(datasets.ImageFolder):
    
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path