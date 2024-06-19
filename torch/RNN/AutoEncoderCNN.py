import torch
class AE_CNN(torch.nn.Module):
    def __init__(self, first_dim=64, encode_dim=128):
        super().__init__()
        
        dim = 64
        half = 32
        fourth = 26
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3,dim,8,stride=2,padding=1), # outputs first_dim, 147, 147
            torch.nn.ReLU(),
            torch.nn.Conv2d(dim,half,8,stride=2,padding=1), # outputs _, 71, 71
            torch.nn.ReLU(),
            torch.nn.Conv2d(half,fourth,8,stride=2,padding=1), # outputs encode_dim, 33, 33
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(fourth*33*33, fourth*33*33)
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Unflatten(1, (fourth, 33, 33)),
            torch.nn.ConvTranspose2d(fourth,half,8,stride=2,padding=1, output_padding = 1), # outputs _, 71, 71
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(half,dim,8, stride=2, padding=1, output_padding = 1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(dim,3,8, stride=2, padding=1, output_padding = 1),    
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded