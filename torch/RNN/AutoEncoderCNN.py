import torch
class AE_CNN(torch.nn.Module):
    def __init__(self, first_dim=64, encode_dim=128):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3,first_dim,8,stride=2,padding=1), # outputs first_dim, 147, 147
            torch.nn.ReLU(),
            torch.nn.Conv2d(first_dim,(first_dim+encode_dim)//2,8,stride=2,padding=1), # outputs _, 71, 71
            torch.nn.ReLU(),
            torch.nn.Conv2d((first_dim+encode_dim)//2,encode_dim,8,stride=2,padding=1), # outputs encode_dim, 33, 33
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(encode_dim,(first_dim+encode_dim)//2,8,stride=2,padding=1, output_padding = 1), # outputs _, 71, 71
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d((first_dim+encode_dim)//2,first_dim,8, stride=2, padding=1, output_padding = 1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(first_dim,3,8, stride=2, padding=1, output_padding = 1),    
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded