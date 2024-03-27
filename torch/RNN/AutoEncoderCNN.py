import torch
class AE_CNN(torch.nn.Module):
    def __init__(self, first_dim=64, encode_dim=128):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3,first_dim,16,stride=3,padding=1), # outputs first_dim, 96, 96
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(16,32,3,stride=2,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(first_dim,encode_dim,16,stride=4,padding=0) # outputs encode_dim, 21, 21
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(encode_dim,first_dim,16,stride=4,padding=0, output_padding = 0), # outputs first_dim, 96, 96
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(first_dim,3,16, stride=3, padding=1, output_padding = 0),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded