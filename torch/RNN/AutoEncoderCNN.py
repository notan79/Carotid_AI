import torch
class AE_CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3,64,16,stride=3,padding=1), # outputs 64, 96, 96
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(16,32,3,stride=2,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,128,16,stride=4,padding=0) # outputs 128, 21, 21
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128,64,16,stride=4,padding=0, output_padding = 0), # outputs 64, 95, 95
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64,3,16, stride=3, padding=1, output_padding = 0),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded