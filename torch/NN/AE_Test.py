import torch
class AE_CNN(torch.nn.Module):
    def __init__(self, dim1:int=64, dim2:int=32, dim3:int=22, encoded_dim:int=2048, rand_mat_dim:int=1024, rand_mat = True) -> None:
        super().__init__()
        
                
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3,dim1,8,stride=2,padding=1), # outputs dim1, 147, 147
            torch.nn.ReLU(),
            torch.nn.Conv2d(dim1,dim2,8,stride=2,padding=1), # outputs dim2, 71, 71
            torch.nn.ReLU(),
            torch.nn.Conv2d(dim2,dim3,16,stride=4,padding=1), # outputs dim3, 15, 15
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(dim3*15*15, dim3*15*15)
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Unflatten(1, (dim3, 15, 15)),
            torch.nn.ConvTranspose2d(dim3,dim2,16,stride=4,padding=1, output_padding = 1), 
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(dim2,dim1,8, stride=2, padding=1, output_padding = 1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(dim1,3,8, stride=2, padding=1, output_padding = 1),    
            torch.nn.Sigmoid()
        )
        
        
    def forward(self, x) -> torch.tensor:
        enc = self.encoder(x)
        decoded = self.decoder(enc)
        return decoded