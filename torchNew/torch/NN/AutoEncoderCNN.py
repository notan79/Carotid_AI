import torch
class AE_CNN(torch.nn.Module):
    def __init__(self, encoded_dim:int=2048, rand_mat_dim:int=1024, rand_mat = True) -> None:
        super().__init__()
        
        self.encoded_vector = None
        
        if rand_mat: 
            self.rand_mat = self.create_rand_mat(rand_mat_dim, encoded_dim)
        else:
            self.rand_mat = torch.randn(rand_mat_dim, encoded_dim, requires_grad=False, device='cuda' if torch.cuda.is_available() else 'cpu') # dummy rand matrix of correct dim
        
        self.softmax = torch.nn.Softmax(dim=1)
        
        self.encoder = Encoder(encoded_dim, rand_mat_dim)
        
        self.decoder = Decoder(rand_mat_dim)
        
    def create_rand_mat(self, rand_mat_dim:int, encoded_dim:int) -> torch.tensor:
        y = torch.randn(rand_mat_dim, encoded_dim, requires_grad=False, device='cuda' if torch.cuda.is_available() else 'cpu')
        norm = torch.norm(y, dim = 0)
        y = y.div(norm)
        return y
        
    def forward(self, x) -> torch.tensor:
        self.encoded_vector = self.softmax(self.encoder(x))
        all_dps = []
        for batch in self.encoded_vector:
            all_dps.append(torch.mv(self.rand_mat, batch))
        Dp = torch.stack(all_dps)
        decoded = self.decoder(Dp)
        return decoded
    
    

class Encoder(torch.nn.Module):
    def __init__(self, encoded_dim: int, rand_mat_dim: int) -> None:
        super().__init__()

        self.nn = torch.nn.Sequential(
            torch.nn.Conv2d(3,16,3,stride=1,padding=1, bias=False), # Outputs: 299 x 299 x 16
            torch.nn.BatchNorm2d(16), torch.nn.ReLU(),
            
            torch.nn.Conv2d(16,32,3,stride=1,padding=1, bias=False), # Outputs: 299 x 299 x 32
            torch.nn.BatchNorm2d(32), torch.nn.ReLU(),
            
            torch.nn.AvgPool2d(2,stride=2, padding=1), # Outputs: 150 x 150 x 32
            torch.nn.Conv2d(32,64,3,stride=1,padding=1, bias=False), # Outputs: 150 x 150 x 64
            torch.nn.BatchNorm2d(64), torch.nn.ReLU(),
            
            torch.nn.AvgPool2d(2,stride=2, padding=1), # Outputs: 76 x 76 x 64
            torch.nn.Conv2d(64,128,3,stride=1,padding=1, bias=False), # Outputs: 76 x 76 x 128
            torch.nn.BatchNorm2d(128), torch.nn.ReLU(),
            
            torch.nn.AvgPool2d(2,stride=2, padding=1), # Outputs: 39 x 39 x 128
            torch.nn.Conv2d(128,320,3,stride=1,padding=1, bias=False), # Outputs: 39 x 39 x 320
            torch.nn.BatchNorm2d(320), torch.nn.ReLU(),
            
            torch.nn.Conv2d(320,240,3,stride=1,padding=1, bias=False), # Outputs: 39 x 39 x 240
            torch.nn.BatchNorm2d(240), torch.nn.ReLU(),
            
            torch.nn.Conv2d(240,150,3,stride=1,padding=1, bias=False), # Outputs: 39 x 39 x 150
            torch.nn.BatchNorm2d(150), torch.nn.ReLU(),
            
            torch.nn.Conv2d(150,80,3,stride=2,padding=1, bias=False), # Outputs: 20 x 20 x 80
            torch.nn.BatchNorm2d(80), torch.nn.ReLU(),
            
            torch.nn.Conv2d(80,40,3,stride=1,padding=1, bias=False), # Outputs: 20 x 20 x 40
            torch.nn.BatchNorm2d(40), torch.nn.ReLU(),
            
            torch.nn.Conv2d(40,15,3,stride=1,padding=1, bias=False), # Outputs: 20 x 20 x 15
            torch.nn.BatchNorm2d(15), torch.nn.ReLU(),
            
            torch.nn.Flatten(),
            torch.nn.Linear(20*20*15, encoded_dim)
        )
        
    def forward(self, x) -> torch.tensor:
        return self.nn(x)
    
class Decoder(torch.nn.Module):
    def __init__(self, rand_mat_dim: int) -> None:
        super().__init__()

        self.nn = torch.nn.Sequential(
                torch.nn.Linear(rand_mat_dim, 20*20*15), # rand_mat_dim -> flattened dim
                torch.nn.Unflatten(1, (15, 20, 20)),

                torch.nn.ConvTranspose2d(15,40,3, stride=1, padding=1, bias=False), 
                torch.nn.BatchNorm2d(40), torch.nn.ReLU(),

                torch.nn.ConvTranspose2d(40,80,3, stride=1, padding=1, bias=False), 
                torch.nn.BatchNorm2d(80), torch.nn.ReLU(),

                torch.nn.ConvTranspose2d(80,150,3, stride=2, padding=1, bias=False), 
                torch.nn.BatchNorm2d(150), torch.nn.ReLU(),

                torch.nn.ConvTranspose2d(150,240,3, stride=1, padding=1, bias=False), 
                torch.nn.BatchNorm2d(240), torch.nn.ReLU(),

                torch.nn.ConvTranspose2d(240,320,3, stride=1, padding=1, bias=False), 
                torch.nn.BatchNorm2d(320), torch.nn.ReLU(),

                torch.nn.ConvTranspose2d(320,128,3, stride=1, padding=1, bias=False), 
                torch.nn.BatchNorm2d(128), torch.nn.ReLU(),

                torch.nn.ConvTranspose2d(128,128,3, stride=2, padding=1, bias=False), 
                torch.nn.BatchNorm2d(128), torch.nn.ReLU(), # maybe remove batch norm or both?

                torch.nn.ConvTranspose2d(128,64,3, stride=1, padding=1, bias=False), 
                torch.nn.BatchNorm2d(64), torch.nn.ReLU(),

                torch.nn.ConvTranspose2d(64,64,3, stride=2, padding=2, bias=False), 
                torch.nn.BatchNorm2d(64), torch.nn.ReLU(),

                torch.nn.ConvTranspose2d(64,32,3, stride=1, padding=2, bias=False), 
                torch.nn.BatchNorm2d(32), torch.nn.ReLU(),

                torch.nn.ConvTranspose2d(32,32,3, stride=2, padding=1, bias=False), 
                torch.nn.BatchNorm2d(32), torch.nn.ReLU(),

                torch.nn.ConvTranspose2d(32,16,3, stride=1, padding=1, bias=False), 
                torch.nn.BatchNorm2d(16), torch.nn.ReLU(),

                torch.nn.ConvTranspose2d(16,3,3, stride=1, padding=0, bias=False), 
                torch.nn.BatchNorm2d(3), torch.nn.Sigmoid() # maybe remove batch norm
            )
    
    def forward(self, x) -> torch.tensor:
        return self.nn(x)