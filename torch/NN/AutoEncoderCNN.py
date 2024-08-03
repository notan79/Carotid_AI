import torch
class AE_CNN(torch.nn.Module):
    def __init__(self, dim1:int=64, dim2:int=32, dim3:int=16, encoded_dim:int=2048, rand_mat_dim:int=1024, rand_mat = True) -> None:
        super().__init__()
        
        self.encoded_vector = None
        
        if rand_mat: 
            self.rand_mat = self.create_rand_mat(rand_mat_dim, encoded_dim)
        else:
            self.rand_mat = torch.randn(rand_mat_dim, encoded_dim, requires_grad=False, device='cuda' if torch.cuda.is_available() else 'cpu') # dummy rand matrix of correct dim
        
        self.softmax = torch.nn.Softmax()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3,dim1,8,stride=2,padding=1), # outputs dim1, 147, 147
            torch.nn.ReLU(),
            torch.nn.Conv2d(dim1,dim2,8,stride=2,padding=1), # outputs dim2, 71, 71
            torch.nn.ReLU(),
            torch.nn.Conv2d(dim2,dim3,16,stride=4,padding=1), # outputs dim3, 15, 15
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(dim3*15*15, dim3*15*15),
            torch.nn.ReLU(),
            torch.nn.Linear(dim3*15*15, encoded_dim)
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(rand_mat_dim, encoded_dim), # rand_mat_dim -> encoded_dim
            torch.nn.ReLU(),
            torch.nn.Linear(encoded_dim, dim3*15*15),# encoded_dim -> 'flattened dim'
            torch.nn.ReLU(),
            torch.nn.Unflatten(1, (dim3, 15, 15)),
            torch.nn.ConvTranspose2d(dim3,dim2,16,stride=4,padding=1, output_padding = 1), 
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(dim2,dim1,8, stride=2, padding=1, output_padding = 1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(dim1,3,8, stride=2, padding=1, output_padding = 1),    
            torch.nn.Sigmoid()
        )
        
    def create_rand_mat(self, rand_mat_dim:int, encoded_dim:int) -> torch.tensor:
        y = torch.randn(rand_mat_dim, encoded_dim, requires_grad=False, device='cuda' if torch.cuda.is_available() else 'cpu')
        norm = torch.norm(y, dim = 0)
        for col in range(len(y[0])):
            for row in range(len(y)):
                y[row][col] /= norm[col]
        return y
        
    def forward(self, x) -> torch.tensor:
        self.encoded_vector = self.softmax(self.encoder(x))
        all_dps = []
        for batch in self.encoded_vector:
            all_dps.append(torch.mv(self.rand_mat, batch))
        Dp = torch.stack(all_dps)
        decoded = self.decoder(Dp)
        return decoded