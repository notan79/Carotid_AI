import torch
class AE(torch.nn.Module):
    def __init__(self, input_shape, hidden_layer, encode_dim):
        super().__init__()
        
        self._hidden_layer = hidden_layer
        self._encode_dim = encode_dim
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_shape, self._hidden_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(self._hidden_layer, self._encode_dim),
            torch.nn.ReLU()

#               torch.nn.Linear(input_shape, self._encode_dim)
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self._encode_dim, self._hidden_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(self._hidden_layer, input_shape),
            torch.nn.Sigmoid()
#             torch.nn.Linear(self._encode_dim, input_shape),
#             torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded