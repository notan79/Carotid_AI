import torch.nn as nn
from uni import get_encoder

class Classifier(nn.Module):
    
    def __init__(self, in_features=1536, activation=nn.GELU()):
        super().__init__()
        self.in_features = in_features
        
        self.encoder, self.transform = get_encoder(enc_name='uni2-h')
        
        self.nn = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=1024),
            nn.BatchNorm1d(1024), 
            activation,
            
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(512), 
            activation,
            
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256), 
            activation,
            
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128), 
            activation,
            
            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(64), 
            activation,
            
            nn.Linear(in_features=64, out_features=32),
            nn.BatchNorm1d(32), 
            activation,
            
            nn.Linear(in_features=32, out_features=16),
            nn.BatchNorm1d(16), 
            activation,
            
            nn.Linear(in_features=16, out_features=8),
            nn.BatchNorm1d(8), 
            activation,
            
            nn.Linear(in_features=8, out_features=4),
            nn.BatchNorm1d(4), 
            activation,
            
            nn.Linear(in_features=4, out_features=2),
            nn.BatchNorm1d(2), 
            activation,
            
            nn.Linear(in_features=2, out_features=1),            
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.nn(x)
        