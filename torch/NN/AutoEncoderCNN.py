import torch
import torch.nn as nn
class AE(torch.nn.Module):
    def __init__(self, encoder, decoder, rand_mat):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.rand_mat = rand_mat
        
    def forward(self, x):
        enc = encoder(x)
        
        all_dps = []
        for batch in enc:
            all_dps.append(torch.mv(self.rand_mat, batch))
        Dp = torch.stack(all_dps)
        
        p_vec = F.softmax(Dp, dim = 1)
        
        
        
        dec = decoder(p_vec)
        return dec, p_vec
    
    

class Encoder(nn.Module):
    def __init__(self, in_channel=3, out_channel=3500, activation=nn.GELU()):
        super().__init__()
        # The input size is 3*299*299
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.encoder = nn.Sequential(
                                nn.Conv2d(in_channels=in_channel,out_channels=16,kernel_size=3,stride=1,padding=1,bias=False),
                                nn.BatchNorm2d(16), 
                                activation,
            
                                nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1,bias=False),
                                nn.BatchNorm2d(32), 
                                activation,
            
                                nn.Conv2d(in_channels=32,out_channels=32,kernel_size=2,stride=2,padding=1,bias=False),
                                nn.BatchNorm2d(32), 
                                activation,
            
                                nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False),
                                nn.BatchNorm2d(64), 
                                activation,
            
                                nn.Conv2d(in_channels=64,out_channels=64,kernel_size=2,stride=2,padding=1,bias=False),
                                nn.BatchNorm2d(64), 
                                activation,
            
                                nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1,bias=False),
                                nn.BatchNorm2d(128), 
                                activation,
            
                                nn.Conv2d(in_channels=128,out_channels=128,kernel_size=2,stride=2,padding=1,bias=False),
                                nn.BatchNorm2d(128), 
                                activation,
            
                                nn.Conv2d(in_channels=128,out_channels=320,kernel_size=3,stride=1,padding=1,bias=False),
                                nn.BatchNorm2d(320), 
                                activation,
            
                                nn.Conv2d(in_channels=320,out_channels=240,kernel_size=3,stride=1,padding=1,bias=False),
                                nn.BatchNorm2d(240), 
                                activation,
            
                                nn.Conv2d(in_channels=240,out_channels=150,kernel_size=3,stride=1,padding=1,bias=False),
                                nn.BatchNorm2d(150), 
                                activation,
            
                                nn.Conv2d(in_channels=150,out_channels=80,kernel_size=3,stride=2,padding=1,bias=False),
                                nn.BatchNorm2d(80), 
                                activation,
            
                                nn.Conv2d(in_channels=80,out_channels=40,kernel_size=3,stride=1,padding=1,bias=False),
                                nn.BatchNorm2d(40), 
                                activation,
            
                                nn.Conv2d(in_channels=40,out_channels=15,kernel_size=3,stride=1,padding=1,bias=False),
                                nn.BatchNorm2d(15), 
                                activation,
            
                                nn.Flatten(),
                                nn.Linear(in_features=15*20*20, out_features=out_channel),
                                nn.Softmax(dim=1))
        
    def forward(self, x):
        x = x.view(-1, self.in_channel, 299, 299)
        prob = self.encoder(x)
        return prob


class Decoder(nn.Module):
    def __init__(self, in_channel=3, out_channel=1600, activation=nn.GELU()):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.mlp = nn.Sequential(
                             nn.Linear(in_features=out_channel, out_features=15*20*20, bias=False),
                             nn.BatchNorm1d(20*20*15),
                             activation,
                             nn.Unflatten(1, (15, 20, 20)))
        
        self.decoder = nn.Sequential(
                                nn.ConvTranspose2d(in_channels=15,out_channels=40,kernel_size=3,stride=1,padding=1,output_padding=0,bias=False),
                                nn.BatchNorm2d(40), 
                                activation,
            
                                nn.ConvTranspose2d(in_channels=40,out_channels=80,kernel_size=3,stride=1,padding=1,output_padding=0,bias=False),
                                nn.BatchNorm2d(80), 
                                activation,
            
                                nn.ConvTranspose2d(in_channels=80,out_channels=150,kernel_size=3,stride=2,padding=1,output_padding=0,bias=False),
                                nn.BatchNorm2d(150), 
                                activation,
            
                                nn.ConvTranspose2d(in_channels=150,out_channels=240,kernel_size=3,stride=1,padding=1,output_padding=0,bias=False),
                                nn.BatchNorm2d(240), 
                                activation,
            
                                nn.ConvTranspose2d(in_channels=240,out_channels=320,kernel_size=3,stride=1,padding=1,output_padding=0,bias=False),
                                nn.BatchNorm2d(320), 
                                activation,
            
                                nn.ConvTranspose2d(in_channels=320,out_channels=128,kernel_size=3,stride=1,padding=1,output_padding=0,bias=False),
                                nn.BatchNorm2d(128), 
                                activation,
            
                                nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=3,stride=2,padding=1,output_padding=0,bias=False),
                                nn.BatchNorm2d(128), 
                                activation,
            
                                nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1,output_padding=0,bias=False),
                                nn.BatchNorm2d(64), 
                                activation,
            
                                nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=3,stride=2,padding=2,output_padding=0,bias=False),
                                nn.BatchNorm2d(64), 
                                activation,
            
                                nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=2,output_padding=0,bias=False),
                                nn.BatchNorm2d(32), 
                                activation,
            
                                nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=1,output_padding=0,bias=False),
                                nn.BatchNorm2d(32), 
                                activation,
            
                                nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=3,stride=1,padding=1,output_padding=0,bias=False),
                                nn.BatchNorm2d(16), 
                                activation,
            
                                nn.ConvTranspose2d(in_channels=16, out_channels=in_channel, kernel_size=3, stride=1, padding=0, output_padding=0),
                                nn.Sigmoid())
    
    def forward(self, x):
        x = x.view(-1, self.out_channel)
        temp = self.mlp(x)
        temp = temp.view(-1,15,20,20)
        output = self.decoder(temp)
        return output