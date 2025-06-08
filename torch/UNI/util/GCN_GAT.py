import torch
import torch_geometric
import torch_geometric.nn as geo_nn

class GCN_GAT(torch.nn.Module):
    def __init__(self, heads=4, activation=torch.nn.ReLU(), conv_activation=torch.relu):
        super(GCN_GAT, self).__init__()
        self.activation = activation
        self.conv_activation = conv_activation
        
        self.conv1 = geo_nn.GATv2Conv(in_channels=1536, out_channels=1024, heads=heads, concat=False)  # First layer
        self.conv2 = geo_nn.GATv2Conv(in_channels=1024, out_channels=512, heads=heads, concat=False)  # Second layer
        self.conv3 = geo_nn.GATv2Conv(in_channels=512, out_channels=256, heads=heads, concat=False)  # Third layer

        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=256, out_features=128),
            self.activation,
            
            torch.nn.Linear(in_features=128, out_features=64),
            self.activation,
            
            torch.nn.Linear(in_features=64, out_features=32),
            self.activation,
            
            torch.nn.Linear(in_features=32, out_features=16),
            self.activation,
            
            torch.nn.Linear(in_features=16, out_features=1)
        )
        
        # Learn which nodes are the most impactful
        self.gate_nn = torch.nn.Sequential(
            torch.nn.Linear(in_features=256, out_features=128),
            self.activation,
            
            torch.nn.Linear(in_features=128, out_features=1)
        )
        
        # Find the attention score 
        self.pool = geo_nn.aggr.AttentionalAggregation(self.gate_nn)
        
    def forward(self, data):
        # Extract Relevant Info
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Pass through GCN
        x = self.conv1(x, edge_index)
        x = self.conv_activation(x)
        

        x = self.conv2(x, edge_index)
        x = self.conv_activation(x)
        
        x = self.conv3(x, edge_index)
        x = self.conv_activation(x)
        
        # Get the graph embedding
        graph_emb = self.pool(x, batch) # outputs a matrix of (batch_size, 512)
        
        # Get the output logits vectors
        x = self.mlp(graph_emb)
        
        return x