import torch
import torch_geometric
import torch_geometric.nn as geo_nn

class GCN_GAT(torch.nn.Module):
    # Note: bias=False since Batch Norms have bias term built in
    def __init__(self, heads=4, activation=torch.nn.ReLU(), conv_activation=torch.relu, dropout=0., gat_dropout=0.):
        super(GCN_GAT, self).__init__()
        self.activation = activation
        self.conv_activation = conv_activation
        self.gat_dropout = gat_dropout
        self.dropout_rate = dropout
        
        self.conv1 = geo_nn.GATv2Conv(in_channels=1536, out_channels=1024, heads=heads, dropout=gat_dropout, concat=False, bias=False)  # First layer
        self.conv2 = geo_nn.GATv2Conv(in_channels=1024, out_channels=512, heads=heads, dropout=gat_dropout, concat=False, bias=False)  # Second layer
        self.conv3 = geo_nn.GATv2Conv(in_channels=512, out_channels=256, heads=heads, dropout=gat_dropout, concat=False, bias=False)  # Third layer

        self.batch_g1 = geo_nn.norm.BatchNorm(1024)
        self.batch_g2 = geo_nn.norm.BatchNorm(512)
        self.batch_g3 = geo_nn.norm.BatchNorm(256)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=256, out_features=128, bias=False),
            torch.nn.LayerNorm(128),
            self.activation,
            
            torch.nn.Linear(in_features=128, out_features=64, bias=False),
            torch.nn.LayerNorm(64),
            self.activation,
            
            torch.nn.Linear(in_features=64, out_features=32, bias=False),
            torch.nn.LayerNorm(32),
            self.activation,
            
            torch.nn.Linear(in_features=32, out_features=16, bias=False),
            torch.nn.LayerNorm(16),
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
        
        # Node level dropout
        self.dropout = torch.nn.Dropout(p=dropout)
        
    def set_gat_dropout(self, p):
        self.gat_dropout = p
        
        self.conv1.dropout = p
        self.conv2.dropout = p
        self.conv3.dropout = p
   
    def set_dropout(self, p):
        self.dropout_rate = p
        
        self.dropout.p = p


        
    def forward(self, data):
        # Extract Relevant Info
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Pass through GCN
        x = self.conv1(x, edge_index)
#         x = self.batch_g1(x)
        x = self.conv_activation(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
#         x = self.batch_g2(x)
        x = self.conv_activation(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
#         x = self.batch_g3(x)
        x = self.conv_activation(x)
        x = self.dropout(x)
        
        # Get the graph embedding
        graph_emb = self.pool(x, batch) # outputs a matrix of (batch_size, 512)
        
        # Get the output logits vectors
        x = self.mlp(graph_emb)
        
        return x