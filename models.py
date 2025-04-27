# Building model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep,global_sort_pool
from torch_geometric.utils import dropout_adj
from torch.optim.lr_scheduler import MultiStepLR

class GCNN(nn.Module):
    def __init__(self, n_output=1, num_features_pro=1024, output_dim=128, dropout=0.2):
        super(GCNN, self).__init__()
        print('GCNN Loaded')
        self.n_output = n_output
        self.pro1_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro1_fc1 = nn.Linear(num_features_pro, output_dim)
        self.pro2_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro2_fc1 = nn.Linear(num_features_pro, output_dim)
        
        # Output processing
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Final layers - only using GNN outputs now
        combined_dim = 2 * output_dim
        self.final_fc = nn.Linear(combined_dim, self.n_output)

    def forward(self, pro1_data, pro2_data, mas1_straight, mas1_flipped, mas2_straight, mas2_flipped):
        # Process protein 1 with GNN
        pro1_x, pro1_edge_index, pro1_batch = pro1_data.x, pro1_data.edge_index, pro1_data.batch
        x = self.pro1_conv1(pro1_x, pro1_edge_index)
        x = self.relu(x)
        x = gep(x, pro1_batch)
        x = self.relu(self.pro1_fc1(x))
        x = self.dropout(x)

        # Process protein 2 with GNN
        pro2_x, pro2_edge_index, pro2_batch = pro2_data.x, pro2_data.edge_index, pro2_data.batch
        xt = self.pro2_conv1(pro2_x, pro2_edge_index)
        xt = self.relu(xt)
        xt = gep(xt, pro2_batch)
        xt = self.relu(self.pro2_fc1(xt))
        xt = self.dropout(xt)

        # Concatenate GNN features only
        combined = torch.cat([x, xt], dim=1)
        
        # Final prediction (logits)
        out = self.final_fc(combined)
        return out


class GCNN_with_descriptors(nn.Module):
    def __init__(self, n_output=1, num_features_pro=1024, output_dim=128, dropout=0.2, descriptor_dim=80, transformer_dim=32, nhead=4, num_layers=2, dim_feedforward=128):
        super(GCNN_with_descriptors, self).__init__()
        print('GCNN Loaded')
        self.n_output = n_output
        self.pro1_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro1_fc1 = nn.Linear(num_features_pro, output_dim)
        self.pro2_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro2_fc1 = nn.Linear(num_features_pro, output_dim)
        
        self.descriptor_dim = descriptor_dim
        self.transformer_dim = transformer_dim
        self.reducer = nn.Linear(self.descriptor_dim, self.transformer_dim)

        # Transformer parameters
        
        self.nhead = nhead
        self.num_layers = num_layers

        # Transformer encoders for masif descriptors
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.transformer_dim + 1,  # +1 for the indicator
                nhead=self.nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ),
            num_layers=self.num_layers
        )
        
        # Output processing
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        
        # Final layers
        combined_dim = 2 * output_dim + 2 * (self.transformer_dim + 1)
        self.final_fc = nn.Linear(combined_dim, self.n_output)

    def forward(self, pro1_data, pro2_data, mas1_straight, mas1_flipped, mas2_straight, mas2_flipped):
        # Process protein 1 with GNN
        pro1_x, pro1_edge_index, pro1_batch = pro1_data.x, pro1_data.edge_index, pro1_data.batch
        x = self.pro1_conv1(pro1_x, pro1_edge_index)
        x = self.relu(x)
        x = gep(x, pro1_batch)
        x = self.relu(self.pro1_fc1(x))
        x = self.dropout(x)

        # Process protein 2 with GNN
        pro2_x, pro2_edge_index, pro2_batch = pro2_data.x, pro2_data.edge_index, pro2_data.batch
        xt = self.pro2_conv1(pro2_x, pro2_edge_index)
        xt = self.relu(xt)
        xt = gep(xt, pro2_batch)
        xt = self.relu(self.pro2_fc1(xt))
        xt = self.dropout(xt)

        # Process masif descriptors with transformers
        # Add indicator (0/1) for straight/flipped
        batch_size = mas1_straight.size(0)
        
        # Prepare indicators
        straight_indicator_mas1 = torch.ones((*mas1_straight.shape[:-1], 1), device=mas1_straight.device)
        flipped_indicator_mas1 = torch.zeros((*mas1_flipped.shape[:-1], 1), device=mas1_flipped.device)

        straight_indicator_mas2 = torch.ones((*mas2_straight.shape[:-1], 1), device=mas2_straight.device)
        flipped_indicator_mas2 = torch.zeros((*mas2_flipped.shape[:-1], 1), device=mas2_flipped.device)
        
        mas1_straight = self.reducer(mas1_straight)
        mas1_flipped = self.reducer(mas1_flipped)
        mas2_straight = self.reducer(mas2_straight)
        mas2_flipped = self.reducer(mas2_flipped)

        # Concatenate descriptors with indicators
        mas1_straight = torch.cat([mas1_straight, straight_indicator_mas1], dim=-1)
        mas1_flipped = torch.cat([mas1_flipped, flipped_indicator_mas1], dim=-1)
        mas2_straight = torch.cat([mas2_straight, straight_indicator_mas2], dim=-1)
        mas2_flipped = torch.cat([mas2_flipped, flipped_indicator_mas2], dim=-1)
        
        # Process through transformers
        # Combine straight and flipped for each protein
        mas1 = torch.cat([mas1_straight, mas1_flipped], dim=1)
        mas2 = torch.cat([mas2_straight, mas2_flipped], dim=1)
        
        # Transform sequences (B, L, D) -> (L, B, D) for transformer
        mas1 = mas1.transpose(0, 1)
        mas2 = mas2.transpose(0, 1)

        # Apply transformers
        mas1_out = self.transformer(mas1)
        mas2_out = self.transformer(mas2)
        
        # Get mean of transformer outputs for global representation
        mas1_out = mas1_out.mean(dim=0)
        mas2_out = mas2_out.mean(dim=0)

        # Concatenate all features
        combined = torch.cat([x, xt, mas1_out, mas2_out], dim=1)
        
        # Final prediction (logits)
        out = self.final_fc(combined)
        return out

class GCNN_mutual_attention(nn.Module):
    def __init__(self, n_output=1, num_features_pro=1024, output_dim=128, dropout=0.2, descriptor_dim=80, transformer_dim=32, nhead=4, num_layers=2, dim_feedforward=128):
        super(GCNN_mutual_attention, self).__init__()
        print('GCNN Loaded')
        self.n_output = n_output
        self.pro1_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro1_fc1 = nn.Linear(num_features_pro, output_dim)
        self.pro2_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro2_fc1 = nn.Linear(num_features_pro, output_dim)
        
        self.descriptor_dim = descriptor_dim
        self.transformer_dim = transformer_dim
        self.reducer = nn.Linear(self.descriptor_dim, self.transformer_dim - 2)

        # Transformer parameters
        
        self.nhead = nhead
        self.num_layers = num_layers

        # Transformer encoders for masif descriptors
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.transformer_dim,
                nhead=self.nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ),
            num_layers=self.num_layers
        )

        # Output processing
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        
        # Final layers
        combined_dim = 2 * output_dim + self.transformer_dim
        self.final_fc = nn.Linear(combined_dim, self.n_output)

    def forward(self, pro1_data, pro2_data, mas1_straight, mas1_flipped, mas2_straight, mas2_flipped):
        # Process protein 1 with GNN
        pro1_x, pro1_edge_index, pro1_batch = pro1_data.x, pro1_data.edge_index, pro1_data.batch
        x = self.pro1_conv1(pro1_x, pro1_edge_index)
        x = self.relu(x)
        x = gep(x, pro1_batch)
        x = self.relu(self.pro1_fc1(x))
        x = self.dropout(x)

        # Process protein 2 with GNN
        pro2_x, pro2_edge_index, pro2_batch = pro2_data.x, pro2_data.edge_index, pro2_data.batch
        xt = self.pro2_conv1(pro2_x, pro2_edge_index)
        xt = self.relu(xt)
        xt = gep(xt, pro2_batch)
        xt = self.relu(self.pro2_fc1(xt))
        xt = self.dropout(xt)

        # Process masif descriptors with transformers
        # Add indicator (0/1) for straight/flipped
        batch_size = mas1_straight.size(0)
        
        # Prepare indicators
        straight_indicator_mas1 = torch.ones((*mas1_straight.shape[:-1], 1), device=mas1_straight.device)
        flipped_indicator_mas1 = torch.zeros((*mas1_flipped.shape[:-1], 1), device=mas1_flipped.device)

        straight_indicator_mas2 = torch.ones((*mas2_straight.shape[:-1], 1), device=mas2_straight.device)
        flipped_indicator_mas2 = torch.zeros((*mas2_flipped.shape[:-1], 1), device=mas2_flipped.device)
        
        first_indicator = torch.ones((*mas1_straight.shape[:-1], 1), device=mas1_straight.device)
        second_indicator = torch.zeros((*mas2_flipped.shape[:-1], 1), device=mas2_flipped.device)

        mas1_straight = self.reducer(mas1_straight)
        mas1_flipped = self.reducer(mas1_flipped)
        mas2_straight = self.reducer(mas2_straight)
        mas2_flipped = self.reducer(mas2_flipped)

        # Concatenate descriptors with indicators
        mas1_straight = torch.cat([mas1_straight, straight_indicator_mas1, first_indicator], dim=-1)
        mas1_flipped = torch.cat([mas1_flipped, flipped_indicator_mas1, first_indicator], dim=-1)
        mas2_straight = torch.cat([mas2_straight, straight_indicator_mas2, second_indicator], dim=-1)
        mas2_flipped = torch.cat([mas2_flipped, flipped_indicator_mas2, second_indicator], dim=-1)
        
        # Process through transformers
        # Combine straight and flipped for each protein
        mas = torch.cat([mas1_straight, mas1_flipped, mas2_straight, mas2_flipped], dim=1)
        
        # Transform sequences (B, L, D) -> (L, B, D) for transformer
        mas = mas.transpose(0, 1)

        # Apply transformers
        mas_out = self.transformer(mas)
        
        # Get mean of transformer outputs for global representation
        mas_out = mas_out.mean(dim=0)

        # Concatenate all features
        combined = torch.cat([x, xt, mas_out], dim=1)
        
        # Final prediction (logits)
        out = self.final_fc(combined)
        return out


net = GCNN()
print(net)

"""# GAT"""

class AttGNN(nn.Module):
    def __init__(self, n_output=1, num_features_pro= 1024, output_dim=128, dropout=0.2, heads = 1 ):
        super(AttGNN, self).__init__()

        print('AttGNN Loaded')

        self.hidden = 8
        self.heads = 1
        
        # for protein 1
        self.pro1_conv1 = GATConv(num_features_pro, self.hidden* 16, heads=self.heads, dropout=0.2)
        self.pro1_fc1 = nn.Linear(128, output_dim)


        # for protein 2
        self.pro2_conv1 = GATConv(num_features_pro, self.hidden*16, heads=self.heads, dropout=0.2)
        self.pro2_fc1 = nn.Linear(128, output_dim)

        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(64, n_output)
        


    def forward(self, pro1_data, pro2_data):

        # get graph input for protein 1 
        pro1_x, pro1_edge_index, pro1_batch = pro1_data.x, pro1_data.edge_index, pro1_data.batch
        # get graph input for protein 2
        pro2_x, pro2_edge_index, pro2_batch = pro2_data.x, pro2_data.edge_index, pro2_data.batch
         
        
        x = self.pro1_conv1(pro1_x, pro1_edge_index)
        x = self.relu(x)
        
	# global pooling
        x = gep(x, pro1_batch)  
       
        # flatten
        x = self.relu(self.pro1_fc1(x))
        x = self.dropout(x)



        xt = self.pro2_conv1(pro2_x, pro2_edge_index)
        xt = self.relu(self.pro2_fc1(xt))
	
	# global pooling
        xt = gep(xt, pro2_batch)  

        # flatten
        xt = self.relu(xt)
        xt = self.dropout(xt)

	
	# Concatenation
        xc = torch.cat((x, xt), 1)

        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        out = self.sigmoid(out)
        return out

net_GAT = AttGNN()
print(net_GAT)

