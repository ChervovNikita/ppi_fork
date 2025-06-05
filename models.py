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

class GCNN_desc_pool(nn.Module):
    def __init__(self, n_output=1, num_features_pro=1024, output_dim=128, dropout=0.2, descriptor_dim=80):
        super(GCNN_desc_pool, self).__init__()
        print('GCNN Loaded')
        self.n_output = n_output
        self.pro1_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro1_fc1 = nn.Linear(num_features_pro, output_dim)
        self.pro2_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro2_fc1 = nn.Linear(num_features_pro, output_dim)
        
        self.conv1d_mas1_straight = nn.Conv1d(descriptor_dim, output_dim, kernel_size=1)
        self.conv1d_mas1_flipped = nn.Conv1d(descriptor_dim, output_dim, kernel_size=1)
        self.conv1d_mas2_straight = nn.Conv1d(descriptor_dim, output_dim, kernel_size=1)
        self.conv1d_mas2_flipped = nn.Conv1d(descriptor_dim, output_dim, kernel_size=1)
        
        # Max pooling
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Output processing
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Final layers - combining graph and descriptor features
        combined_dim = 2 * output_dim + 4 * output_dim  # 2 graph outputs + 4 descriptor outputs
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

        # Process descriptors with 1D convolutions
        # Transpose for conv1d (B, L, D) -> (B, D, L)
        mas1_straight = mas1_straight.transpose(1, 2)
        mas1_flipped = mas1_flipped.transpose(1, 2)
        mas2_straight = mas2_straight.transpose(1, 2)
        mas2_flipped = mas2_flipped.transpose(1, 2)

        # Apply convolutions and pooling
        mas1_straight = self.max_pool(self.relu(self.conv1d_mas1_straight(mas1_straight))).squeeze(-1)
        mas1_flipped = self.max_pool(self.relu(self.conv1d_mas1_flipped(mas1_flipped))).squeeze(-1)
        mas2_straight = self.max_pool(self.relu(self.conv1d_mas2_straight(mas2_straight))).squeeze(-1)
        mas2_flipped = self.max_pool(self.relu(self.conv1d_mas2_flipped(mas2_flipped))).squeeze(-1)

        print(mas1_straight.shape)

        # Concatenate all features
        combined = torch.cat([x, xt, mas1_straight, mas1_flipped, mas2_straight, mas2_flipped], dim=1)
        
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

class GCNN_geom_transformer(nn.Module):
    def __init__(self, n_output=1, num_features_pro=1024, output_dim=128, dropout=0.2, descriptor_dim=80, transformer_dim=128, nhead=8, num_layers=2, dim_feedforward=256):
        super(GCNN_geom_transformer, self).__init__()
        print('GCNN_geom_transformer Loaded')
        self.n_output = n_output
        self.transformer_dim = transformer_dim
        
        # GCN layers - same as other models
        self.pro1_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro1_fc1 = nn.Linear(num_features_pro, output_dim)
        self.pro2_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro2_fc1 = nn.Linear(num_features_pro, output_dim)
        
        # 6 linear layers as specified
        self.node_proj_pro1 = nn.Linear(output_dim, transformer_dim)           # For graph nodes of first protein
        self.node_proj_pro2 = nn.Linear(output_dim, transformer_dim)           # For graph nodes of second protein
        self.desc_proj_mas1_straight = nn.Linear(descriptor_dim, transformer_dim)  # For straight descriptors of first protein
        self.desc_proj_mas1_flipped = nn.Linear(descriptor_dim, transformer_dim)   # For flipped descriptors of first protein
        self.desc_proj_mas2_straight = nn.Linear(descriptor_dim, transformer_dim)  # For straight descriptors of second protein
        self.desc_proj_mas2_flipped = nn.Linear(descriptor_dim, transformer_dim)   # For flipped descriptors of second protein
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_dim))
        
        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=transformer_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Output processing
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Final prediction layer using CLS token
        self.final_fc = nn.Linear(transformer_dim, n_output)

    def forward(self, pro1_data, pro2_data, mas1_straight, mas1_flipped, mas2_straight, mas2_flipped, 
                mas1_straight_lengths=None, mas1_flipped_lengths=None, mas2_straight_lengths=None, mas2_flipped_lengths=None):
        # Process protein 1 with GNN - same logic as other models
        pro1_x, pro1_edge_index, pro1_batch = pro1_data.x, pro1_data.edge_index, pro1_data.batch
        pro1_nodes = self.pro1_conv1(pro1_x, pro1_edge_index)
        pro1_nodes = self.relu(pro1_nodes)
        # Apply linear layer to each node (instead of global pooling)
        pro1_nodes = self.relu(self.pro1_fc1(pro1_nodes))
        pro1_nodes = self.dropout(pro1_nodes)
        pro1_nodes = self.node_proj_pro1(pro1_nodes)  # Project to transformer dim
        
        # Process protein 2 with GNN - same logic as other models  
        pro2_x, pro2_edge_index, pro2_batch = pro2_data.x, pro2_data.edge_index, pro2_data.batch
        pro2_nodes = self.pro2_conv1(pro2_x, pro2_edge_index)
        pro2_nodes = self.relu(pro2_nodes)
        # Apply linear layer to each node (instead of global pooling)
        pro2_nodes = self.relu(self.pro2_fc1(pro2_nodes))
        pro2_nodes = self.dropout(pro2_nodes)
        pro2_nodes = self.node_proj_pro2(pro2_nodes)  # Project to transformer dim
        
        # Process masif descriptors
        batch_size = mas1_straight.size(0)
        
        # Project descriptors through separate linear layers
        mas1_straight_proj = self.desc_proj_mas1_straight(mas1_straight)    # (B, L, transformer_dim)
        mas1_flipped_proj = self.desc_proj_mas1_flipped(mas1_flipped)       # (B, L, transformer_dim)
        mas2_straight_proj = self.desc_proj_mas2_straight(mas2_straight)    # (B, L, transformer_dim)
        mas2_flipped_proj = self.desc_proj_mas2_flipped(mas2_flipped)       # (B, L, transformer_dim)
        
        # Create sequences for each sample in the batch
        sequences = []
        attention_masks = []
        
        # Get node counts per sample
        pro1_node_counts = torch.bincount(pro1_batch, minlength=batch_size)
        pro2_node_counts = torch.bincount(pro2_batch, minlength=batch_size)
        
        for i in range(batch_size):
            # Get nodes for current sample
            pro1_mask = (pro1_batch == i)
            pro2_mask = (pro2_batch == i)
            
            pro1_sample_nodes = pro1_nodes[pro1_mask]  # Shape: (num_nodes_pro1, transformer_dim)
            pro2_sample_nodes = pro2_nodes[pro2_mask]  # Shape: (num_nodes_pro2, transformer_dim)
            
            # Get descriptors for current sample - handle both padded and unpadded cases
            if mas1_straight_lengths is not None:
                # Unpadded mode: use only the real (non-padded) portions
                mas1_straight_len = mas1_straight_lengths[i].item()
                mas1_flipped_len = mas1_flipped_lengths[i].item()
                mas2_straight_len = mas2_straight_lengths[i].item()
                mas2_flipped_len = mas2_flipped_lengths[i].item()
                
                mas1_straight_sample = mas1_straight_proj[i][:mas1_straight_len]  # Shape: (real_len, transformer_dim)
                mas1_flipped_sample = mas1_flipped_proj[i][:mas1_flipped_len]     # Shape: (real_len, transformer_dim)
                mas2_straight_sample = mas2_straight_proj[i][:mas2_straight_len]  # Shape: (real_len, transformer_dim)
                mas2_flipped_sample = mas2_flipped_proj[i][:mas2_flipped_len]     # Shape: (real_len, transformer_dim)
            else:
                # Padded mode: use entire sequences (original behavior)
                mas1_straight_sample = mas1_straight_proj[i]  # Shape: (seq_len, transformer_dim)
                mas1_flipped_sample = mas1_flipped_proj[i]    # Shape: (seq_len, transformer_dim)
                mas2_straight_sample = mas2_straight_proj[i]  # Shape: (seq_len, transformer_dim)
                mas2_flipped_sample = mas2_flipped_proj[i]    # Shape: (seq_len, transformer_dim)
            
            # Combine all elements for this sample: nodes first, then descriptors
            sequence = torch.cat([
                pro1_sample_nodes,
                pro2_sample_nodes,
                mas1_straight_sample,
                mas1_flipped_sample,
                mas2_straight_sample,
                mas2_flipped_sample
            ], dim=0)  # Shape: (total_seq_len, transformer_dim)
            
            sequences.append(sequence)
            
            # Create attention mask (all positions are valid for this sample)
            attention_mask = torch.ones(sequence.size(0), device=sequence.device)
            attention_masks.append(attention_mask)
        
        # Pad sequences to same length for batching
        max_len = max(seq.size(0) for seq in sequences)
        padded_sequences = []
        padded_masks = []
        
        for seq, mask in zip(sequences, attention_masks):
            seq_len = seq.size(0)
            if seq_len < max_len:
                # Pad with zeros
                padding = torch.zeros(max_len - seq_len, self.transformer_dim, device=seq.device)
                padded_seq = torch.cat([seq, padding], dim=0)
                
                # Pad mask with zeros (0 = padding)
                mask_padding = torch.zeros(max_len - seq_len, device=mask.device)
                padded_mask = torch.cat([mask, mask_padding], dim=0)
            else:
                padded_seq = seq
                padded_mask = mask
            
            padded_sequences.append(padded_seq)
            padded_masks.append(padded_mask)
        
        # Stack into batch
        sequence_batch = torch.stack(padded_sequences, dim=0)  # Shape: (batch_size, max_len, transformer_dim)
        attention_mask = torch.stack(padded_masks, dim=0)      # Shape: (batch_size, max_len)
        
        # Add CLS token to the beginning of each sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, transformer_dim)
        sequence_batch = torch.cat([cls_tokens, sequence_batch], dim=1)  # Shape: (batch_size, max_len+1, transformer_dim)
        
        # Update attention mask for CLS token
        cls_mask = torch.ones(batch_size, 1, device=attention_mask.device)
        attention_mask = torch.cat([cls_mask, attention_mask], dim=1)  # Shape: (batch_size, max_len+1)
        
        # Create padding mask for transformer (True for padding positions)
        src_key_padding_mask = (attention_mask == 0)
        
        # Apply transformer
        transformer_output = self.transformer(
            sequence_batch,
            src_key_padding_mask=src_key_padding_mask
        )  # Shape: (batch_size, max_len+1, transformer_dim)
        
        # Extract CLS token representation (first token)
        cls_output = transformer_output[:, 0, :]  # Shape: (batch_size, transformer_dim)
        
        # Final prediction using CLS token
        out = self.final_fc(cls_output)
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

