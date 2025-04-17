class GCNN(nn.Module):
    def __init__(self, n_output=1, num_features_pro=1024, output_dim=128, dropout=0.2):
        super(GCNN, self).__init__()
        print('GCNN Loaded')
        self.n_output = n_output
        self.pro1_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro1_fc1 = nn.Linear(num_features_pro, output_dim)
        self.pro2_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro2_fc1 = nn.Linear(num_features_pro, output_dim)
        
        self.descriptor_dim = 80
        self.transformer_dim = 32 - 1
        self.reducer = nn.Linear(self.descriptor_dim, self.transformer_dim)

        # Transformer parameters
        
        self.nhead = 4
        self.num_layers = 2

        # Transformer encoders for masif descriptors
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.transformer_dim + 1,  # +1 for the indicator
                nhead=self.nhead,
                dim_feedforward=128,
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
        straight_indicator = torch.ones((*mas1_straight.shape[:-1], 1), device=mas1_straight.device)
        flipped_indicator = torch.zeros((*mas1_flipped.shape[:-1], 1), device=mas1_flipped.device)
        
        mas1_straight = self.reducer(mas1_straight)
        mas1_flipped = self.reducer(mas1_flipped)
        mas2_straight = self.reducer(mas2_straight)
        mas2_flipped = self.reducer(mas2_flipped)

        # Concatenate descriptors with indicators
        mas1_straight = torch.cat([mas1_straight, straight_indicator], dim=-1)
        mas1_flipped = torch.cat([mas1_flipped, flipped_indicator], dim=-1)
        mas2_straight = torch.cat([mas2_straight, straight_indicator], dim=-1)
        mas2_flipped = torch.cat([mas2_flipped, flipped_indicator], dim=-1)
        
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
        
        # Final prediction
        out = self.final_fc(combined)
        out = self.sigmoid(out)
        return out