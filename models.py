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
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(2 * output_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.masif_straight_1 = nn.Conv1d(1, 1, 1)
        self.masif_flipped_1 = nn.Conv1d(1, 1, 1)
        self.masif_straight_2 = nn.Conv1d(1, 1, 1)
        self.masif_flipped_2 = nn.Conv1d(1, 1, 1)
        self.masif_pool_1 = nn.AdaptiveAvgPool1d(80)
        self.masif_pool_2 = nn.AdaptiveAvgPool1d(80)
        self.masif_fc1 = nn.Linear(80, 64)
        self.masif_fc2 = nn.Linear(80, 64)
        self.out = nn.Linear(64 + 128, self.n_output)

    def forward(self, pro1_data, pro2_data, mas1_straight, mas1_flipped, mas2_straight, mas2_flipped):
        pro1_x, pro1_edge_index, pro1_batch = pro1_data.x, pro1_data.edge_index, pro1_data.batch
        pro2_x, pro2_edge_index, pro2_batch = pro2_data.x, pro2_data.edge_index, pro2_data.batch
        x = self.pro1_conv1(pro1_x, pro1_edge_index)
        x = self.relu(x)
        x = gep(x, pro1_batch)
        x = self.relu(self.pro1_fc1(x))
        x = self.dropout(x)
        xt = self.pro2_conv1(pro2_x, pro2_edge_index)
        xt = self.relu(xt)
        xt = gep(xt, pro2_batch)
        xt = self.relu(self.pro2_fc1(xt))
        xt = self.dropout(xt)
        xc = torch.cat((x, xt), 1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        mas1_straight = mas1_straight.mean(dim=1, keepdim=True)
        mas1_straight = self.masif_straight_1(mas1_straight)
        mas1_straight = F.relu(mas1_straight)
        mas1_flipped = mas1_flipped.mean(dim=1, keepdim=True)
        mas1_flipped = self.masif_flipped_1(mas1_flipped)
        mas1_flipped = F.relu(mas1_flipped)
        mas1 = torch.cat((mas1_straight, mas1_flipped), dim=1)
        mas1 = self.masif_pool_1(mas1)
        mas1 = mas1.mean(dim=1)
        mas1 = self.masif_fc1(mas1)
        mas2_straight = mas2_straight.mean(dim=1, keepdim=True)
        mas2_straight = self.masif_straight_2(mas2_straight)
        mas2_straight = F.relu(mas2_straight)
        mas2_flipped = mas2_flipped.mean(dim=1, keepdim=True)
        mas2_flipped = self.masif_flipped_2(mas2_flipped)
        mas2_flipped = F.relu(mas2_flipped)
        mas2 = torch.cat((mas2_straight, mas2_flipped), dim=1)
        mas2 = self.masif_pool_2(mas2)
        mas2 = mas2.mean(dim=1)
        mas2 = self.masif_fc2(mas2)
        xc = torch.cat((xc, mas1, mas2), 1)
        out = self.out(xc)
        out = self.sigmoid(out)
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

