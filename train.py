import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
import pathlib
import math
import sklearn
import torch_optimizer as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from metrics import *
import random
import argparse

# Add argument parsing
parser = argparse.ArgumentParser(description='Train PPI models with different configurations')
parser.add_argument('--run', type=str, required=True, 
                   choices=['mutual_01', 'baseline', 'pool', 'mutual_02', 'mutual_03', 
                           'desc_01', 'desc_02', 'desc_03', 'geom_01', 'geom_02', 'geom_03'],
                   help='Run name to select model configuration')
args = parser.parse_args()

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

import torch.nn as nn
import networkx as nx
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from data_prepare import trainloader, testloader
from models import GCNN, AttGNN, GCNN_mutual_attention, GCNN_with_descriptors, GCNN_desc_pool, GCNN_geom_transformer
from torch_geometric.data import DataLoader as DataLoader_n

print("Datalength")
# print(len(dataset))
print(len(trainloader))
print(len(testloader))

# Model configuration based on run name
def get_model_config(run_name):
    if run_name == 'mutual_01':
        model = GCNN_mutual_attention(num_layers=1, dropout=0.1)
        checkpoint_path = "../masif_features/GCN_01.pth"
        save_path = "../masif_features/GCN_mutual_01_new.pth"
        load_checkpoint = True
        # optimizer =  torch.optim.Adam(model.parameters(), lr= 0.001)
    elif run_name == 'baseline':
        model = GCNN()
        checkpoint_path = "../masif_features/GCN_baseline.pth"
        save_path = "../masif_features/GCN_baseline_new.pth"
        load_checkpoint = True
        # optimizer =  torch.optim.Adam(model.parameters(), lr= 0.001)
    elif run_name == 'pool':
        model = GCNN_desc_pool()
        checkpoint_path = "../masif_features/GCN_pool.pth"
        save_path = "../masif_features/GCN_pool_new.pth"
        load_checkpoint = True
        # optimizer =  torch.optim.Adam(model.parameters(), lr= 0.001)
    elif run_name == 'mutual_02':
        model = GCNN_mutual_attention(num_layers=1, dropout=0.2)
        checkpoint_path = "../masif_features/GCN_02.pth"
        save_path = "../masif_features/GCN_mutual_02_new.pth"
        load_checkpoint = True
        # optimizer =  torch.optim.Adam(model.parameters(), lr= 0.001)
    elif run_name == 'mutual_03':
        model = GCNN_mutual_attention(num_layers=1, dropout=0.3)
        checkpoint_path = "../masif_features/GCN.pth"
        save_path = "../masif_features/GCN_mutual_03_new.pth"
        load_checkpoint = True
        # optimizer =  torch.optim.Adam(model.parameters(), lr= 0.001)
    elif run_name == 'desc_01':
        model = GCNN_with_descriptors(num_layers=1, dropout=0.1, num_features_pro=1024, output_dim=128, descriptor_dim=80, transformer_dim=31, nhead=4, dim_feedforward=128)
        checkpoint_path = "../masif_features/GCN_01_double_transforer.pth"
        save_path = "../masif_features/GCN_desc_01_new.pth"
        load_checkpoint = True
        # optimizer =  torch.optim.Adam(model.parameters(), lr= 0.001)
    elif run_name == 'desc_02':
        model = GCNN_with_descriptors(num_layers=1, dropout=0.2, num_features_pro=1024, output_dim=128, descriptor_dim=80, transformer_dim=31, nhead=4, dim_feedforward=128)
        checkpoint_path = "../masif_features/GCN_02_double_transforer.pth"
        save_path = "../masif_features/GCN_desc_02_new.pth"
        load_checkpoint = True
        # optimizer =  torch.optim.Adam(model.parameters(), lr= 0.001)
    elif run_name == 'desc_03':
        model = GCNN_with_descriptors(num_layers=1, dropout=0.3, num_features_pro=1024, output_dim=128, descriptor_dim=80, transformer_dim=31, nhead=4, dim_feedforward=128)
        checkpoint_path = "../masif_features/GCN_03_double_transforer.pth"
        save_path = "../masif_features/GCN_desc_03_new.pth"
        load_checkpoint = True
        # optimizer =  torch.optim.Adam(model.parameters(), lr= 0.001)
    elif run_name == 'geom_01':
        model = GCNN_geom_transformer(num_layers=2, dropout=0.1, num_features_pro=1024, output_dim=128, descriptor_dim=80, transformer_dim=128, nhead=8, dim_feedforward=256)
        checkpoint_path = None
        save_path = "../masif_features/GCN_geom_01.pth"
        load_checkpoint = False
        # optimizer =  torch.optim.Adam(model.parameters(), lr= 0.0001)
    elif run_name == 'geom_02':
        model = GCNN_geom_transformer(num_layers=2, dropout=0.2, num_features_pro=1024, output_dim=128, descriptor_dim=80, transformer_dim=128, nhead=8, dim_feedforward=256)
        checkpoint_path = None
        save_path = "../masif_features/GCN_geom_02.pth"
        load_checkpoint = False
        # optimizer =  torch.optim.Adam(model.parameters(), lr= 0.0001)
    elif run_name == 'geom_03':
        model = GCNN_geom_transformer(num_layers=2, dropout=0.3, num_features_pro=1024, output_dim=128, descriptor_dim=80, transformer_dim=128, nhead=8, dim_feedforward=256)
        checkpoint_path = None
        save_path = "../masif_features/GCN_geom_03.pth"
        load_checkpoint = False
        # optimizer =  torch.optim.Adam(model.parameters(), lr= 0.0001)
    optimizer =  torch.optim.Adam(model.parameters(), lr= 0.0001)

    return model, checkpoint_path, save_path, load_checkpoint, optimizer

# Get model configuration
model, checkpoint_path, save_path, load_checkpoint, optimizer = get_model_config(args.run)

# Load checkpoint if specified
if load_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))
    print("Checkpoint loaded successfully")
elif load_checkpoint:
    print(f"Warning: Checkpoint {checkpoint_path} not found, starting from scratch")

print(f"Running configuration: {args.run}")
print(f"Model: {type(model).__name__}")
print(f"Save path: {save_path}")

# total_samples = len(dataset)
# n_iterations = math.ceil(total_samples/5)

 
#utilities
def train(model, device, trainloader, optimizer, epoch, num_epochs):

  print(f'Training on {len(trainloader)} samples.....')
  model.train()
  loss_func = nn.BCEWithLogitsLoss()
  predictions_tr = torch.Tensor()
  scheduler = MultiStepLR(optimizer, milestones=[1,5], gamma=0.5)
  labels_tr = torch.Tensor()
  total_loss = 0
  total_count = 0
  loop = tqdm(trainloader, total=len(trainloader), desc=f'Epoch {epoch}/{num_epochs}')
  for count,(prot_1, prot_2, label, mas1_straight, mas1_flipped, mas2_straight, mas2_flipped, mas1_straight_lengths, mas1_flipped_lengths, mas2_straight_lengths, mas2_flipped_lengths) in enumerate(loop):
    prot_1 = prot_1.to(device)
    prot_2 = prot_2.to(device)
    mas1_straight = mas1_straight.to(device)
    mas1_flipped = mas1_flipped.to(device)
    mas2_straight = mas2_straight.to(device)
    mas2_flipped = mas2_flipped.to(device)
    optimizer.zero_grad()
    output = model(prot_1, prot_2, mas1_straight, mas1_flipped, mas2_straight, mas2_flipped, mas1_straight_lengths, mas1_flipped_lengths, mas2_straight_lengths, mas2_flipped_lengths)
    predictions_tr = torch.cat((predictions_tr, output.cpu()), 0)
    labels_tr = torch.cat((labels_tr, label.view(-1,1).cpu()), 0)
    loss = loss_func(output, label.view(-1,1).float().to(device))
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    total_count += len(prot_1.x) if hasattr(prot_1, 'x') else 1
  scheduler.step()
  labels_tr = labels_tr.detach().numpy()
  predictions_tr = torch.sigmoid(torch.tensor(predictions_tr)).numpy()
  acc_tr = get_accuracy(labels_tr, predictions_tr, 0.5)
  print(f'Epoch [{epoch}/{num_epochs}] [==============================] - train_loss : {total_loss / total_count} - train_accuracy : {acc_tr}')
    
 

def predict(model, device, loader):
  model.eval()
  predictions = torch.Tensor()
  labels = torch.Tensor()
  with torch.no_grad():
    loop = tqdm(loader, total=len(loader), desc=f'Epoch {epoch}/{num_epochs}')
    for prot_1, prot_2, label, mas1_straight, mas1_flipped, mas2_straight, mas2_flipped, mas1_straight_lengths, mas1_flipped_lengths, mas2_straight_lengths, mas2_flipped_lengths in loop:
      prot_1 = prot_1.to(device)
      prot_2 = prot_2.to(device)
      mas1_straight = mas1_straight.to(device)
      mas1_flipped = mas1_flipped.to(device)
      mas2_straight = mas2_straight.to(device)
      mas2_flipped = mas2_flipped.to(device)
      output = model(prot_1, prot_2, mas1_straight, mas1_flipped, mas2_straight, mas2_flipped, mas1_straight_lengths, mas1_flipped_lengths, mas2_straight_lengths, mas2_flipped_lengths)
      predictions = torch.cat((predictions, output.cpu()), 0)
      labels = torch.cat((labels, label.view(-1,1).cpu()), 0)
  labels = labels.numpy()
  predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
  return labels.flatten(), predictions.flatten()


# training 

#early stopping
n_epochs_stop = 3
epochs_no_improve = 0
early_stop = False

model.to(device)
num_epochs = 50
loss_func = nn.BCEWithLogitsLoss()
min_loss = 100
best_accuracy = 0
for epoch in range(num_epochs):
  train(model, device, trainloader, optimizer, epoch+1, num_epochs)
  G, P = predict(model, device, testloader)
  loss = get_mse(G,P)
  accuracy = get_accuracy(G,P, 0.5)
  print(f'Epoch [{epoch+1}/{num_epochs}] [==============================] - val_loss : {loss} - val_accuracy : {accuracy}')
  if(accuracy > best_accuracy):
    best_accuracy = accuracy
    best_acc_epoch = epoch
    torch.save(model.state_dict(), save_path) #path to save the model
    print("Model saved to", save_path)
  if(loss< min_loss):
    epochs_no_improve = 0
    min_loss = loss
    min_loss_epoch = epoch
  elif loss> min_loss :
    epochs_no_improve += 1
  if epoch > 3 and epochs_no_improve == n_epochs_stop:
    print('Early stopping!' )
    early_stop = True
    break

print(f'min_val_loss : {min_loss} for epoch {min_loss_epoch} ............... best_val_accuracy : {best_accuracy} for epoch {best_acc_epoch}')
print("Model saved to", save_path)

