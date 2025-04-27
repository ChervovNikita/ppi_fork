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

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.cuda("cpu")

import torch.nn as nn
import networkx as nx
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from data_prepare import dataset, trainloader, testloader
from models import GCNN, AttGNN, GCNN_mutual_attention, GCNN_with_descriptors, GCNN_desc_pool
from torch_geometric.data import DataLoader as DataLoader_n

print("Datalength")
print(len(dataset))
print(len(trainloader))
print(len(testloader))



total_samples = len(dataset)
n_iterations = math.ceil(total_samples/5)

 
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
  for count,(prot_1, prot_2, label, mas1_straight, mas1_flipped, mas2_straight, mas2_flipped) in enumerate(loop):
    prot_1 = prot_1.to(device)
    prot_2 = prot_2.to(device)
    mas1_straight = mas1_straight.to(device)
    mas1_flipped = mas1_flipped.to(device)
    mas2_straight = mas2_straight.to(device)
    mas2_flipped = mas2_flipped.to(device)
    optimizer.zero_grad()
    output = model(prot_1, prot_2, mas1_straight, mas1_flipped, mas2_straight, mas2_flipped)
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
    for prot_1, prot_2, label, mas1_straight, mas1_flipped, mas2_straight, mas2_flipped in loop:
      prot_1 = prot_1.to(device)
      prot_2 = prot_2.to(device)
      mas1_straight = mas1_straight.to(device)
      mas1_flipped = mas1_flipped.to(device)
      mas2_straight = mas2_straight.to(device)
      mas2_flipped = mas2_flipped.to(device)
      output = model(prot_1, prot_2, mas1_straight, mas1_flipped, mas2_straight, mas2_flipped)
      predictions = torch.cat((predictions, output.cpu()), 0)
      labels = torch.cat((labels, label.view(-1,1).cpu()), 0)
  labels = labels.numpy()
  predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
  return labels.flatten(), predictions.flatten()


# training 

#early stopping
n_epochs_stop = 6
epochs_no_improve = 0
early_stop = False


# model = GCNN_mutual_attention(num_layers=1) [85.69242540168325, 87.28943338437979, 87.36600306278713, 86.75344563552832, 89.82402448355012] == simple_run
model = GCNN() # [86.52373660030628, 86.67687595712098, 87.67228177641654, 85.6159143075746, 89.28844682478959] == baseline_run
# model = GCNN_with_descriptors(num_features_pro=1024, output_dim=128, dropout=0.2, descriptor_dim=80, transformer_dim=31, nhead=4, num_layers=2, dim_feedforward=128)
# []
# model = GCNN_desc_pool()
model.to(device)
num_epochs = 50
loss_func = nn.BCEWithLogitsLoss()
min_loss = 100
best_accuracy = 0
optimizer =  torch.optim.Adam(model.parameters(), lr= 0.001)
for epoch in range(num_epochs):
  train(model, device, trainloader, optimizer, epoch+1, num_epochs)
  G, P = predict(model, device, testloader)
  loss = get_mse(G,P)
  accuracy = get_accuracy(G,P, 0.5)
  print(f'Epoch [{epoch+1}/{num_epochs}] [==============================] - val_loss : {loss} - val_accuracy : {accuracy}')
  if(accuracy > best_accuracy):
    best_accuracy = accuracy
    best_acc_epoch = epoch
    torch.save(model.state_dict(), "../masif_features/GCN.pth") #path to save the model
    print("Model")
  if(loss< min_loss):
    epochs_no_improve = 0
    min_loss = loss
    min_loss_epoch = epoch
  elif loss> min_loss :
    epochs_no_improve += 1
  if epoch > 5 and epochs_no_improve == n_epochs_stop:
    print('Early stopping!' )
    early_stop = True
    break

print(f'min_val_loss : {min_loss} for epoch {min_loss_epoch} ............... best_val_accuracy : {best_accuracy} for epoch {best_acc_epoch}')
print("Model saved")

