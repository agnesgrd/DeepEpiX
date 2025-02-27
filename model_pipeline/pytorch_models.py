import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pickle
import numpy as np
import os
import os.path as op
import mne
from tqdm import tqdm
from warnings import warn
import string
import sys
import matplotlib.pyplot as plt
import params
import pandas as pd
import csv
from utils import save_obj, load_obj, standardize

def resume(model, filename):
    model.load_state_dict(torch.load(filename))

class SFCN(nn.Module):
    def __init__(self):
        super(SFCN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding="same")
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding="same")
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding="same")
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, padding="same")
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 64, kernel_size=1, padding="same")
        self.bn5 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.swapaxes(x, 2, 3)
        x = F.max_pool2d(F.leaky_relu(self.bn1(self.conv1(x))),2)
        x = F.max_pool2d(F.leaky_relu(self.bn2(self.conv2(x))),2)
        x = F.max_pool2d(F.leaky_relu(self.bn3(self.conv3(x))),2)
        x = F.max_pool2d(F.leaky_relu(self.bn4(self.conv4(x))),2)
        x = F.avg_pool2d(F.leaky_relu(self.bn5(self.conv5(x))),2,padding=(1,0))
        emb = torch.flatten(x,start_dim=1, end_dim=-1)  # flatten
        x = F.dropout(emb, p=0.5)
        x = torch.sigmoid(self.fc1(x))
        return x


def z_score_normalize(data):
    return (data - data.mean()) / data.std()

class TimeSeriesDataSet(Dataset):
  """
  This is a custom dataset class

  """
  def __init__(self, X_ids, dim, path):
    self.X = X_ids
    self.path = path
    self.dim = dim

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    # note that this isn't randomly selecting. It's a simple get a single item that represents an x and y
    win = np.array(self.X[index])[0]
    sub = np.array(self.X[index])[1]

    prefixe = 'data_raw_'
    suffixe = '_windows_bi'
    path_data = self.path 

    # Opens windows_bi binary file
    f = open(path_data+prefixe+str(sub)+suffixe)
    # Set cursor position to 30 (nb time points)*274 (nb channels)*windows_id*4 because data is stored as float64 and dtype.itemsize = 8
    f.seek(self.dim[0]*self.dim[1]*win*4)
    # From cursor location, get data from 1 window
    sample = np.fromfile(f, dtype='float32', count=self.dim[0]*self.dim[1])
    # Reshape to create a 2D array (data from the binary file is just a vector)
    sample = sample.reshape(self.dim[1],self.dim[0])
    # Swap axis to have time point in first dim, nb channels in 2nd dim
    #sample = np.swapaxes(sample,0,1)
    sample = z_score_normalize(sample)
    
    if len(self.dim) == 3:
        sample = np.expand_dims(sample,axis=0)

    _x = sample
    _y = np.array(self.X[index])[2]
    rand_nb = np.random.randint(2)

    return torch.tensor(_x,dtype = torch.float32),torch.tensor(_y,dtype = torch.float32)

def load_generators_memeff(X_test_ids):

    t = TimeSeriesDataSet(X_test_ids.tolist(), params.dim, params.args.path_output)
    testing_generator = DataLoader(t, batch_size=1, shuffle=False)

    return testing_generator

def test_model(model_name, testing_generator, X_test_ids):

    model = SFCN()
    resume(model, model_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval() 

    y_pred = list()

    with torch.no_grad():
        for data in testing_generator:

            inputs, labels = data
            inputs = inputs.to(device)
            inputs = inputs.float()
            outputs = model(inputs)

            labels = labels.to(device)
            labels = labels.float()
            labels = labels.squeeze()

            y_pred_probas = torch.squeeze(outputs)
            y_pred.append((y_pred_probas.cpu().numpy() > 0.5).astype("int32"))

    y_test = X_test_ids[:,2]

    sub = X_test_ids[:,1]
    win = X_test_ids[:,0]

    y_timing_data = load_obj("data_raw_"+str(params.args.subject_number)+'_timing.pkl',params.args.path_output)
    y_block_data = load_obj("data_raw_"+str(params.args.subject_number)+'_blocks.pkl',params.args.path_output)

    with open(params.args.path_output+'subject_'+str(params.args.subject_number)+'_predictions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["subject","block","timing","pred"])

    for ind, i in enumerate(sub):
        y_timing = y_timing_data[win[ind]]
        y_block = y_block_data[win[ind]]

        with open(params.args.path_output+'subject_'+str(params.args.subject_number)+'_predictions.csv', 'a', newline='') as f:
            writer = csv.writer(f)           
            writer.writerow([i,y_block,y_timing,y_pred[ind]])

    del model