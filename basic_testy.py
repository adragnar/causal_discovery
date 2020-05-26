import torch
import argparse
import csv
import pickle
import itertools
import json
import logging
import os
from sklearn.linear_model import LinearRegression
import torch
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
from torch.autograd import grad
from torch import nn

import data_processing as dp

def pred_binarize(v):
    '''Convert all values to 0 if <0.5, 1 otherwise'''
    def thresh(x):
        if (x >= 0.5): return 1
        else: return 0
    print(v.shape)
    return np.apply_along_axis(thresh, 1, v)

# Data Generation
# np.random.seed(42)
# x = np.random.rand(100, 1)
# y = 1 + 2 * x + .1 * np.random.randn(100, 1)
#
# # Shuffles the indices
# idx = np.arange(100)
# np.random.shuffle(idx)
#
# # Uses first 80 random indices for train
# train_idx = idx[:80]
# # Uses the remaining indices for validation
# val_idx = idx[80:]
#
# # Generates train and validation sets
# x_train, y_train = x[train_idx], y[train_idx]
# x_val, y_val = x[val_idx], y[val_idx]

x_train, y_train, datts = dp.adult_dataset_processing('data/adult.csv', [1000], bin=1, testing=False)
x_train, y_train = x_train.values, y_train.values.squeeze()

device =  'cpu'

# Our data was in Numpy arrays, but we need to transform them into PyTorch's Tensors
# and then we send them to the chosen device
x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)

lr = 1e-10
n_epochs = 1000

torch.manual_seed(42)
# a = torch.randn(45086, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(41, requires_grad=True, dtype=torch.float, device=device)
opt = torch.optim.Adam([b], lr=lr)
loss = torch.nn.MSELoss()

for epoch in range(n_epochs):
    yhat = x_train_tensor @ b
    # print(y_train_tensor.shape)
    # print(yhat.shape)
    error = loss(y_train_tensor, yhat)
    # error = y_train_tensor - yhat
    # loss = (error ** 2).mean()

    error.backward()
    # Let's check the computed gradients...
    # print(a.grad)
    # print(b.grad)
    opt.step()
    # with torch.no_grad():
    #     # a -= lr * a.grad
    #     b -= lr * b.grad

    # PyTorch is "clingy" to its computed gradients, we need to tell it to let it go...
    # a.grad.zero_()
    # b.grad.zero_()
    opt.zero_grad()
    # if epoch % 1 == 0:
    #     print(b)
    # print(n_epochs)
# print(a, b)
print(b)
print((x_train @ b.detach().numpy()).shape)
yhat = pred_binarize(np.expand_dims(x_train @ b.detach().numpy(), axis=1))
print(1 - np.abs(y_train-yhat).sum()/45061)
