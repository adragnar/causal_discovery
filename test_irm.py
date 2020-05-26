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
import pandas as pd
from tqdm import tqdm
from itertools import combinations
from torch.autograd import grad
from torch import nn

import data_processing as dp

def get_environments(df, e):
    '''Compute values of df satisfying each environment in e

    :param df: Pandas df of dataset without labels
    :param e: Dictionary of {base_cat:[all assoc df columns]} for all speicfied
              environments. Excludes columns of transformed features
    :return store: Dict of {env:e_in_values}
    '''

    store = {}
    for env in itertools.product(*[e[cat] for cat in e]):
        #Get the stratification columns associated with env
        dummy_atts = []
        live_atts = []
        for att in env:
            if '_DUMmY' in att:
                dummy_atts = [a for a in e[att.split('_')[0]] if '_DUMmY' not in a]
            else:
                live_atts.append(att)

        #Compute e_in
        if not dummy_atts:
            e_in = ((df[live_atts] == 1)).all(1)
        elif not live_atts:
            e_in = ((df[dummy_atts] == 0)).all(1)
        else:
            e_in = ((df[live_atts] == 1).all(1) & (df[dummy_atts] == 0).all(1))
        store[env] = e_in
    return store

class MLP(nn.Module):
    def __init__(self, isize, osize, hidden_dim):
        super(MLP, self).__init__()

        lin1 = nn.Linear(isize, hidden_dim)
        lin2 = nn.Linear(hidden_dim, hidden_dim)
        lin3 = nn.Linear(hidden_dim, isize)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)

    def forward(self, input):
        out = input
        out = self._main(out)
        return out

def train(data, y_all, environments, args, reg=0):
    dim_x = data.shape[1]

    phi = MLP(dim_x, dim_x, 100)  #torch.nn.Parameter(torch.ones(dim_x, dim_x))
    phi = phi.train()
    w = torch.ones(dim_x)  #torch.ones(dim_x, 1)
    w.requires_grad = True

    opt = torch.optim.Adam(phi.parameters(), lr=args["lr"])
    loss = torch.nn.MSELoss()
    logging.info('Using Adam optimizer, LR = {}'.format(args["lr"]))
    logging.info('Loss function MSE')
    print([p.requires_grad for p in phi.parameters()])

    for iteration in range(args["n_iterations"]):
        opt.zero_grad()
        penalty = 0
        error = 0
        for e, e_in in environments.items():
            error_e = loss(torch.from_numpy(data.loc[e_in].values.squeeze()).float() \
                           @ phi(w), \
                           torch.from_numpy(y_all.loc[e_in].values.squeeze()).float())
            penalty += grad(error_e, w,
                            create_graph=True)[0].pow(2).mean()
            error += error_e
        total = (reg * error + (1 - reg) * penalty)

        total.backward()
        opt.step()

        if args["verbose"] and iteration % 250 == 0:
            # w_str = pretty(solution())
            print("{:05d} | {:.5f} | {:.5f} | {:.5f}".format(iteration,
                                                                  reg,
                                                                  error,
                                                                  penalty))
            # print(phi)
            # print(phi.grad)
            # print(torch.from_numpy(y_all.loc[e_in].values).float().data)
            # print((torch.from_numpy(data.loc[e_in].values).float() \
            #                @ phi @ w).data)
            # print((phi).grad_fn)
            # print((w).grad_fn)
            # z = torch.ones(45, 1, requires_grad=True)
            # a = torch.from_numpy(y_all.loc[e_in].values).float()
            # b = torch.ones(6435, 45, requires_grad=False)@ phi @ z                #torch.ones(torch.ones(38651, 45), 1, requires_grad=True)
            # print((phi).shape)
            # print(a.shape)
            # print(b.shape)
            # c = loss(a, b)
            # opt.zero_grad()
            # c.backward()
            # print(b.grad)
            # print(phi.grad)
            # assert False

if __name__ == '__main__':
    args = {'lr':0.000001, \
                 'n_iterations':5000, \
                 'verbose':True}
    data, yall, d_atts = dp.adult_dataset_processing('data/adult.csv', [1], reduce_dsize=-1, bin=True, seed=1000, testing=False)
    e_ins_store = get_environments(data, {'workclass':d_atts['workclass']})

    train(data, yall, e_ins_store, args)
