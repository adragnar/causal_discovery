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
import torch.autograd as autograd
from torch import nn

from utils import powerset, dname_from_fpath, make_tensor
import data_processing as dp
import environment_processing as eproc

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

#########################################
import numpy as np
from scipy.stats import f as fdist
from scipy.stats import ttest_ind

import random

class InvarianceBase(object):
    '''Basic methods for all causation as invariance algorithms'''

    def __init__(self):
        pass

    def equalize_strats(self, store, threshold, dlen, seed):
        '''Preprocess all e_ins to have the same number of samples_wanted
        :param store: {env:e_in}, where env is tuple of df cols, e_in is series of
                      True/False vals for each row of the df's inclusion in env
        :param threshold: minimum number of samples in each strat
        :param: length of dataset
        :return None: Modify store e_in values
        '''

        sizes = []
        for env in store:
            sizes.append(store[env].sum())

        if (min(sizes) < threshold) or \
               (max(sizes) > (dlen - threshold)) : #Check if normalization broken
            logging.error('Environment Stratification Below Threshold')
            for env, e_in in store:
                logging.error('{} : {}'.format(env, store[env].sum()))
            assert True == False

        for env in store: #Now normalize with min samples
            raw = store[env].to_frame(name='vals')
            chosen_cols = raw[raw['vals'] == True].sample(min(sizes), random_state=seed)
            raw.loc[:,:] = False
            raw.update(chosen_cols)
            store[env] = raw.squeeze()

class MLP(nn.Module):
    def __init__(self, d, hid_dim):
        super(MLP, self).__init__()
        lin1 = nn.Linear(d, hid_dim)
        lin2 = nn.Linear(hid_dim, hid_dim)
        lin3 = nn.Linear(hid_dim, 1)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)

    def forward(self, input):
        out = self._main(input)
        return out

class InvariantRiskMinimization(InvarianceBase):
    """Object Wrapper around IRM"""

    def __init__(self):
        self.args = {'lr':0.001, \
                     'n_iterations':5000, \
                     'penalty_anneal_iters':100, \
                     'l2_reg':0.001, \
                     'verbose':True}

    def mean_nll(self, logits, y):
        return nn.functional.binary_cross_entropy_with_logits(logits, y)

    def mean_accuracy(self, logits, y):
        preds = (logits > 0.).float()
        return ((preds - y).abs() < 1e-2).float().mean()

    def penalty(self, logits, y):
        scale = torch.tensor(1.).requires_grad_()
        loss = self.mean_nll(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

    def train(self, data, y_all, environments, args, hid_layers=100, reg=0):
        dim_x = data.shape[1]

        self.errors = []
        self.penalties = []
        self.losses = []

        self.phi = MLP(dim_x, hid_layers)
        optimizer = torch.optim.Adam(self.phi.parameters(), lr=args['lr'])

        logging.info('[step, train nll, train acc, train penalty, test acc]')

        #Start the training loop
        for step in tqdm(range(args['n_iterations'])):
            e_comp = {}
            for e, e_in in environments.items():
                e_comp[e] = {}
                # import pdb; pdb.set_trace()
                # d = make_tensor(data.loc[e_in].values)
                logits = self.phi(make_tensor(data.loc[e_in].values))
                labels = make_tensor(y_all.loc[e_in].values)
                e_comp[e]['nll'] = self.mean_nll(logits, labels)
                e_comp[e]['acc'] = self.mean_accuracy(logits, labels)
                e_comp[e]['penalty'] = self.penalty(logits, labels)

            train_nll = torch.stack([e_comp[e]['nll'] for e in e_comp.keys()]).mean()
            train_acc = torch.stack([e_comp[e]['acc'] for e in e_comp.keys()]).mean()
            train_penalty = torch.stack([e_comp[e]['penalty'] for e in e_comp.keys()]).mean()
            loss = train_nll.clone()

            #Regularize the weights
            weight_norm = torch.tensor(0.)
            for w in self.phi.parameters():
                weight_norm += w.norm().pow(2)
            loss += args['l2_reg'] * weight_norm

            #Regularize the weights
            weight_norm = torch.tensor(0.)
            for w in self.phi.parameters():
                weight_norm += w.norm().pow(2)
            loss += args['l2_reg'] * weight_norm

            #Add the invariance penalty
            penalty_weight = (reg
                if step >= args['penalty_anneal_iters'] else 1.0)
            loss += penalty_weight * train_penalty
            if penalty_weight > 1.0: # Rescale big loss
                loss /= penalty_weight

            #Do the backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #Printing and Logging
            if step % 1000 == 0:
                logging.info([np.int32(step),
                              train_nll.detach().cpu().numpy(),
                              train_acc.detach().cpu().numpy(),
                              train_penalty.detach().cpu().numpy()]
                             )


            self.errors.append(train_nll.detach().numpy())
            self.penalties.append(train_penalty.detach().numpy())
            self.losses.append(loss.detach().numpy())



    def run(self, data, y_all, d_atts, unid, expdir, seed, env_atts_types, eq_estrat, val=['-1']):
        phi_fname = os.path.join(expdir, 'phi_{}.pt'.format(unid))
        errors_fname = os.path.join(expdir, 'errors_{}.npy'.format(unid))
        penalties_fname = os.path.join(expdir, 'penalties_{}.npy'.format(unid))
        losses_fname = os.path.join(expdir, 'losses_{}.npy'.format(unid))

        #Set allowable datts as PCPs
        allowed_datts = {cat:d_atts[cat] for cat in d_atts.keys() if cat not in env_atts_types}

        #Generate Environments     (assuming only cat vars)
        e_ins_store = eproc.get_environments(data, \
                                {cat:d_atts[cat] for cat in env_atts_types})

        #Remove Val Data
        if val != ['-1']:
            val_ein = e_ins_store.pop(tuple([val[0]]))
            e_ins_store = {e:e_ins_store[e][np.logical_not(val_ein.values)] \
                           for e in e_ins_store.keys()}
            data, y_all = data[np.logical_not(val_ein.values)], y_all[np.logical_not(val_ein.values)]


        logging.info('{} environment attributes'.format(len(e_ins_store)))
        logging.debug('Environment attributes are {}'.format( \
                                            str([str(e) for e in e_ins_store.keys()])))
        logging.info('Validation Parameters are {}'.format(val))

        #Normalize operation on e_ins
        if eq_estrat != -1:
            assert eq_estrat > 0
            self.equalize_strats(e_ins_store, eq_estrat, data.shape[0], seed)


        #Now start with IRM itself
        reg = 10000
        self.train(data, y_all, e_ins_store, self.args, reg=reg)

        #Save Results
        torch.save(self.phi.state_dict(), phi_fname)
        np.save(errors_fname, self.errors)
        np.save(penalties_fname, self.penalties)
        np.save(losses_fname, self.losses)

    def predict(self, data, phi_params):
        '''
        :param data: the dataset (nparray)
        :param phi_params: The state dict of the MLP'''
        phi = MLP(data.shape[1], 100)
        phi.load_state_dict(phi_params)
        return pd.DataFrame(phi(make_tensor(data)).detach().numpy())

class InvariantCausalPrediction(InvarianceBase):
    """Object Wrapper around ICP"""

    def __init__(self):
        pass

    def get_data_regressors(self, atts, sub, ft_eng, data):
        '''From a given subset of attributes being predicted on and the attributes
        dictionary with the original columns, extract all coluns to predict on from
        dataset

        :param atts - dictionary of attributes, {att, [one-hot col list of at vals]}
        :param sub - subset of atts being predicted on
        :param ft_eng - [which mods applicable]
        '''
        orig_regressors = [atts[cat] for cat in sub]
        orig_regressors = [item for sublist in orig_regressors for item in sublist if '_DUMmY' not in item]
        #Now have all the actual one-hot columns in dataset

        if not ft_eng:
            return orig_regressors

        one_regressors = []
        two_regressors = []
        if 1 in ft_eng: #Assumes only single-col vals are squared
            sq_regressors = [col for col in data.columns if '_sq' in col]
            for r in orig_regressors:
                for r_sq in sq_regressors:
                    if r in r_sq:
                        one_regressors.append(r_sq)

        if 2 in ft_eng:
            x_regressors = [col for col in data.columns if '_x_' in col]
            for r in [com for com in combinations(orig_regressors, 2) \
                if (com[0].split('_')[0] != com[1].split('_')[0])]:

                for x_reg in x_regressors:
                    if ((r[0] in x_reg) and (r[1] in x_reg)):
                        two_regressors.append(x_reg)

        return orig_regressors + one_regressors + two_regressors

    def mean_var_test(self, x, y):
        pvalue_mean = ttest_ind(x, y, equal_var=False).pvalue
        pvalue_var1 = 1 - fdist.cdf(np.var(x, ddof=1) / np.var(y, ddof=1),
                                    x.shape[0] - 1,
                                    y.shape[0] - 1)

        pvalue_var2 = 2 * min(pvalue_var1, 1 - pvalue_var1)

        return 2 * min(pvalue_mean, pvalue_var2)

    def get_coeffs(self, causal_ps, data, y_all, env_datts={}, eq_estrat=-1, seed=None):
        '''Generate the average coefficients for certain causal predictor set
        :param causal_ps: List of dset variables to predict on list(str)
        :param data: Dataset (pandas df)
        :param y_all: labels (pandas series)
        :param env_datts: {original_env_name:onehot_env_names}
        :param eq_estrat: -1 if no, min num samples if yes (int)

        '''
        # e_ins_store = self.get_environments(data, env_datts)
        #
        # #Normalize operation on e_ins
        # if eq_estrat != -1:
        #     assert eq_estrat > 0
        #     self.equalize_strats(e_ins_store, eq_estrat, data.shape[0], seed)

        if len(causal_ps) > 0:
            x_s = data[causal_ps]
            reg = LinearRegression(fit_intercept=False).fit(x_s.values, y_all.values).coef_[0]
            n = list(x_s.columns)
        else:
            return pd.DataFrame()

        coeffs = sorted(zip(reg, n), reverse=True, key=lambda x: abs(x[0]))
        coeffs = pd.DataFrame(coeffs, columns=['coeff', 'predictor'])
        return coeffs

    def predict(self, data, coeffs):
        #Order dataframe by coefficients column
        if coeffs.empty:
            return pd.DataFrame()
        assert set(list(coeffs['predictor'].values)).issubset(set(list(data.columns)))
        data = data[list(coeffs['predictor'].values)]  #make sure cols align

        return pd.DataFrame(data.values @ coeffs['coeff'].values)


    def run(self, data, y_all, d_atts, unid, expdir, feateng_type, seed, env_atts_types, eq_estrat, val=['-1']):
        rawres_fname = os.path.join(expdir, 'rawres_{}.json'.format(unid))
        #Set allowable datts as PCPs
        allowed_datts = {cat:d_atts[cat] for cat in d_atts.keys() if cat not in env_atts_types}

        #Generate Environments     (assuming only cat vars)
        e_ins_store = eproc.get_environments(data, \
                                {cat:d_atts[cat] for cat in env_atts_types})

        #Remove Val Data
        if val != ['-1']:
            val_ein = e_ins_store.pop(tuple([val[0]]))
            e_ins_store = {e:e_ins_store[e][np.logical_not(val_ein.values)] \
                           for e in e_ins_store.keys()}
            data, y_all = data[np.logical_not(val_ein.values)], y_all[np.logical_not(val_ein.values)]

        logging.info('{} environment attributes'.format(len(e_ins_store)))
        logging.debug('Environment attributes are {}'.format( \
                                            str([str(e) for e in e_ins_store.keys()])))
        logging.info('Validation Parameters are {}'.format(val))

        #Now start enumerating PCPs
        full_res = {}
        with open(rawres_fname, mode='w+') as rawres:
            for i, subset in enumerate(tqdm(powerset(allowed_datts.keys()), desc='pcp_sets',
                               total=len(list(powerset(allowed_datts.keys()))))):  #powerset of PCPs

                #Setup raw result logging
                full_res[str(subset)] = {}

                #Check for empty set
                if not subset:
                    continue


                #Linear regression on all data
                regressors = self.get_data_regressors(allowed_datts, subset, feateng_type, data)
                x_s = data[list(itertools.chain(regressors))]
                reg = LinearRegression(fit_intercept=False).fit(x_s.values, y_all.values)

                #Use the normalized e_ins to compute the residuals + Find p_values for every environment
                assert len(e_ins_store.keys()) > 1   #For validation performance
                for env in e_ins_store.keys():
                    e_in = e_ins_store[env]
                    e_out = np.logical_not(e_in)

                    if (e_in.sum() < 2) or (e_out.sum() < 2) :  #No data from environment
                        raise Exception('Not enough data in environment to do the computation')

                    res_in = (
                    y_all.loc[e_in].values - reg.predict(\
                              x_s.loc[e_in].values)).ravel()

                    res_out = (y_all.loc[e_out].values - reg.predict(
                        x_s.loc[e_out].values)).ravel()

                    #Check for NaNs
                    if (self.mean_var_test(res_in, res_out) is np.nan) or \
                    (self.mean_var_test(res_in, res_out) != self.mean_var_test(res_in, res_out)):
                        full_res[str(subset)][str(env)] = 'NaN'
                    else:
                        full_res[str(subset)][str(env)] = self.mean_var_test(res_in,
                                                                        res_out)

                # # TODO: Jonas uses "min(p_values) * len(environments) - 1"
                try:
                    full_res[str(subset)]['Final_tstat'] = min([p for p in full_res[str(subset)].values() if type(p) != str]) * len(e_ins_store.keys())
                except:
                    import pdb; pdb.set_trace()
            logging.info('Enumerated all steps')

            #Save results
            json.dump(full_res, rawres, indent=4, separators=(',',':'))

class Linear():

    def __init__(self):
        pass

    def run(self, data, y_all, unid, expdir, seed=1000, val=['-1']):
        reg_fname = os.path.join(expdir, 'regs_{}.pkl'.format(unid))

        #Deal with validation

        if val != ['-1']:
            assert 0.0 < float(val[0]) < 1.0
            data = data.sample(frac=(1-float(val[0])), random_state=seed)
            y_all = y_all.sample(frac=(1-float(val[0])), random_state=seed)

        reg = LinearRegression(fit_intercept=False).fit(data.values, y_all.values).coef_[0]
        coeffs = sorted(zip(reg, data.columns), reverse=True, key=lambda x: abs(x[0]))
        coeffs = pd.DataFrame(coeffs, columns=['coeff', 'predictor'])
        pd.to_pickle(coeffs, reg_fname)

    def predict(self, data, coeffs):
        #Order dataframe by coefficients column
        if coeffs.empty:
            return pd.DataFrame()
        assert set(list(coeffs['predictor'].values)).issubset(set(list(data.columns)))
        data = data[list(coeffs['predictor'].values)]  #make sure cols align

        return pd.DataFrame(data.values @ coeffs['coeff'].values)
