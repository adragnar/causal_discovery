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

from utils import powerset, dname_from_fpath, pretty
import data_processing as dp

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

    def get_environments(self, df, e):
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

class InvariantRiskMinimization(InvarianceBase):
    """Object Wrapper around IRM"""

    def __init__(self):
        self.args = {'lr':0.001, \
                     'n_iterations': 2, \
                     'verbose':False}

    def train(self, data, y_all, environments, args, reg=0):
        dim_x = data.shape[1]

        self.phi = torch.nn.Parameter(torch.eye(dim_x, dim_x))
        self.w = torch.ones(dim_x, 1)
        self.w.requires_grad = True

        opt = torch.optim.Adam([self.phi], lr=self.args["lr"])
        loss = torch.nn.MSELoss()

        for iteration in range(self.args["n_iterations"]):
            penalty = 0
            error = 0
            for e, e_in in environments.items():
                # print(((torch.from_numpy(data.loc[e_in].values)).float()).type(), self.phi.type(), self.w.type())
                # print((torch.from_numpy(data.loc[e_in].values) @ self.phi @ self.w).shape)
                # print((torch.tensor(y_all.loc[e_in].values, dtype=torch.long).shape))
                error_e = loss(torch.from_numpy(data.loc[e_in].values).float() \
                               @ self.phi @ self.w, \
                               torch.from_numpy(y_all.loc[e_in].values).float())
                penalty += grad(error_e, self.w,
                                create_graph=True)[0].pow(2).mean()
                error += error_e

            opt.zero_grad()
            (reg * error + (1 - reg) * penalty).backward()
            opt.step()

            if self.args["verbose"] and iteration % 1000 == 0:
                w_str = pretty(self.solution())
                print("{:05d} | {:.5f} | {:.5f} | {:.5f} | {}".format(iteration,
                                                                      reg,
                                                                      error,
                                                                      penalty,
                                                                      w_str))

    def run(self, data, y_all, d_atts, unid, expdir, seed, env_atts_types, eq_estrat):
        phi_fname = os.path.join(expdir, 'phi_{}.pt'.format(unid))
        w_fname = os.path.join(expdir, 'w_{}.pt'.format(unid))

        #Set allowable datts as PCPs
        allowed_datts = {cat:d_atts[cat] for cat in d_atts.keys() if cat not in env_atts_types}

        #Generate Environments     (assuming only cat vars)
        e_ins_store = self.get_environments(data, \
                                      {cat:d_atts[cat] for cat in env_atts_types})
        logging.info('{} environment attributes'.format(len(e_ins_store)))
        logging.debug('Environment attributes are {}'.format( \
                                            str([str(e) for e in e_ins_store.keys()])))

        #Normalize operation on e_ins
        if eq_estrat != -1:
            assert eq_estrat > 0
            self.equalize_strats(e_ins_store, eq_estrat, data.shape[0], seed)

        #Now start with IRM itself
        reg = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        val_env = random.sample(set(e_ins_store.keys()), 1)[0]
        val_ein = e_ins_store.pop(val_env)
        val_data = torch.from_numpy(data.loc[val_ein].values).float()
        val_labels = torch.from_numpy(y_all.loc[val_ein].values).float()

        best_reg = 0
        best_err = 1e50
        for r in reg:
            self.train(data, y_all, e_ins_store, self.args, reg=r)
            err = (val_data @ self.solution() - val_labels).pow(2).mean().item()

            if self.args["verbose"]:
                print("IRM (reg={:.3f}) has {:.3f} validation error.".format(
                    r, err))

            if err < best_err:
                best_err = err
                best_reg = r
                best_phi = self.phi.clone()
                best_w = self.w.clone()

        self.phi = best_phi
        self.w = best_w
        torch.save(self.phi, phi_fname)
        torch.save(self.w, w_fname)

    def solution(self):
        return self.phi @ self.w


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

    def run(self, data, y_all, d_atts, unid, expdir, feateng_type, seed, env_atts_types, eq_estrat):
        rawres_fname = os.path.join(expdir, 'rawres_{}.json'.format(unid))
        #Set allowable datts as PCPs
        allowed_datts = {cat:d_atts[cat] for cat in d_atts.keys() if cat not in env_atts_types}

        #Generate Environments     (assuming only cat vars)
        e_ins_store = self.get_environments(data, \
                                      {cat:d_atts[cat] for cat in env_atts_types})
        logging.info('{} environment attributes'.format(len(e_ins_store)))
        logging.debug('Environment attributes are {}'.format( \
                                            str([str(e) for e in e_ins_store.keys()])))

        #Normalize operation on e_ins
        if eq_estrat != -1:
            assert eq_estrat > 0
            self.equalize_strats(e_ins_store, eq_estrat, data.shape[0], seed)


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
                for env in e_ins_store.keys():
                    e_in = e_ins_store[env]
                    e_out = np.logical_not(e_in)

                    if (e_in.sum() < 10) or (e_out.sum() < 10) :  #No data from environment
                        full_res[str(subset)][str(env)] = 'EnvNA'
                        continue

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
                full_res[str(subset)]['Final_tstat'] = min([p for p in full_res[str(subset)].values() if type(p) != str]) * len(e_ins_store.keys())

            logging.info('Enumerated all steps')

            #Save results
            json.dump(full_res, rawres, indent=4, separators=(',',':'))
