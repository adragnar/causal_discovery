import argparse
import csv
import pickle
import itertools
import json
import logging
from sklearn.linear_model import LinearRegression
import warnings
import pandas as pd
from tqdm import tqdm
from itertools import combinations

from utils import powerset
import data_processing as dp

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

#########################################
import numpy as np
from scipy.stats import f as fdist
from scipy.stats import ttest_ind

import random

def get_data_regressors(atts, sub, ft_eng, data):
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

def mean_var_test(x, y):
    pvalue_mean = ttest_ind(x, y, equal_var=False).pvalue
    pvalue_var1 = 1 - fdist.cdf(np.var(x, ddof=1) / np.var(y, ddof=1),
                                x.shape[0] - 1,
                                y.shape[0] - 1)

    pvalue_var2 = 2 * min(pvalue_var1, 1 - pvalue_var1)

    return 2 * min(pvalue_mean, pvalue_var2)
#########################################
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

def equalize_strats(store, threshold, dlen, seed):
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



#########################################
def default(dataset_fname, env_atts_types, feateng_type=[], \
            logger_fname='rando.txt', rawres_fname='rando2.txt', \
            d_size=-1, bin_env=False, eq_estrat=-1, SEED=100,
            toy_data=[False], testing=False):

    '''

    :param dataset_fname:
    :param env_atts:
    :param feateng_type: The particular preprocess methodology
    :param logger: filepath to log file
    '''

    random.seed(SEED)

    #Meta-function Accounting
    logging.basicConfig(filename=logger_fname, level=logging.DEBUG)


     #Select correct dataset
    data, y_all, d_atts = dp.data_loader(dataset_fname, feateng_type, dsize=d_size, \
                                    bin=bin_env, toy=toy_data, testing=testing)
    logging.info('{} Dataset loaded - size {}'.format(dataset_fname.split('/')[-1], \
                str(data.shape)))

    #Set allowable datts as PCPs
    allowed_datts = {cat:d_atts[cat] for cat in d_atts.keys() if cat not in env_atts_types}

    #Generate Environments     (assuming only cat vars)
    e_ins_store = get_environments(data, \
                                  {cat:d_atts[cat] for cat in env_atts_types})
    logging.info('{} environment attributes'.format(len(e_ins_store)))
    logging.debug('Environment attributes are '.format(\
                                        [str(e) for e in e_ins_store.keys()]))

    #Normalize operation on e_ins
    if eq_estrat != -1:
        assert eq_estrat > 0
        equalize_strats(e_ins_store, eq_estrat, data.shape[0], SEED)


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
            regressors = get_data_regressors(allowed_datts, subset, feateng_type, data)
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
                if (mean_var_test(res_in, res_out) is np.nan) or \
                (mean_var_test(res_in, res_out) != mean_var_test(res_in, res_out)):
                    full_res[str(subset)][str(env)] = 'NaN'
                else:
                    full_res[str(subset)][str(env)] = mean_var_test(res_in,
                                                                    res_out)

            # # TODO: Jonas uses "min(p_values) * len(environments) - 1"
            full_res[str(subset)]['Final_tstat'] = min([p for p in full_res[str(subset)].values() if type(p) != str]) * len(e_ins_store.keys())

        logging.info('Enumerated all steps')

        #Save results
        json.dump(full_res, rawres, indent=4, separators=(',',':'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument("feat_eng", type=str, \
                        help="each digit id of diff feat engineering")
    parser.add_argument("data_fname", type=str,
                        help="filename adult.csv")
    parser.add_argument("rawres_fname", type=str, default=None,
                        help="filename saving raw results")
    parser.add_argument("log_fname", type=str, default=None,
                        help="filename saving log")
    parser.add_argument('env_atts', nargs='+',  \
                        help='atts categorical defining envs')

    parser.add_argument("-reduce_dsize", type=int, default=-1)
    parser.add_argument("-binarize", type=int, required=True)
    parser.add_argument("-eq_estrat", type=int, default=-1)
    parser.add_argument("-seed", type=int, default=100)
    parser.add_argument("--testing", action='store_true')
    args = parser.parse_args()

    if args.testing:
        print("feat_eng:", args.feat_eng)
        print("data:", args.data_fname)
        print("rawres:", args.rawres_fname)
        print("log:", args.log_fname)
        print("env_list:", args.env_atts)
        print("d_size:", args.reduce_dsize)
        print("binarize?:", args.binarize)
        print("eq_estrat?:", args.eq_estrat)
        print("seed?:", args.seed)
        print("testing?:", args.testing)
        quit()

    default(args.data_fname, args.env_atts, feateng_type=[int(c) for c in args.feat_eng], \
            logger_fname=args.log_fname, rawres_fname=args.rawres_fname, \
             d_size=args.reduce_dsize, \
            bin_env=bool(args.binarize),  \
            eq_estrat=args.eq_estrat, SEED=args.seed, testing=args.testing)
