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

def alpha_2_range(alpha):
    ''' Convert encoded alpha values into range to test
    :param: alpha of form 'start,end,step' or '(list of alphas)'
    '''
    if ('range' in alpha):
        alpha = [float(a) for a in alpha.split('-')[1:]]
        a_list = []
        for i in np.linspace(alpha[0], alpha[1], ((alpha[1] - alpha[0])/alpha[2])):
            a_list.append(float(i))
    else:
        a_list = [float(a) for a in alpha.split('-')]
    return a_list


def mean_var_test(x, y):
    pvalue_mean = ttest_ind(x, y, equal_var=False).pvalue
    pvalue_var1 = 1 - fdist.cdf(np.var(x, ddof=1) / np.var(y, ddof=1),
                                x.shape[0] - 1,
                                y.shape[0] - 1)

    pvalue_var2 = 2 * min(pvalue_var1, 1 - pvalue_var1)

    return 2 * min(pvalue_mean, pvalue_var2)
#########################################
def default(d_fname, env_atts_types, alpha='(0.05)', feateng_type=[], \
            logger_fname='rando.txt', e_stop=True, rawres_fname='rando2.txt', \
            d_size=-1, bin_env=False, takeout_envs=False, eq_estrat=-1, SEED=100,
            testing=False):

    '''

    :param d_fname:
    :param env_atts:
    :param alpha:
    :param feateng_type: The particular preprocess methodology
    :param logger: filepath to log file
    '''
    random.seed(SEED)

    #Meta-function Accounting
    logging.basicConfig(filename=logger_fname, level=logging.DEBUG)

    a_list = alpha_2_range(alpha)
    accepted_subsets = {a:[] for a in a_list}

     #Select correct dataset
    if 'adult' in d_fname:
        data, y_all, d_atts = dp.adult_dataset_processing(d_fname, \
                              feateng_type, reduce_dsize=d_size, \
                              estrat_red=args.binarize, \
                              testing=testing)
        logging.info('Adult Dataset loaded - size ' + str(data.shape))
    elif 'german' in d_fname:
        data, y_all, d_atts = dp.german_credit_dataset_processing(d_fname, \
                              feateng_type, estrat_red=args.binarize, \
                              testing=testing)
        logging.info('German Dataset loaded - size ' + str(data.shape))


    env_atts = [d_atts[cat] for cat in env_atts_types]  #Note - assuming only split on categorical vars
    #Set whether we iterate through env_atts as PCPs
    if takeout_envs:
        allowed_datts = {cat:d_atts[cat] for cat in d_atts.keys() if cat not in env_atts_types}
    else:
        allowed_datts = d_atts

    #Clean data to make sure no cases where some environments are too low


    logging.info('{} environment attributes'.format(len(env_atts)))
    logging.debug('Environment attributes are ' + str(env_atts))
    logging.debug('Alphas tested are ' + str(a_list))
    #coefficients = torch.zeros(data.shape[1])  #regression vector confidence intervals
    max_pval = 0
    # Setup rawres
    full_res = {}


    #First, figure out the available individuals in each environment strat
    #Compute & store the e_in for each environment
    e_ins_store = {}
    for env in itertools.product(*env_atts):
        dummy_envs = []
        live_envs = []
        for att in env:
            if '_DUMmY' in att:
                dummy_envs = [d for d in d_atts[att.split('_')[0]] if d != att]
            else:
                live_envs.append(att)

        #Compute e_in without error
        if not dummy_envs:
            e_in = ((data[live_envs] == 1)).all(1)
        elif not live_envs:
            e_in = ((data[dummy_envs] == 0)).all(1)
        else:
            e_in = ((data[live_envs] == 1).all(1) & (data[dummy_envs] == 0).all(1))
        e_ins_store[str(env)] = e_in

    #Normalize operation on e_ins
    if eq_estrat != -1:
        assert eq_estrat > 0
        sizes = []
        for env in e_ins_store:
            sizes.append(e_ins_store[env].sum())

        if (min(sizes) < eq_estrat) or \
               (max(sizes) > (data.shape[0] - eq_estrat)) : #Check if normalization broken
            logging.error('Environment Stratification Below Threshold')
            for env, e_in in e_ins_store:
                logging.error('{} : {}'.format(env, e_ins_store[env].sum()))
            assert True == False

        for env in e_ins_store: #Now normalize with min samples
            raw = e_ins_store[env].to_frame(name='vals')
            chosen_cols = raw[raw['vals'] == True].sample(min(sizes), random_state=SEED)
            raw.loc[:,:] = False
            raw.update(chosen_cols)
            e_ins_store[env] = raw.squeeze()

    #Now start enumerating PCPs

    with open(rawres_fname, mode='w+') as rawres:
        for i, subset in enumerate(tqdm(powerset(allowed_datts.keys()), desc='pcp_sets',
                           total=len(list(powerset(allowed_datts.keys()))))):  #powerset of PCPs

            #Setup raw result logging
            full_res[str(subset)] = {}

            #Check for empty set
            if not subset:
                continue

            #Check if 2 ME subsets have been accepted
            if e_stop:
                if not a_list:  #Check if any remaining alphas
                    break
                else:  #See if any ME subsets accepted
                    del_list = []
                    for a in a_list:
                        if (len(accepted_subsets[a]) > 0) and \
                                (set.intersection(*(accepted_subsets[a])) == set()):
                            logging.info('Null Hyp accepted from MECE subsets for alpha={}'.format(a))
                            del_list.append(a)
                    for a in del_list:
                        a_list.remove(a)


            #Linear regression on all data
            regressors = get_data_regressors(allowed_datts, subset, feateng_type, data)
            x_s = data[list(itertools.chain(regressors))]
            reg = LinearRegression(fit_intercept=False).fit(x_s.values, y_all.values)

            #Use the normalized e_ins to compute the residuals + Find p_values for every environment
            for env in itertools.product(*env_atts):
                e_in = e_ins_store[str(env)]
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
                    continue
                    # print(env)
                    # pickle.dump(res_in, open('res_in.txt', 'wb'))
                    # pickle.dump(res_out, open('res_out.txt', 'wb'))
                    # quit()
                else:
                    full_res[str(subset)][str(env)] = mean_var_test(res_in,
                                                                    res_out)


            # # TODO: Jonas uses "min(p_values) * len(environments) - 1"
            full_res[str(subset)]['Final_tstat'] = min([p for p in full_res[str(subset)].values() if type(p) != str]) * len(list(itertools.product(*env_atts)))

            any_acc = False
            for a in a_list:
                if full_res[str(subset)]['Final_tstat'] > a:
                    accepted_subsets[a].append(set(subset))
                    logging.info('Subset Accepted for alpha={}'.format(a))
                    any_acc = True
            if any_acc:
                logging.info('Interation_{}'.format(i))
        logging.info('Enumerated all steps')

        #Save results
        json.dump(full_res, rawres, indent=4, separators=(',',':'))


    # if args["verbose"]:
    #     print("Intersection:", accepted_features)
    # coefficients = np.zeros(data.shape[1])
    #
    # if len(accepted_features):
    #     x_s = x_all[:, list(accepted_features)]
    #     reg = LinearRegression(fit_intercept=False).fit(x_s, y_all)
    #     self.coefficients[list(accepted_features)] = reg.coef_
    #
    # self.coefficients = torch.Tensor(self.coefficients)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument("alpha", type=str, \
                        help="significance level for PCP acceptance")
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

    parser.add_argument("-early_stopping", type=int, required=True)
    parser.add_argument("-reduce_dsize", type=int, default=-1)
    parser.add_argument("-binarize", type=int, required=True)
    parser.add_argument("-takeout_envs", type=int, required=True)
    parser.add_argument("-eq_estrat", type=int, default=-1)
    parser.add_argument("-seed", type=int, default=100)
    parser.add_argument("--testing", action='store_true')
    args = parser.parse_args()

    if args.testing:
        print("alpha:", args.alpha)
        print("feat_eng:", args.feat_eng)
        print("data:", args.data_fname)
        print("rawres:", args.rawres_fname)
        print("log:", args.log_fname)
        print("env_list:", args.env_atts)
        print("early_stopping?:", args.early_stopping)
        print("d_size:", args.reduce_dsize)
        print("binarize?:", args.binarize)
        print("takeout_envs?:", args.takeout_envs)
        print("eq_estrat?:", args.eq_estrat)
        print("seed?:", args.seed)
        print("testing?:", args.testing)
        quit()

    default(args.data_fname, args.env_atts, alpha=args.alpha, feateng_type=[int(c) for c in args.feat_eng], \
            logger_fname=args.log_fname, rawres_fname=args.rawres_fname, \
            e_stop=bool(args.early_stopping), d_size=args.reduce_dsize, \
            bin_env=bool(args.binarize), takeout_envs=args.takeout_envs, \
            eq_estrat=args.eq_estrat, SEED=args.seed, testing=args.testing)
