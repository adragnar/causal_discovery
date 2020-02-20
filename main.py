import argparse
import csv
import pickle
import itertools
import torch
from sklearn.linear_model import LinearRegression
import warnings
import pandas as pd
from tqdm import tqdm

from utils import powerset
from data_processing import adult_dataset_processing

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

#########################################
import numpy as np
from scipy.stats import f as fdist
from scipy.stats import ttest_ind

def mean_var_test(x, y):
    pvalue_mean = ttest_ind(x, y, equal_var=False).pvalue
    pvalue_var1 = 1 - fdist.cdf(np.var(x, ddof=1) / np.var(y, ddof=1),
                                x.shape[0] - 1,
                                y.shape[0] - 1)

    pvalue_var2 = 2 * min(pvalue_var1, 1 - pvalue_var1)

    return 2 * min(pvalue_mean, pvalue_var2)
#########################################
def default(d_fname, s_fname, f_fname, env_atts=[], alpha=0.05, feateng_type=[], \
            logger=None, testing=False):
    '''
    
    :param d_fname: 
    :param s_fname: 
    :param f_fname: 
    :param env_atts: 
    :param alpha: 
    :param feateng_type: The particular preprocess methodology 
    :param logger: filepath to log file 
    '''
    accepted_subsets = []

    data, y_all, d_atts = adult_dataset_processing(d_fname, feateng_type)
    env_atts = [d_atts[cat] for cat in env_atts]  #Note - assuming only split on categorical vars
    #coefficients = torch.zeros(data.shape[1])  #regression vector confidence intervals
    max_pval = 0

    # Setup logger and write header
    if logger is not None:
        f = open(logger, mode='w')
        logger = csv.writer(f)
        logger.writerow(list(itertools.product(*env_atts)))

    # Define whatever you want in here to make sure that stuff works
    if testing:
        return

    #Now start the loop
    for subset in tqdm(powerset(d_atts), desc='pcp_sets',
                       total=len(list(powerset(d_atts)))):  #powerset of PCPs

        #Check for empty set
        if not subset:
            continue

        #Check if 2 ME subsets have been accepted
        if (len(accepted_subsets) > 0) and \
                (set.intersection(*accepted_subsets) == set()):
            break

        #Linear regression on all data
        regressors = [d_atts[cat] for cat in subset]
        regressors = [item for sublist in regressors for item in sublist if '_DUMmY' not in item]
        x_s = data[list(itertools.chain(regressors))]

        reg = LinearRegression(fit_intercept=False).fit(x_s.values, y_all.values)
        p_values = []

        #Find p_values for every environment
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

            if e_in.isin([True]).all() or e_in.isin([False]).all():  #No data from environment
                p_values.append('NA')
                continue
            e_out = ~e_in

            res_in = (
            y_all.loc[e_in].values - reg.predict(x_s.loc[e_in].values)).ravel()
            res_out = (y_all.loc[e_out].values - reg.predict(
                x_s.loc[e_out].values)).ravel()

            p_values.append(mean_var_test(res_in, res_out))


        # # TODO: Jonas uses "min(p_values) * len(environments) - 1"
        if logger is not None:
            logger.writerow(list(subset) + p_values)
        p_value = min([p for p in p_values if type(p) != str]) * len(list(itertools.product(*env_atts)))

        ###Hack for debugging
        if p_value > max_pval:
            p_value = max_pval
        #####################

        if p_value > alpha:
            accepted_subsets.append(set(subset))

    #STEP 2
    if len(accepted_subsets):
        accepted_features = list(set.intersection(*accepted_subsets))
    else:
        accepted_features = []

    ###Hack for debugging
    accepted_features.append(max_pval)
    #########

    #Save results

    #First the data
    pickle.dump(accepted_subsets, open(s_fname,'wb'))
    pickle.dump(accepted_features, open(f_fname,'wb'))

    if logger is not None:
        f.close()


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
    parser.add_argument("alpha", type=float, \
                        help="significance level for PCP acceptance")
    parser.add_argument("feat_eng", type=str, \
                        help="each digit id of diff feat engineering")
    parser.add_argument("data_fname", type=str,
                        help="filename adult.csv")
    parser.add_argument("subsets_fname", type=str,
                        help="filename saving acc_subsets")
    parser.add_argument("features_fname", type=str,
                        help="filename saving acc_features")
    parser.add_argument("log_fname", type=str, default=None,
                        help="filename saving log")
    parser.add_argument('env_atts', nargs='+',  \
                        help='atts categorical defining envs')


    args = parser.parse_args()
    # print(args.env_atts)
    # print(args.log_fname)
    default(args.data_fname, args.subsets_fname, args.features_fname,  \
            args.env_atts, alpha=args.alpha, feateng_type=[int(c) for c in args.feat_eng], \
            logger=args.log_fname, testing=False)

    # default('data/adult.csv',0,0, \
    #         ["race", "workclass"], alpha=0.05, feateng_type=[1,2], testing=False)