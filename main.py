import argparse
import csv
import pickle
import itertools
import torch
from sklearn.linear_model import LinearRegression
import pandas as pd
from tqdm import tqdm

from utils import powerset
from data_processing import adult_dataset_processing

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
def default(d_fname, s_fname, f_fname, env_atts, alpha=0.05, feateng_type=[]):
    '''
    
    :param d_fname: 
    :param s_fname: 
    :param f_fname: 
    :param env_atts: 
    :param alpha: 
    :param feateng_type: The particular preprocess methodology 
    '''
    accepted_subsets = []

    data, d_atts = adult_dataset_processing(d_fname, feateng_type)
    env_atts = [d_atts[cat] for cat in env_atts]  #Note - assuming only split on categorical vars
    print(data.shape[1])
    #coefficients = torch.zeros(data.shape[1])  #regression vector confidence intervals

    max_pval = 0
    f = open('fulllog.csv', mode='w')
    logger = csv.writer(f)


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
        y_all = data['income_>50K']
        x_s = data[list(itertools.chain(*[d_atts[cat] for cat in subset]))]
        reg = LinearRegression(fit_intercept=False).fit(x_s.values, y_all.values)

        p_values = []
        for e_type in env_atts:#tqdm(env_atts, desc='env_atts', leave=False):

            for e in e_type:  # currently missing the all 0
                e_in = (data[e] == 1)
                e_out = (data[e] == 0)

                res_in = (y_all.loc[e_in].values - reg.predict(x_s.loc[e_in].values)).ravel()
                res_out = (y_all.loc[e_out].values - reg.predict(x_s.loc[e_out].values)).ravel()

                p_values.append(mean_var_test(res_in, res_out))

            # Cover the dummy variable
            for i, e in enumerate(e_type):
                if i == 0:
                    e_in = (data[e] == 0)
                else:
                    e_in = e_in & (data[e] == 0)
            e_out = ~e_in

            res_in = (
            y_all.loc[e_in].values - reg.predict(x_s.loc[e_in].values)).ravel()
            res_out = (
            y_all.loc[e_out].values - reg.predict(x_s.loc[e_out].values)).ravel()

            p_values.append(mean_var_test(res_in, res_out))


        # # TODO: Jonas uses "min(p_values) * len(environments) - 1"
        logger.writerow(list(subset) + p_values)
        p_value = min(p_values) * sum(len(e_type) for e_type in env_atts)

        ###Hack for debugging
        if p_value > max_pval:
            p_value = max_pval
        #####################

        if p_value > alpha:
            accepted_subsets.append(set(subset))
            # if args["verbose"]:
            #     print("Accepted subset:", subset)

    # #STEP 2
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

    #First, add header to logger file
    try:
        h_row = []
        for e_type in env_atts:
            for e in e_type:
                h_row.append(e)
        logger.writerow(h_row)
        f.close()
    except:
        pass



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
    parser.add_argument('env_atts', nargs='+', \
                        help='atts categorical defining envs')
    args = parser.parse_args()
    print(args.env_atts)
    default(args.data_fname, args.subsets_fname, args.features_fname, \
            args.env_atts, alpha=args.alpha, feateng_type=[int(c) for c in args.feat_eng])

    # default('data/adult.csv',0,0, \
    #         ["occupation", "workclass", "native-country", "education", "marital-status"], alpha=0.05, feateng_type=[1,2])