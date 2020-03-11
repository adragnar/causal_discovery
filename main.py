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

from utils import powerset
import data_processing as dp

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
            logger_fname='rando.txt', e_stop=True, rawres_fname='rando2.txt', \
            d_size=-1, bin_env=False, testing=False):

    '''

    :param d_fname:
    :param s_fname:
    :param f_fname:
    :param env_atts:
    :param alpha:
    :param feateng_type: The particular preprocess methodology
    :param logger: filepath to log file
    '''
    #Meta-function Accounting
    logging.basicConfig(filename=logger_fname, level=logging.DEBUG)


    accepted_subsets = []
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


    env_atts = [d_atts[cat] for cat in env_atts]  #Note - assuming only split on categorical vars
    logging.info('{} environment attributes'.format(len(env_atts)))
    logging.debug('Environment attributes are ' + str(env_atts))
    #coefficients = torch.zeros(data.shape[1])  #regression vector confidence intervals
    max_pval = 0

    # Setup rawres
    full_res = {}

    with open(rawres_fname, mode='w+') as rawres:
        #Now start the loop
        for i, subset in enumerate(tqdm(powerset(d_atts), desc='pcp_sets',
                           total=len(list(powerset(d_atts))))):  #powerset of PCPs

            #Setup raw result logging
            full_res[str(subset)] = {}

            #Check for empty set
            if not subset:
                continue

            #Check if 2 ME subsets have been accepted
            if e_stop and (len(accepted_subsets) > 0) and \
                    (set.intersection(*accepted_subsets) == set()):
                logging.info('Null Hyp accepted from MECE subsets')
                break

            #Linear regression on all data
            regressors = [d_atts[cat] for cat in subset]
            regressors = [item for sublist in regressors for item in sublist if '_DUMmY' not in item]
            x_s = data[list(itertools.chain(regressors))]
            reg = LinearRegression(fit_intercept=False).fit(x_s.values, y_all.values)

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
                e_out = ~e_in

                if (e_in.sum() < 10) or (e_out.sum() < 10) :  #No data from environment
                    full_res[str(subset)][str(env)] = 'EnvNA'
                    continue

                res_in = (
                y_all.loc[e_in].values - reg.predict(x_s.loc[e_in].values)).ravel()
                res_out = (y_all.loc[e_out].values - reg.predict(
                    x_s.loc[e_out].values)).ravel()

                #Check for NaNs
                if (mean_var_test(res_in, res_out) is np.nan) or (mean_var_test(res_in, res_out) != mean_var_test(res_in, res_out)):
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


            if full_res[str(subset)]['Final_tstat'] > alpha:
                accepted_subsets.append(set(subset))
                logging.info('Subset Accepted')

            ########DEBUG HACK
            # if i == 100:
            #     json.dump(full_res, open('testy.json', 'w'), indent=4, separators=(',',':'))
            #     quit()
            ################


        #STEP 2
        if len(accepted_subsets):
            accepted_features = list(set.intersection(*accepted_subsets))
        else:
            accepted_features = []

        #Save results

        #First the data results
        pickle.dump(accepted_subsets, open(s_fname,'wb'))
        pickle.dump(accepted_features, open(f_fname,'wb'))

        #Next the Raw results
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
    parser.add_argument("rawres_fname", type=str, default=None,
                        help="filename saving raw results")
    parser.add_argument("log_fname", type=str, default=None,
                        help="filename saving log")
    parser.add_argument('env_atts', nargs='+',  \
                        help='atts categorical defining envs')

    parser.add_argument("-early_stopping", type=int, required=True)
    parser.add_argument("-reduce_dsize", type=int, default=-1)
    parser.add_argument("-binarize", type=int, required=True)
    parser.add_argument("--testing", action='store_true')
    args = parser.parse_args()

    if args.testing:
        print("alpha:", args.alpha)
        print("feat_eng:", args.feat_eng)
        print("data:", args.data_fname)
        print("subsets:", args.subsets_fname)
        print("feats:", args.features_fname)
        print("rawres:", args.rawres_fname)
        print("log:", args.log_fname)
        print("env_list:", args.env_atts)
        print("early_stopping?:", args.early_stopping)
        print("d_size:", args.reduce_dsize)
        print("binarize?:", args.binarize)
        print("testing?:", args.testing)
        #quit()

    default(args.data_fname, args.subsets_fname, args.features_fname,  \
            args.env_atts, alpha=args.alpha, feateng_type=[int(c) for c in args.feat_eng], \
            logger_fname=args.log_fname, rawres_fname=args.rawres_fname, \
            e_stop=bool(args.early_stopping), d_size=args.reduce_dsize, \
            bin_env=bool(args.binarize), testing=args.testing)






    # default('data/adult.csv',0,0, \
    #         ["race", "workclass"], alpha=0.05, feateng_type=[1,2], testing=False)
