import pickle
import itertools
from sklearn.linear_model import LinearRegression
import pandas as pd

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

alpha = 0.05
accepted_subsets = []



data, d_atts = adult_dataset_processing()

env_atts = [d_atts[cat] for cat in ['sex']]  #Note - assuming only split on categorical vars

for subset in powerset(d_atts):  #enumerate over powerset of PCPs
    if not subset:
        continue
    #Linear regression on all data
    y_all = data['income_>50K']
    x_s = data[list(itertools.chain(*[d_atts[cat] for cat in subset]))]
    reg = LinearRegression(fit_intercept=False).fit(x_s.values, y_all.values)

    p_values = []
    for e_type in env_atts:

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
    p_value = min(p_values) * sum(len(e_type) for e_type in env_atts)

    if p_value > alpha:
        accepted_subsets.append(set(subset))
        # if args["verbose"]:
        #     print("Accepted subset:", subset)

pickle.dump(accepted_subsets, open('lol.txt','wb'))

print(accepted_subsets)

#
# #STEP 2
#
# if len(accepted_subsets):
#     accepted_features = list(set.intersection(*accepted_subsets))
#     # if args["verbose"]:
#     #     print("Intersection:", accepted_features)
#     # self.coefficients = np.zeros(dim)