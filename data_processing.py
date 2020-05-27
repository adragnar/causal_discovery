import argparse
import pickle
from time import time

from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from utils import dname_from_fpath

# from model import MultiLayerPerceptron
# from dataset import AdultDataset
# from util import *

#import matplotlib.pyplot as plt

def data_loader(fname, fteng, dsize=-1, bin=0, toy=[False], seed=1000, testing=0):
    '''From dataset name, optional flags, return dataset, labels,
    and column names

    :param fname - path to dataset (csv)
    :param fteng - list of ints, dataset transforms
    :param reduce_size: subsample data -1 for None, val for size of dataset
    :param bin - binarize environments - 0 for No, 1 for yes
    :param toy - If not = [False], = [True, data, y_all, d_atts]
    '''
    if toy[0] == True:
        data, y_all, d_atts = toy[1], toy[2], toy[3]
    elif dname_from_fpath(fname) == 'adult':
        data, y_all, d_atts = adult_dataset_processing(fname, \
                              fteng, reduce_dsize=dsize, \
                              bin=bin, seed=seed, \
                              testing=testing)
    elif dname_from_fpath(fname) == 'german':
        assert (seed==1000)
        data, y_all, d_atts = german_credit_dataset_processing(fname, \
                              fteng, bin=bin, \
                              testing=testing)

    return data, y_all, d_atts


def data_conversion(data, categorical_feats, continous_feats, predictor, fteng):
    '''

    :param data: Dataframe of cleaned data - feats & labels
    :param categorical_feats: categorical feats (all including labels)
    :param continous_feats: (all including labels)
    :param predictor_feats: (str) single label. Assume is Categorical, Binary
    :param fteng: ids of feature manipulations wanted
    :return:
    '''

    labels = data.pop(predictor)

    # Categorical to one-hot
    data = pd.get_dummies(data, columns=categorical_feats, drop_first=True)
    labels = pd.get_dummies(labels, columns=[predictor], drop_first=True)

    #Feature engineering
    orig_cols = data.columns

    if 1 in fteng:  #x^2 continous feats
        for col in orig_cols:
            if col in continous_feats:  #Assume only transform orig cont col
                data[(col + '_sq')] = data[col] ** 2

    if 2 in fteng:  #add new col with multiplied feats
            for cp in [com for com in combinations(orig_cols, 2) \
                       if (com[0].split('_')[0] != com[1].split('_')[0])]:
                data[(cp[0] + '_x_' + cp[1])] = data[cp[0]] * data[cp[1]]

    #Note - doing a logical or of features will break dummy encoding


    #Get the unmodified attribute classes for possible envs
    all_cats={cat:[] for cat in (categorical_feats + continous_feats)}

    for col in data.columns:
        for cat in all_cats:
            if (cat == col) or (((cat+'_') in col) and ('_sq' not in col) \
                                        and ('_x_' not in col)):
                all_cats[cat].append(col)
    for cat in categorical_feats:  #Dummy case
        all_cats[cat].append(cat + '_DUMmY')

    print(data.shape)
    return data, labels, all_cats

def adult_dataset_processing(fname, fteng, reduce_dsize=-1, bin=False, seed=1000, testing=False):
    '''Process the adult dataset from csv. Return the dataframe, as well as a
        list of columns that must be treated as one block during the enumeration of plausible causal predictors

        fteng: List of the feature engineering steps to do on data.
        1) Introtuce products of cat + cont vars
        2) Introduce square of cont vars
        3) Convert education to continous
        reduce_dsize: The size of randomly sampled rows from dset to take. -1 if not applicable

        :return: cleaned dataframe with info
        '''

    seed = 0

    # 2.1 YOUR CODE HERE
    data = pd.read_csv(fname)

    # let's first count how many missing entries there are for each feature
    col_names = data.columns
    num_rows = data.shape[0]

    # next let's throw out all rows (samples) with 1 or more "?"
    # Hint: take a look at what data[data["income"] != ">50K"] returns
    # Hint: if data[field] do not contain strings then data["income"] != ">50K" will return an error

    ######

    # 2.3 YOUR CODE HERE
    for feature in col_names:
        try:
            data = data.loc[data[str(feature)] != "?"]
        except:  # If the column is all numbers and str comparisons dont work
            pass
            ######

    #Deal with the external forced dataset size reduction
    if reduce_dsize != -1:
        assert reduce_dsize > 0
        data = data.sample(n=reduce_dsize, random_state=seed)

    #Get rid of unwanted columns before making feat lists
    data.drop('educational-num', axis=1, inplace=True)
    data.drop('fnlwgt', axis=1, inplace=True)
    data = data[data['native-country'] != 'South']  #no entries
    data = data[(data['workclass'] != 'Without-pay') & \
            (data['workclass'] != 'Never-worked')] #small num-_entries (21)
    data = data[(data['occupation'] != 'Armed-Forces')] #small num-_entries (14)

    #NOTE - These lists aren't MECE
    #cat_feat:{acceptable stratifications:orig_cols corresponding}
    cat_feats = {'workclass':{'selfWork':['Private', 'Self-emp-not-inc', 'Self-emp-inc'], \
                              'govWork':['Federal-gov', 'Local-gov', 'State-gov']}, \
                 'education':{}, \
                 'marital-status':{'married':['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse', 'Widowed'],\
                                   'divorced':['Divorced', 'Separated'], \
                                   'neither':['Never-married']}, \
                 'occupation':{'craft':['Craft-repair'],
                               'professional':['prof-specialty'],
                               'manager':['Exec-managerial'],
                               'secretary':['Adm-clerical'],
                               'sales':['Sales'],
                               'other':['Other-service', 'Protective-serv', 'Priv-house-serv'],
                               'machine':['Machine-op-inspct'],
                               'transport':['Transport-moving'],
                               'cleaners':['Handlers-cleaners'],
                               'agriculture':['Farming-fishing'],
                               'infotech':['Tech-support'],}, \
                 'relationship':{'spouse':['Husband', 'Wife'],
                                 'nofam':['Unmarried', 'Other-relative', 'Not-in-family'],
                                 'ownchild':['Own-child']}, \
                 'race':{}, \
                 'gender':{}, \
                 'native-country':{'highHDI':['United-States', 'England', 'Canada', 'Germany', 'Japan', 'Greece', \
                                              'Italy', 'Poland', 'Portugal', 'Ireland', 'France', 'Hungary', 'Scotland', 'Hong', 'Holand-Netherlands'], \
                                   'midHDI':['Puerto-Rico', 'Outlying-US(Guam-USVI-etc)', 'Cuba', 'Iran', 'Jamaica', \
                                             'Mexico', 'Dominican-Republic', 'Ecuador', 'Taiwan', 'Columbia', 'Thailand', 'Yugoslavia', \
                                             'Trinadad&Tobago', 'Peru'], \
                                   'lowHDI':['Cambodia', 'India', 'China', 'Honduras', 'Philippines', 'Vietnam', 'Laos', 'Haiti', \
                                             'Guatemala', 'Nicaragua', 'El-Salvador']}
                 }
    cont_feats = ['age', 'capital-gain', \
                       'capital-loss', 'hours-per-week']
    pred_feats = 'income'

    #Custom binarize the stratification categories
    if bin:
        print('hi')
        for ft in cat_feats:
            for agg_ft in cat_feats[ft]:
                data[ft] = data[ft].apply(lambda val: agg_ft if val in cat_feats[ft][agg_ft] else val)

    if testing:  #Return dataset before processing for testing
        return data

    return data_conversion(data, list(cat_feats.keys()), cont_feats, pred_feats, fteng)

    #Custom binarize the stratification categories
    # if testing:
    #     print(type(data_fteng))
    #     print(data_fteng.columns.values)
    #     with open('adult_testy.txt', 'w') as f:
    #         for item in data_fteng.columns.values:
    #             f.write(item+"\n")
    #     quit()


def german_credit_dataset_processing(fname, fteng=[], bin=False, testing=False):
    data = pd.read_csv(fname)

    #Get rid of unwanted stuff before making feat lists
    data = data[data['Purpose'] != 10]

    # NOTE - These dictionary partitions are MECE over all possible att vals
    cat_feats = {'Purpose':{'investment':[6, 8, 9], \
                            'car':[0, 1], \
                            'domestic_needs':[2, 3, 4, 5]}, \
                 'Savings':{}, \
                 'Personal':{}, \
                 'OtherDebtors':{}, \
                 'Property':{'1':[1],
                             '2':[2],
                             '3':[3],
                             '4':[4]}, \
                 'OtherInstallmentPlans':{}, \
                 'Housing':{'1':[1],
                            '2':[2],
                            '3':[3]}, \
                 'Telephone':{'none':[1], 'registered':[2]}, \
                 'Foreign':{}  #No balance for env-split
                 }

    cont_feats = ['CreditAmount', 'InstallmentDisposable', 'PresRes', \
                       'NumExistCredits', 'CheqAccountStatus', 'Duration', 'CreditHistory', \
                     'PresentEmployment', 'Age', 'Job', 'Deps']
    pred_feats = 'Labels'
    in_order = ['Labels', 'CheqAccountStatus', 'Duration', 'CreditHistory', 'Purpose', \
                'CreditAmount', 'Savings', 'PresentEmployment', \
                'InstallmentDisposable', 'Personal', 'OtherDebtors', 'PresRes', \
                'Property', 'Age', 'OtherInstallmentPlans', 'Housing', \
                'NumExistCredits', 'Job', 'Deps', 'Telephone', 'Foreign']

    # Note - for knowing which column is which, might be good to associate numbers with labels
    data.columns = in_order

    #Custom binarize the stratification categories
    if estrat_red:
        for ft in cat_feats:
            for agg_ft in cat_feats[ft]:
                data[ft] = data[ft].apply(lambda val: agg_ft if val in cat_feats[ft][agg_ft] else val)

    if testing:
        return data

    return data_conversion(data, list(cat_feats.keys()), cont_feats, pred_feats, fteng=fteng)

    #Custom binarize the stratification categories




if __name__ == '__main__':
    a,b = german_credit_dataset_processing('data/german_credit.csv', [1,2])
    print(3)
