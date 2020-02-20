import argparse
from time import time

from itertools import combinations
import scipy.signal as sig
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

# from model import MultiLayerPerceptron
# from dataset import AdultDataset
# from util import *

#import matplotlib.pyplot as plt

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
    for cat in all_cats:  #Dummy case
        all_cats[cat].append(cat + '_DUMmY')

    print(data.shape)
    return data, labels, all_cats

def adult_dataset_processing(fname, fteng):
    '''Process the adult dataset from csv. Return the dataframe, as well as a
        list of columns that must be treated as one block during the enumeration of plausible causal predictors

        fteng: List of the feature engineering steps to do on data. 
        1) Introtuce products of cat + cont vars
        2) Introduce square of cont vars 
        3) Convert education to continous
        :return: cleaned dataframe with info
        '''

    """ Adult income classification

    In this lab we will build our own neural network pipeline to do classification on the adult income dataset. More
    information on the dataset can be found here: http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html

    """
    seed = 0

    # =================================== LOAD DATASET =========================================== #

    ######

    # 2.1 YOUR CODE HERE
    data = pd.read_csv(fname)

    ######

    # =================================== DATA VISUALIZATION =========================================== #

    # the dataset is imported as a DataFrame object, for more information refer to
    # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
    # we can check the number of rows and columns in the dataset using the .shape field
    # to get a taste of what our datset looks like, let's visualize the first 5 rows of the dataset using the .head() method
    # the task will be to predict the "income" field (>50k or <50k) based on the other fields in the dataset
    # check how balanced our dataset is using the .value_counts() method.



    # =================================== DATA CLEANING =========================================== #

    # datasets often come with missing or null values, this is an inherent limit of the data collecting process
    # before we run any algorithm, we should clean the data of any missing values or unwanted outliers which could be
    # detrimental to the performance or training of the algorithm. In this case, we are told that missing values are
    # indicated with the symbol "?" in the dataset

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

    #NOTE - These lists aren't MECE
    cat_feats = ['workclass', 'education', 'marital-status', \
                         'occupation', 'relationship', 'race', 'gender', \
                         'native-country']
    cont_feats = ['age', 'fnlwgt', 'educational-num', 'capital-gain', \
                       'capital-loss', 'hours-per-week']
    pred_feats = 'income'

    return data_conversion(data, cat_feats, cont_feats, pred_feats, fteng)


def german_credit_dataset_processing(fname, fteng=[]):
    data = pd.read_csv(fname)
    data = data.drop('Telephone', axis=1)

    # NOTE - These lists are MECE
    cat_feats = ['Purpose', 'Savings', 'Personal', 'OtherDebtors', \
                         'Property', 'OtherInstallmentPlans', 'Housing', \
                         'Foreign']
    cont_feats = ['CreditAmount', 'InstallmentDisposable', 'PresRes', \
                       'NumExistCredits', 'CheqAccountStatus', 'Duration', 'CreditHistory', \
                     'PresentEmployment', 'Age', 'Job', 'Deps']
    pred_feats = 'Labels'
    in_order = ['Labels', 'CheqAccountStatus', 'Duration', 'CreditHistory', 'Purpose', \
                'CreditAmount', 'Savings', 'PresentEmployment', \
                'InstallmentDisposable', 'Personal', 'OtherDebtors', 'PresRes', \
                'Property', 'Age', 'OtherInstallmentPlans', 'Housing', \
                'NumExistCredits', 'Job', 'Deps', 'Foreign']

    # Note - for knowing which column is which, might be good to associate numbers with labels
    data.columns = in_order

    return data_conversion(data, cat_feats, cont_feats, pred_feats, fteng=fteng)

if __name__ == '__main__':
    a,b = german_credit_dataset_processing('data/german_credit.csv', [1,2])
    print(3)