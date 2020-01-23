import argparse
from time import time

import scipy.signal as sig
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

# from model import MultiLayerPerceptron
# from dataset import AdultDataset
# from util import *

#import matplotlib.pyplot as plt

def adult_dataset_processing(fname):
    '''Process the adult dataset from csv. Return the dataframe, as well as a
    list of columns that must be treated as one block during the enumeration of plausible causal predictors'''

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

    ######

    # 2.2 YOUR CODE HERE

    print("shape is ", data.shape)
    print("columns are:", data.columns)
    print("first 5 rows: \n", data.head())
    print(data["income"].value_counts())

    ######


    # =================================== DATA CLEANING =========================================== #

    # datasets often come with missing or null values, this is an inherent limit of the data collecting process
    # before we run any algorithm, we should clean the data of any missing values or unwanted outliers which could be
    # detrimental to the performance or training of the algorithm. In this case, we are told that missing values are
    # indicated with the symbol "?" in the dataset

    # let's first count how many missing entries there are for each feature
    col_names = data.columns
    num_rows = data.shape[0]
    for feature in col_names:
        ######

        # 2.3 YOUR CODE HERE
        print("For column", feature, data[feature].isin(["?"]).sum())
        ######
    print(3)
    # next let's throw out all rows (samples) with 1 or more "?"
    # Hint: take a look at what data[data["income"] != ">50K"] returns
    # Hint: if data[field] do not contain strings then data["income"] != ">50K" will return an error

        ######

        # 2.3 YOUR CODE HERE
    for feature in col_names:
        try:
            data = data.loc[data[str(feature)] != "?"]
            print(data.shape)
        except:  #If the column is all numbers and str comparisons dont work
            pass
    print("cleaned shape: ", data.shape)
        ######



    #Categorical to one-hot
    categorical_feats = ['workclass',  'education', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'native-country', 'income', ]

    data = pd.get_dummies(data, prefix=categorical_feats, drop_first=True)
    #print(data.head())


    #Get the relevant attribute classes
    all_cats = {'age':[], 'workclass':[], 'fnlwgt':[], 'education':[], 'educational-num':[], 'marital-status':[], \
                'occupation':[], 'relationship':[], \
     'race':[], 'sex':[], 'capital-gain':[], 'capital-loss':[], 'hours-per-week':[], 'native-country':[]}

    for col in data.columns:
        for cat in all_cats:
            if (cat == col) or ((cat+'_') in col):
                all_cats[cat].append(col)

    return data, all_cats

