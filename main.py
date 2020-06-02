import argparse
import csv
import pickle
import itertools
import json
import logging
import os
from sklearn.linear_model import LinearRegression
import warnings
import pandas as pd
from tqdm import tqdm
from itertools import combinations

from utils import powerset, dname_from_fpath, proc_fteng
import data_processing as dp

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

#########################################
import numpy as np
from scipy.stats import f as fdist
from scipy.stats import ttest_ind

import random
import models
import utils

def default(id, algo, dataset_fname, expdir, env_atts_types, feateng_type='-1', \
            d_size=-1, bin_env=False, eq_estrat=-1, SEED=100, val_info=['-1'],
            toy_data=[False], testing=False):

    '''

    :param dataset_fname:
    :param env_atts:
    :param feateng_type: The particular preprocess methodology
    :param logger: filepath to log file
    '''

    random.seed(SEED)

    #Meta-function Accounting
    unid = '''{}_{}_{}_{}_{}_{}'''.format(id, feateng_type,\
                                       dname_from_fpath(dataset_fname), \
                                       str(d_size), \
                                       str(SEED), \
                                       ''.join([str(f) for f in env_atts_types])
                                       )
    logger_fname = os.path.join(expdir, 'log_{}.txt'.format(unid))
    logging.basicConfig(filename=logger_fname, level=logging.DEBUG)
    logging.info('id: {}'.format(id))
    logging.info('fteng: {}'.format(feateng_type))
    logging.info('dataset: {}'.format(dname_from_fpath(dataset_fname)))
    logging.info('env_atts: {}'.format(str(env_atts_types)))
    logging.info('dataset size: {}'.format(str(d_size)))
    logging.info('binarize envs: {}'.format(str(bin_env)))
    logging.info('equalize envs: {}'.format(str(eq_estrat)))
    logging.info('seed: {}'.format(str(SEED)))
    logging.info('val_info: {}'.format(val_info))

    #Select correct dataset
    data, y_all, d_atts = dp.data_loader(dataset_fname, proc_fteng(feateng_type), dsize=d_size, \
                                    bin=bin_env, toy=toy_data, testing=testing)
    logging.info('{} Dataset loaded - size {}'.format(dataset_fname.split('/')[-1], \
                str(data.shape)))

    if algo == 'icp':
        icp = models.InvariantCausalPrediction()
        icp.run(data, y_all, d_atts, unid, expdir, proc_fteng(feateng_type), \
                SEED, env_atts_types, eq_estrat, val=val_info)
    elif algo == 'irm':
        irm = models.InvariantRiskMinimization()
        irm.run(data, y_all, d_atts, unid, expdir, SEED, env_atts_types, eq_estrat, val=val_info)
    elif algo == 'linreg':
        linreg = models.Linear()
        linreg.run(data, y_all, unid, expdir, val=val_info)
    else:
        raise Exception('Algorithm not Implemented')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('id', type=str, \
                        help='unique id of exp')
    parser.add_argument('algo', type=str, \
                        help='prediciton algo used')

    parser.add_argument("data_fname", type=str,
                        help="filename adult.csv")
    parser.add_argument("expdir", type=str, default=None,
                        help="path to location to save files")


    parser.add_argument("-fteng", type=str, \
                        help="each digit id of diff feat engineering")
    parser.add_argument("-reduce_dsize", type=int, default=-1)
    parser.add_argument("-binarize", type=int, required=True)
    parser.add_argument("-eq_estrat", type=int, default=-1)
    parser.add_argument("-seed", type=int, default=100)
    parser.add_argument('-env_atts', type=str, default='[]', \
                        help='atts categorical defining envs')
    parser.add_argument('-val_info', type=str, default='[]', \
                        help='validation envs per environment')
    parser.add_argument("--testing", action='store_true')

    args = parser.parse_args()

    if args.testing:
        print("id:", args.id)
        print("algo:", args.algo)
        print("data:", args.data_fname)
        print("expdir:", args.expdir)
        print("feat_eng:", args.fteng)
        print("d_size:", args.reduce_dsize)
        print("binarize?:", args.binarize)
        print("eq_estrat?:", args.eq_estrat)
        print("seed?:", args.seed)
        print("env_list:", args.env_atts)
        print("val_info:", args.val_info)
        print("testing?:", args.testing)
        quit()

    default(args.id, args.algo, args.data_fname, args.expdir, utils.str_2_strlist_parser(args.env_atts), \
           feateng_type=args.fteng, d_size=args.reduce_dsize, \
            bin_env=bool(args.binarize), eq_estrat=args.eq_estrat, SEED=args.seed, \
            val_info=utils.str_2_strlist_parser(args.val_info), testing=args.testing)
