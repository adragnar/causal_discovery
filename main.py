import argparse
import logging
import os
import random
import warnings

import algo_hyperparams as ahp
import data_processing as dp
import models
from utils import dname_from_fpath, proc_fteng

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def default(id_val, algo, dataset_fname, expdir, env_atts_types, \
            feateng_type='-1', d_size=-1, bin_env=False, eq_estrat=-1, SEED=100,
            test_info='-1',val_split=0.0, irm_args={}, linear_irm_args={}, \
            mlp_args={}, linreg_args={}, logreg_args={}, \
            toy_data=[False], testing=False):

    '''
    :param id_val: Numerical identifier for run (str)
    :param algo: Name of algo to be applied (str)
    :param dataset_fname: path to dataset (str)
    :param expdir: directory where results stored (str)
    :param env_atts_types: environment vairanbbles in list (list len=1)
    :param feateng_type: The feature engineering steps to be taken (str)
    :param d_size: subsampling number on dataset (int)
    :param bin_env: 0 to not bin, 1 to bin (0-1 int)
    :param eq_estrat: -1 to not apply, int for min num samples per env (int)
    :param seed: random seed used (int)
    :param test_info: -1 or name of test environment (str)
    '''

    random.seed(SEED)

    #Meta-function Accounting
    unid = '''{}_{}_{}_{}_{}_{}'''.format(id_val, feateng_type,\
                                          dname_from_fpath(dataset_fname), \
                                          str(d_size), \
                                          str(SEED), \
                                          ''.join([str(f) \
                                                   for f in env_atts_types])
                                         )
    logger_fname = os.path.join(expdir, 'log_{}.txt'.format(unid))
    logging.basicConfig(filename=logger_fname, level=logging.DEBUG)
    logging.info('id: {}'.format(id_val))
    logging.info('algo: {}'.format(algo))
    logging.info('fteng: {}'.format(feateng_type))
    logging.info('dataset: {}'.format(dname_from_fpath(dataset_fname)))
    logging.info('env_atts: {}'.format(str(env_atts_types)))
    logging.info('dataset size: {}'.format(str(d_size)))
    logging.info('binarize envs: {}'.format(str(bin_env)))
    logging.info('equalize envs: {}'.format(str(eq_estrat)))
    logging.info('seed: {}'.format(str(SEED)))
    logging.info('test_info: {}'.format(test_info))

    #Select correct dataset
    data, y_all, d_atts = dp.data_loader(dataset_fname, \
                                proc_fteng(feateng_type), dsize=d_size, \
                                binar=bin_env, toy=toy_data, testing=testing)
    logging.info('{} Dataset loaded - size {}'.format(\
                                dataset_fname.split('/')[-1], str(data.shape)))

    # #Remove Validation and Test Data
    data, y_all, d_atts, _, _, _, _ = dp.train_val_test_split(\
                                         data, y_all, d_atts, val_split, \
                                         test_info, SEED)

    logging.info('Val, Test Environment Removed - Dataset size {}'.format(\
                                                              str(data.shape)))

    if algo == 'icp':
        icp = models.InvariantCausalPrediction()
        icp.run(data, y_all, d_atts, unid, expdir, proc_fteng(feateng_type), \
                 env_atts_types)

    elif algo == 'irm':
        assert irm_args > 0
        logging.info('irm_params: {}'.format(str(irm_args)))
        irm = models.InvariantRiskMinimization()
        irm.run(data, y_all, d_atts, unid, expdir, SEED, env_atts_types, \
                eq_estrat, irm_args)

    elif algo == 'linear-irm':
        assert linear_irm_args > 0
        logging.info('linear-irm_params: {}'.format(str(linear_irm_args)))
        l_irm = models.LinearInvariantRiskMinimization()
        l_irm.run(data, y_all, d_atts, unid, expdir, SEED, env_atts_types, \
                  eq_estrat, linear_irm_args)

    elif algo == 'mlp':
        mlp = models.MLP()
        mlp.run(data, y_all, unid, expdir, mlp_args)

    elif algo == 'linreg':
        linreg = models.Linear()
        linreg.run(data, y_all, unid, expdir, linreg_args)

    elif algo == 'logreg':
        logreg = models.LogisticReg()
        logreg.run(data, y_all, unid, expdir, logreg_args)

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
    parser.add_argument('-test_info', type=str, default='[]', \
                        help='test env per environment')
    parser.add_argument("--testing", action='store_true')

    #Additions for Hyperparameter Tuning
    parser.add_argument('-inc_hyperparams', type=int, default=0)
    parser.add_argument('-val_split', type=float, default=0.0)

    #Regression
    parser.add_argument('-linreg_lambda', type=float, default=0.0)
    parser.add_argument('-logreg_c', type=float, default=1.0)

    #Vanilla MLP
    parser.add_argument('-mlp_lr', type=float, default=0.001)
    parser.add_argument('-mlp_niter', type=int, default=5000)
    parser.add_argument('-mlp_l2', type=float, default=0.001)
    parser.add_argument('-mlp_hid_layers', type=int, default=100)

    #IRM
    parser.add_argument('-irm_lr', type=float, default=0.001)
    parser.add_argument('-irm_niter', type=int, default=5000)
    parser.add_argument('-irm_l2', type=float, default=0.001)
    parser.add_argument('-irm_penalty_weight', type=float, default=10000)
    parser.add_argument('-irm_penalty_anneal', type=float, default=100)
    parser.add_argument('-irm_hid_layers', type=int, default=100)


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
        print("test_info:", args.test_info)
        print("testing?:", args.testing)
        quit()

    if args.inc_hyperparams == 1:
        irm_args = {'lr':args.irm_lr, \
                     'n_iterations':args.irm_niter, \
                     'penalty_anneal_iters':args.irm_penalty_anneal, \
                     'l2_reg':args.irm_l2, \
                     'pen_wgt':args.irm_penalty_weight, \
                     'hid_layers':args.irm_hid_layers, \
                     'verbose':True}
        linear_irm_args = {'lr':args.irm_lr, \
                            'n_iterations':args.irm_niter, \
                            'penalty_anneal_iters':args.irm_penalty_anneal, \
                            'l2_reg':args.irm_l2, \
                            'pen_wgt':args.irm_penalty_weight, \
                            'hid_layers':args.irm_hid_layers, \
                            'verbose':True}

        mlp_args = {'lr':args.mlp_lr, \
                     'n_iterations':args.mlp_niter, \
                     'l2_reg':args.mlp_l2, \
                     'hid_layers':args.mlp_hid_layers}

        linreg_args = {'lambda':args.linreg_lambda}
        logreg_args = {'C':args.logreg_c}
    else:
        irm_args = ahp.get_irm_args(args.data_fname)
        linear_irm_args = ahp.get_linear_irm_args(args.data_fname)
        mlp_args = ahp.get_mlp_args(args.data_fname)
        linreg_args = ahp.get_linreg_args(args.data_fname)
        logreg_args = ahp.get_logreg_args(args.data_fname)


    default(args.id, args.algo, args.data_fname, args.expdir, [args.env_atts], \
           feateng_type=args.fteng, d_size=args.reduce_dsize, \
           bin_env=bool(args.binarize), eq_estrat=args.eq_estrat, \
           SEED=args.seed, test_info=args.test_info, val_split=args.val_split, \
           testing=args.testing, irm_args=irm_args, \
           linear_irm_args=linear_irm_args, mlp_args=mlp_args, \
           linreg_args=linreg_args, logreg_args=logreg_args)
