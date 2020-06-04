import argparse
import csv
import enum
import json
import pickle
import os
import shutil

import numpy as np
import pandas as pd

import copy
import itertools
from collections import Counter
import pprint

import models
import data_processing as dp
import utils

def get_id_from_fname(f):
    return f.split('_')[1]

def get_ftype_from_fname(f):
    return f.split('_')[0]

def open_pvals(filename):
    try:
        pvals = json.load(open(filename, 'rb'))
        del pvals["()"]
    except:
        return None
    pvals = pd.DataFrame.from_dict(pvals, orient='index')
    return pvals

def str_2_pcp(pcpstr):
    pcpstr = (pcpstr.split('(')[1]).split(')')[0]
    pcpstr = pcpstr.replace(' ', '')
    ret = set([e.strip("'") for e in pcpstr.split(',')])
    ret.discard('')
    return ret

class POS(enum.Enum):
   big = 1
   small = 2
   perf = 3

#Alpha tuning values
START_ALPHA = 1.0
FACTOR = 2
EPS = 1e-20
#Part 2
STEP = 1e-2
FACTOR2 = 2
EPS2 = 1e-10

def alpha_tune(pVals, amin, flag=0):
    #First find a CP returning alpha
    a0 = START_ALPHA
    bounds0 = [0, 100.0]
    cp_ret = False
    while not cp_ret:
        pos = 0
        accepted = pVals[pVals['Final_tstat'] > a0]

        #Determine position of alpha
        if len(accepted.index) == 0:
            pos = POS.big
        else:
            accepted_sets = [str_2_pcp(a) for a in list(accepted.index)]
            causal_preds = set.intersection(*accepted_sets)
            if len(causal_preds) == 0:
                pos = POS.small
            else:
                pos = POS.perf
                cp_ret = True

                if flag:
                    print(causal_preds)
                    print(a0)

                continue

        #Determine what alpha to check next
        if pos == POS.big:
            bounds0[1] = a0
            if a0/FACTOR <= bounds0[0]:
                a0 = a0 - abs((a0 - bounds0[0])/2)
            else:
                a0 = a0/FACTOR
        elif pos == POS.small:
            bounds0[0] = a0
            if a0 * FACTOR >= bounds0[1]:
                a0 = a0 + abs((a0 - bounds0[1])/2)
            else:
                a0 = a0 * FACTOR

        #Stability check in case no CPs
        if abs(bounds0[0] - bounds0[1]) < EPS:
            return (-1, -1)

    #Then establish interval bounds
    lowerB = [0, a0]
    upperB = [a0, 100]

    #Upper Bound
    a1 = a0
    step = STEP
    pos = POS.perf
    while abs(upperB[0] - upperB[1]) > EPS2:
        a1 = a1 + step
        accepted = pVals[pVals['Final_tstat'] > a1]

        #Determine position of alpha
        if len(accepted.index) == 0:
            pos = POS.big
        else:
            pos = POS.perf

        #Determine what alpha to check next
        if pos == POS.perf:
            upperB[0] = a1
            if a1 + abs(step * FACTOR2) >= upperB[1]:
                step = abs(a1 - upperB[1])/FACTOR2
            else:
                step = abs(step * FACTOR2)
        elif pos == POS.big:
            upperB[1] = a1
            if (a1 - abs(step * FACTOR2)) <= upperB[0]:
                step = -1 * abs(a1 - upperB[0])/FACTOR2
            else:
                step = -1 * abs(step * FACTOR2)
        else:
            assert False

    #Lower Bound
    a2 = a0
    if a2 - STEP > 1e-20:
        step = STEP
    else:
        step = a2/FACTOR2
    pos = POS.perf
    while abs(lowerB[0] - lowerB[1]) > EPS2:
        a2 = a2 - step
        accepted = pVals[pVals['Final_tstat'] > a2]

        #Determine position of alpha
        accepted_sets = [str_2_pcp(a) for a in list(accepted.index)]
        causal_preds = set.intersection(*accepted_sets)
        if len(causal_preds) == 0:
            pos = POS.small
        else:
            pos = POS.perf

        #Determine what alpha to check next
        if pos == POS.perf:
            lowerB[1] = a2
            if a2 - abs(step * FACTOR2) <= lowerB[0]:
                step = abs(a2 - lowerB[0])/FACTOR2
            else:
                step = abs(step * FACTOR2)
        elif pos == POS.small:
            lowerB[0] = a2
            if (a1 + abs(step * FACTOR2)) >= lowerB[1]:
                step = -1 * abs(a2 - lowerB[1])/FACTOR2
            else:
                step = -1 * abs(step * FACTOR2)
        else:
            assert False

    #Check if interval is too close to 0 to be meaningful
    if a2 < amin:
        return (-1, -1)

    #Establish 0-padding to interval
    interval = abs(a1 - a2)/5

    assert (a2 < a0) and (a0 < a1)

    return (max(0, a2 - interval), a1 + interval)


def max_alpha(pVals, arange, eps=1000):
    '''Given a computed range of CP returning alphas (maybe with interval) and pvals for exp, return highest CP returning alpha'''
    ctr = arange[1]
    while ctr > arange[0]:
        accepted = pVals[pVals['Final_tstat'] > ctr]
        if len(accepted.index) > 0:
            return ctr
        else:
            ctr = ctr - (arange[1] - arange[0])/eps
    return -1


def icp_process(res_dir, dset_dir, NUM_POINTS=100, MIN_ALPHA=1e-10):
    ''''''
    expdir = os.path.join(res_dir, 'causal_discovery')
    paramfile = os.path.join(res_dir, 'icp_paramfile.pkl')
    params = pd.read_pickle(paramfile)

    savedir = os.path.join(res_dir, 'processed_results')
    if os.path.exists(savedir):
        shutil.rmtree(savedir)
    os.mkdir(savedir)

    #Collect all raw files
    rawres_files= []
    for f in os.listdir(expdir):
        if ('rawres_' in f):
            rawres_files.append(f)

    #Generate alphas
    for col in ['alpha_start', 'alpha_stop', 'max_alpha']:
        params[col] = np.nan

    alphas = {}
    for fname in rawres_files:
        pvals = open_pvals(os.path.join(expdir, fname))
        if pvals is None:
            continue
        id = get_id_from_fname(fname)
        arange = alpha_tune(pvals, MIN_ALPHA)
        params.loc[id, 'alpha_start'] = arange[0]
        params.loc[id, 'alpha_stop'] = arange[1]
        params.loc[id, 'max_alpha'] = max_alpha(pvals, arange)

    #Generate All Other Derivative Results
    params['coeffs'] = np.NaN
    for fname in rawres_files:
        id = get_id_from_fname(fname)

        #Get Data
        if params.loc[id, 'Dataset'] ==  'adult':
            dataset_fname = os.path.join(dset_dir, 'adult.csv')
        elif params.loc[id, 'Dataset'] ==  'german':
            dataset_fname = os.path.join(dset_dir, 'germanCredit.csv')
        else:
            raise Exception('Dataset not imlemented')

        data, y_all, d_atts = dp.data_loader(dataset_fname, \
                            utils.proc_fteng(params.loc[id, 'Fteng']), \
                            dsize=int(params.loc[id, 'ReduceDsize']), \
                            bin=int(params.loc[id, 'Bin']), \
                            testing=0)

        # env_datts = {e:d_atts[e]}
        # eq_estrat = -1

        #Load pvals
        pvals = open_pvals(os.path.join(expdir, fname))
        if pvals is None:
            continue

        #Get the Causal Predictors, Regressors
        accepted = pvals[pvals['Final_tstat'] > params.loc[id, 'max_alpha']]
        accepted_sets = [str_2_pcp(a) for a in list(accepted.index)]
        causal_preds = set.intersection(*accepted_sets)

        icp = models.InvariantCausalPrediction()
        causal_preds = icp.get_data_regressors(d_atts, causal_preds, \
                                utils.proc_fteng(params.loc[id, 'Fteng']), data)


        ##Get the Coefficients of the Predictor
        res = icp.get_coeffs(causal_preds, data, y_all)  #, env_datts, eq_estrat, params.loc[id, 'Seed'])

        #Store Results in HDD and param df
        coeffs_fname = os.path.join(savedir, '{}_coeffs.pkl'.format(id))
        pd.to_pickle(res, coeffs_fname)
        params.loc[id, 'coeffs'] = coeffs_fname

    pd.to_pickle(params, paramfile)

def irm_process(res_dir, dset_dir):
    expdir = os.path.join(res_dir, 'causal_discovery')
    paramfile = os.path.join(res_dir, 'irm_paramfile.pkl')
    params = pd.read_pickle(paramfile)

    savedir = os.path.join(res_dir, 'processed_results')
    if os.path.exists(savedir):
        shutil.rmtree(savedir)
    os.mkdir(savedir)

    #Load IRM Parameters into dataframe
    params['phi'] = np.NaN
    # params['w'] = np.NaN

    for fname in os.listdir(expdir):
        id = get_id_from_fname(fname)
        ftype = get_ftype_from_fname(fname)
        if ftype == 'phi':
            params.loc[id, 'phi'] = os.path.join(expdir, fname)
        # elif ftype == 'w':
        #     params.loc[id, 'w'] = os.path.join(expdir, fname)

    pd.to_pickle(params, paramfile)

def linreg_process(res_dir, dset_dir):
    expdir = os.path.join(res_dir, 'causal_discovery')
    paramfile = os.path.join(res_dir, 'linreg_paramfile.pkl')
    params = pd.read_pickle(paramfile)

    savedir = os.path.join(res_dir, 'processed_results')
    if os.path.exists(savedir):
        shutil.rmtree(savedir)
    os.mkdir(savedir)

    #Load Coefficients into dataframe
    params['linregressors'] = np.NaN
    for fname in os.listdir(expdir):
        id = get_id_from_fname(fname)
        ftype = get_ftype_from_fname(fname)
        if ftype == 'regs':
            params.loc[id, 'linregressors'] = os.path.join(expdir, fname)

    pd.to_pickle(params, paramfile)

def aggregate_loader(resdir, dsetdir, algo):
    if algo == 'icp':
        icp_process(resdir, dsetdir)
    elif algo == 'irm':
        irm_process(resdir, dsetdir)
    elif algo == 'linreg':
        linreg_process(resdir, dsetdir)
    else:
        raise Exception('algo not implemented')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filename Parameters')
    parser.add_argument("resdir", type=str, help="dirname of results")
    parser.add_argument("dsetdir", type=str, help="dirname of datasets")
    parser.add_argument("algo", type=str, help="algo used")
    args = parser.parse_args()

    aggregate_loader(args.resdir, args.dsetdir, args.algo)
