'''Given a set of parameters (python func args), generates cmdfile with
commands with all combinations of argument inputs '''

import argparse
import itertools
import pandas as pd
import os
import utils

def unid_from_algo(id, a=None, data=None, env=None):
    '''Generate name of experiment filename descriptor
    :param id: Unique numerical identiifer
    :param a: Name of algo used (String)
    :param feat: Feature modifications on dset (string)
    :param data: Filepath to dataset (string)
    :param env: The env names to be included (list)
    '''

    if (a == 'icp') or (a == 'irm'):
        uniqueid = '''{id}_{algo}_{data}_{env_list}'''
        uniqueid= uniqueid.format(
            id=id,
            algo=a,
            data=utils.dname_from_fpath(data),
            env_list=list_2_string(e, '--')
        )
    else:
        raise Exception('Unimplemented Dataset')
    return uniqueid

def list_2_string(elems, bchar):
    '''

    :param elems: all elems want to line up
    :param bchar: buffer character
    :return:
    '''
    uid=''
    for i, e in enumerate(elems):
        if len(elems) == 1:
            uid = '{}'.format(e)
        elif i == 0:
            uid = '{}{}{}'.format(uid, e, bchar)
        elif i == (len(elems) -1):
            uid = '{}{}'.format(uid, e)
        else:
            uid = '{}{}{}'.format(uid, e, bchar)
    return uid

def threshold(num):
    if num == 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('id', type=str, \
                        help='unique id of exp')
    parser.add_argument('algo', type=str, \
                        help='prediciton algo used')
    parser.add_argument('feat_eng', type=str, \
                        help='c-interval val')
    parser.add_argument('datafname', type=str, \
                        help='data.csv')
    parser.add_argument('expdir', type=str, \
                        help='where dir is')
    parser.add_argument("cmdfile", type=str, \
                       help="filename to write all commands")
    parser.add_argument("paramfile", type=str, \
                       help="filename of pickled padas df storing hyperparams")
    parser.add_argument("env_list", nargs='+', \
                        help="All the environment variables to split on")

    parser.add_argument("-envcombos", type=str, required=True)
    parser.add_argument("-reduce_dsize", type=int, required=True)
    parser.add_argument("-binarize", type=int, required=True)
    parser.add_argument("-eq_estrat", type=int, default=-1)
    parser.add_argument("-seed", type=int, default=100)
    parser.add_argument("--testing", action='store_true')

    args = parser.parse_args()

    if args.testing:
        print("id:", args.id)
        print("algo:", args.algo)
        print("feat_eng:", args.feat_eng)
        print("expdir:", args.expdir)
        print("data:", args.datafname)
        print("cmdfile:", args.cmdfile)
        print("paramfile:", args.paramfile)
        print("env_list:", args.env_list)
        print("envcombo?:", args.envcombos)
        print("d_size:", args.reduce_dsize)
        print("testing?:", args.testing)
        print("binarize?:", args.binarize)
        print("eq_estrat?:", args.eq_estrat)
        print("seed?:", args.seed)
        quit()

    if args.envcombos == 'all_combos':
        allenvs = utils.powerset(args.env_list)
    elif args.envcombos == 'single':
        allenvs = [[a] for a in args.env_list]

    for i, e in enumerate(allenvs):
        id = str(int(args.id) + i)
        uniqueid = unid_from_algo(id, a=args.algo, \
                                  data=args.datafname, env=e)

        if args.testing:
            print(uniqueid)
            print(((args.datafname).split('/')[-1]))
            quit()

        #Write Exp Command to commandfile
        with open(args.cmdfile, 'a') as f:
            spacing = len(list_2_string(e, ' '))
            command_str = \
                '''python main.py {id} {algo} {feat_eng} {data} {expdir}{e_spacing}{env_list} -reduce_dsize {d_size} -binarize {bin} -eq_estrat {eq} -seed {s}\n'''

            command_str = command_str.format(
                id=id,
                algo=args.algo,
                feat_eng=args.feat_eng,
                data=args.datafname,
                expdir=args.expdir,
                e_spacing=(' ' * threshold(len(list_2_string(e, ' ')))),
                env_list=list_2_string(e, ' '),
                d_size=args.reduce_dsize,
                bin=args.binarize,
                eq=args.eq_estrat,
                s=args.seed
            )
            f.write(command_str)

        #Log Parameters in Datafame
        addnxt = pd.DataFrame([id, args.algo, args.feat_eng, \
        utils.dname_from_fpath(args.datafname), args.seed, args.reduce_dsize, \
        args.binarize, args.eq_estrat, list_2_string(e, ' ')]).T
        if i == 0:
            add = addnxt
        else:
             add = add.append(addnxt)

    #Save parameters in dataframe
    try:
        param_df = pd.read_pickle(args.paramfile)
        add.columns = param_df.columns
        param_df = param_df.append(add)
        pd.to_pickle(param_df, args.paramfile)

    except:  #Df not yet initialized
        if (args.algo == 'icp') or (args.algo == 'irm'):
            param_df = add
            param_df.columns = ['Id', 'Algo', 'Fteng', 'Dataset', \
                                'Seed', 'ReduceDsize', 'Bin', 'Eq_Estrat', \
                                'Envs']
            pd.to_pickle(param_df, args.paramfile)
        else:
            raise Exception('Algorithm not yet implemented')
