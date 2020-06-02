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
            env_list=list_2_string(env, '--')
        )
    elif (a == 'linreg'):
        uniqueid = '''{id}_{algo}_{data}'''
        uniqueid= uniqueid.format(
            id=id,
            algo=a,
            data=utils.dname_from_fpath(data),
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
    parser.add_argument('datafname', type=str, \
                        help='data.csv')
    parser.add_argument('expdir', type=str, \
                        help='where dir is')
    parser.add_argument("cmdfile", type=str, \
                       help="filename to write all commands")
    parser.add_argument("paramfile", type=str, \
                       help="filename of pickled padas df storing hyperparams")

    parser.add_argument("-env_list", type=str, default='[]', \
                        help="All the environment variables to split on")
    parser.add_argument("-envcombos", type=str, default='single')
    parser.add_argument("-reduce_dsize", type=int, default=-1)
    parser.add_argument('-fteng', type=str, help='c-interval val', default='1')
    parser.add_argument("-binarize", type=int, default=0)
    parser.add_argument("-eq_estrat", type=int, default=-1)
    parser.add_argument("-seed", type=int, default=100)
    parser.add_argument("-val_info", type=str, default=['-1'])
    parser.add_argument("--testing", action='store_true')

    args = parser.parse_args()

    if args.testing:
        print("id:", args.id)
        print("algo:", args.algo)
        print("expdir:", args.expdir)
        print("data:", args.datafname)
        print("cmdfile:", args.cmdfile)
        print("paramfile:", args.paramfile)
        print("env_list:", args.env_list)
        print("envcombo?:", args.envcombos)
        print("feat_eng:", args.fteng)
        print("d_size:", args.reduce_dsize)
        print("testing?:", args.testing)
        print("binarize?:", args.binarize)
        print("eq_estrat?:", args.eq_estrat)
        print("seed?:", args.seed)
        print("val_info?:", args.val_info)
        quit()
    assert args.envcombos == 'single'
    e_list = utils.str_2_strlist_parser(args.env_list)

    if len(e_list) > 0:
        #Set up evironment combos
        if args.envcombos == 'all_combos':
            allenvs = utils.powerset(e_list)
        elif args.envcombos == 'single':
            allenvs = [[a] for a in e_list]

        #Set up considerations around validation
        if args.val_info != '[-1]':
            val_e_list = utils.str_2_strlist_parser(args.val_info)
            assert len(val_e_list) == len(e_list)

            if args.envcombos == 'all_combos':
                allvalenvs = utils.powerset(val_e_list)
            elif args.envcombos == 'single':
                allvalenvs = [[a] for a in val_e_list]
        else:
            allvalenvs = ['[-1]']*len(e_list)

        for i, _ in enumerate(allenvs):
            id = str(int(args.id) + i)
            uniqueid = unid_from_algo(id, a=args.algo, \
                                      data=args.datafname, env=allenvs[i])

            #Write Exp Command to commandfile
            with open(args.cmdfile, 'a') as f:
                command_str = \
                    '''python main.py {id} {algo} {data} {expdir} -fteng {feat_eng} -reduce_dsize {d_size} -binarize {bin} -eq_estrat {eq} -seed {s} -env_atts {env_list} -val_info {val}\n'''

                command_str = command_str.format(
                    id=id,
                    algo=args.algo,
                    data=args.datafname,
                    expdir=args.expdir,
                    feat_eng=args.fteng,
                    d_size=args.reduce_dsize,
                    bin=args.binarize,
                    eq=args.eq_estrat,
                    s=args.seed,
                    env_list=allenvs[i],
                    val=allvalenvs[i]
                )
                f.write(command_str)

            #Log Parameters in Datafame
            addnxt = pd.DataFrame([id, args.algo, args.fteng, \
            utils.dname_from_fpath(args.datafname), args.seed, args.reduce_dsize, \
            args.binarize, args.eq_estrat, list_2_string(allenvs[i], ' '), \
            list_2_string(allvalenvs[i], ' ')]).T

            if i == 0:
                add = addnxt
            else:
                 add = add.append(addnxt)

        parameter_cols = ['Id', 'Algo', 'Fteng', 'Dataset', \
                            'Seed', 'ReduceDsize', 'Bin', 'Eq_Estrat', \
                            'Envs', 'Val']

    else:

        id = str(int(args.id) + 1)
        uniqueid = unid_from_algo(id, a=args.algo, \
                                  data=args.datafname)

        #Deal with validation
        assert ('-1' in args.val_info) or (args.val_info.strip('[').strip(']').strip('-').isdigit())

        #Write Exp Command to commandfile
        with open(args.cmdfile, 'a') as f:
            command_str = \
                '''python main.py {id} {algo} {data} {expdir} -fteng {feat_eng} -reduce_dsize {d_size} -binarize {bin} -seed {s} -val_info {val}\n'''

            command_str = command_str.format(
                id=id,
                algo=args.algo,
                data=args.datafname,
                expdir=args.expdir,
                feat_eng=args.fteng,
                d_size=args.reduce_dsize,
                bin=args.binarize,
                s=args.seed,
                val=args.val_info
            )
            f.write(command_str)

        #Log Parameters in Datafame
        add = pd.DataFrame([id, args.algo, args.fteng, \
                            utils.dname_from_fpath(args.datafname), args.seed, \
                             args.reduce_dsize, args.binarize, args.val_info]).T

        parameter_cols = ['Id', 'Algo', 'Fteng', 'Dataset', 'Seed', 'ReduceDsize', 'Bin', 'Val']


    #Save parameters in dataframe
    try:
        param_df = pd.read_pickle(args.paramfile)
        add.columns = [param_df.index.name] + list(param_df.columns)
        add = add.set_index('Id')
        param_df = param_df.append(add)
        print(param_df.shape)
        pd.to_pickle(param_df, args.paramfile)

    except:  #Df not yet initialized
        param_df = add
        param_df.columns = parameter_cols
        param_df.set_index('Id', inplace=True)
        pd.to_pickle(param_df, args.paramfile)
