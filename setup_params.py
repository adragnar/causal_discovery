'''Given a set of parameters (python func args), generates cmdfile with
commands with all combinations of argument inputs '''

import argparse
import pandas as pd
import utils

def cmd_append(cstr, newstr):
    '''Take an existing, properly formatted command string, append the newstr
    (with new flags and vals) to it and reformat'''
    cstr = cstr.replace('\n', ' ')
    cstr = cstr + newstr
    cstr = cstr + '\n'

    assert newstr[0] != ' '
    assert cstr[-1] == '\n'
    return cstr


def unid_from_algo(id_val, a=None, data=None, env=None):
    '''Generate name of experiment filename descriptor
    :param id: Unique numerical identiifer
    :param a: Name of algo used (String)
    :param feat: Feature modifications on dset (string)
    :param data: Filepath to dataset (string)
    :param env: The env names to be included (list)
    '''

    if a in ['icp', 'irm', 'linear-irm']:
        uniqueid = '''{id}_{algo}_{data}_{env_list}'''
        uniqueid = uniqueid.format(
            id=id_val,
            algo=a,
            data=utils.dname_from_fpath(data),
            env_list=list_2_string(env, '--')
        )
    elif a in ['linreg', 'logreg', 'mlp', 'constant']:
        uniqueid = '''{id}_{algo}_{data}'''
        uniqueid = uniqueid.format(
            id=id_val,
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
    uid = ''
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

    parser.add_argument("-env_att", type=str, default='-1', \
                        help="List of environment vairables to split on")
    parser.add_argument("-reduce_dsize", type=int, default=-1)
    parser.add_argument('-fteng', type=str, help='c-interval val', default='1')
    parser.add_argument("-binarize", type=int, default=0)
    parser.add_argument("-eq_estrat", type=int, default=-1)
    parser.add_argument("-seed", type=int, default=100)
    parser.add_argument("-test_info", type=str, default='-1')
    parser.add_argument("--testing", action='store_true')

    #Hyperparameter Testing
    parser.add_argument('-inc_hyperparams', type=int, default=0)
    parser.add_argument('-val_split', type=float, default=0.0)

    #Irm + LinearIRM
    parser.add_argument('-irm_lr', type=float, default=None)
    parser.add_argument('-irm_niter', type=int, default=None)
    parser.add_argument('-irm_l2', type=float, default=None)
    parser.add_argument('-irm_penalty_weight', type=float, default=None)
    parser.add_argument('-irm_penalty_anneal', type=float, default=None)
    parser.add_argument('-irm_hid_layers', type=int, default=None)

    #MLP
    parser.add_argument('-mlp_lr', type=float, default=None)
    parser.add_argument('-mlp_niter', type=int, default=None)
    parser.add_argument('-mlp_l2', type=float, default=None)
    parser.add_argument('-mlp_hid_layers', type=int, default=None)

    #Regressions
    parser.add_argument('-linreg_lambda', type=float, default=None)
    parser.add_argument('-logreg_c', type=float, default=None)

    args = parser.parse_args()

    if args.testing:
        print("id:", args.id)
        print("algo:", args.algo)
        print("expdir:", args.expdir)
        print("data:", args.datafname)
        print("cmdfile:", args.cmdfile)
        print("paramfile:", args.paramfile)
        print("env_att:", args.env_att)
        print("feat_eng:", args.fteng)
        print("d_size:", args.reduce_dsize)
        print("testing?:", args.testing)
        print("binarize?:", args.binarize)
        print("eq_estrat?:", args.eq_estrat)
        print("seed?:", args.seed)
        print("test_info?:", args.test_info)
        quit()

    id_val = args.id

    #First get unique Ids
    if args.algo in ['icp', 'irm', 'linear-irm']:
        uniqueid = unid_from_algo(id_val, a=args.algo, \
                                  data=args.datafname, env=args.env_att)
    elif args.algo in ['linreg', 'logreg', 'mlp', 'constant']:
        uniqueid = unid_from_algo(id_val, a=args.algo, \
                                  data=args.datafname)
    else:
        raise Exception('ALgo Unimplemented')

    #Setup baseline command for all experiment runs
    command_str = \
        '''python main.py {id} {algo} {data} {expdir} -fteng {feat_eng} -reduce_dsize {d_size} -binarize {bin} -seed {s} -test_info {test}\n'''
    command_str = command_str.format(
        id=id_val,
        algo=args.algo,
        data=args.datafname,
        expdir=args.expdir,
        feat_eng=args.fteng,
        d_size=args.reduce_dsize,
        bin=args.binarize,
        s=args.seed,
        test=args.test_info,
    )
    add = [id_val, args.algo, args.fteng, \
           utils.dname_from_fpath(args.datafname), \
           args.seed, args.reduce_dsize, args.binarize, args.test_info]
    parameter_cols = ['Id', 'Algo', 'Fteng', 'Dataset', 'Seed', \
                          'ReduceDsize', 'Bin', 'TestSet']

    #Add flags for invariance algos
    if args.algo in ['icp', 'irm', 'linear-irm']:
        command_str = cmd_append(command_str, \
                        '-eq_estrat {} -env_atts {}'.format(\
                        args.eq_estrat, args.env_att))
        add.extend([args.eq_estrat, args.env_att])
        parameter_cols.extend(['Eq_Estrat', 'Envs'])


    if args.inc_hyperparams == 1:
        command_str = cmd_append(command_str, \
                        '-inc_hyperparams {} -val_split {}'.format(\
                        args.inc_hyperparams, args.val_split))


        if args.algo in ['irm', 'linear-irm']:
            command_str = cmd_append(command_str, \
                          ('-irm_lr {lr}' + \
                            ' -irm_niter {niter}' ' -irm_l2 {l2}' + \
                            ' -irm_penalty_anneal {n_ann}' + \
                            ' -irm_penalty_weight {pen_weight}' + \
                            ' -irm_hid_layers {hid}').format(
                            lr=args.irm_lr,
                            niter=args.irm_niter,
                            l2=args.irm_l2,
                            n_ann=args.irm_penalty_anneal,
                            pen_weight=args.irm_penalty_weight,
                            hid=args.irm_hid_layers)
                          )
            add.extend([args.irm_lr, args.irm_niter, args.irm_l2, \
                        args.irm_penalty_anneal, \
                        args.irm_penalty_weight, args.irm_hid_layers])
            parameter_cols.extend(['LR', 'N_Iterations', 'L2_WeightPen', \
                                   'N_AnnealIter', 'PenWeight', 'HidLayers'])

        elif args.algo in ['linreg', 'logreg']:
            #Get paramnames
            if args.algo == 'linreg':
                reg_pname = '-linreg_lambda'
                reg_val = args.linreg_lambda
            elif args.algo == 'logreg':
                reg_pname = '-logreg_c'
                reg_val = args.logreg_c

            command_str = cmd_append(command_str, '{} {}'.format(\
                                                  reg_pname, reg_val))
            add.extend([reg_val])
            parameter_cols.extend(['Reg'])

        elif args.algo in ['mlp']:
            command_str = cmd_append(command_str, \
                          ('-mlp_lr {lr}' + \
                          ' -mlp_niter {niter}' ' -mlp_l2 {l2}' + \
                          ' -mlp_hid_layers {hid}').format(
                          lr=args.mlp_lr,
                          niter=args.mlp_niter,
                          l2=args.mlp_l2,
                          hid=args.mlp_hid_layers)
                          )
            add.extend([args.mlp_lr, args.mlp_niter, args.mlp_l2, \
                        args.mlp_hid_layers])
            parameter_cols.extend(['LR', 'N_Iterations', 'L2_WeightPen', \
                                   'HidLayers'])

        elif args.algo in ['constant']:
            pass
        else:
            raise Exception('Algorithm not implemented')

    #Write Exp Command to commandfile
    if args.algo != 'constant':
        with open(args.cmdfile, 'a') as f:
            f.write(command_str)

    #Save parameters in paramdf
    add = pd.DataFrame(add).T
    try:
        print('hi')
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
