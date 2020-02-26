'''Given a set of parameters (python func args), generates cmdfile with 
commands with all combinations of argument inputs '''

import argparse
import itertools
import os
import utils

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
    parser.add_argument('alpha', type=float, \
                        help='c-interval val')
    parser.add_argument('feat_eng', type=str, \
                        help='c-interval val')
    parser.add_argument('datafname', type=str, \
                        help='data.csv')
    parser.add_argument('expdir', type=str, \
                        help='where dir is')
    parser.add_argument("cmdfile", type=str, \
                       help="filename to write all commands")
    parser.add_argument("env_list", nargs='+', \
                        help="All the environment variables to split on")

    parser.add_argument("-envcombos", type=str, required=True)
    parser.add_argument("--testing", action='store_true')

    args = parser.parse_args()

    if args.testing:
        print("alpha:", args.alpha)
        print("feat_eng:", args.feat_eng)
        print("expdir:", args.expdir)
        print("data:", args.datafname)
        print("cmdfile:", args.cmdfile)
        print("env_list:", args.env_list)
        print("envcombo?:", args.envcombos)
        print("testing?:", args.testing)
        quit()

    if args.envcombos == 'all_combos':
        allenvs = utils.powerset(args.env_list)
    elif args.envcombos == 'single':
        allenvs = [[a] for a in args.env_list]

    with open(os.path.join(args.expdir, args.cmdfile), 'a') as f:
        for e in allenvs:
            uniqueid = '''{alpha}_{feat_eng}_{data}_{env_list}'''
            uniqueid= uniqueid.format(
                alpha=args.alpha,
                feat_eng=args.feat_eng,
                data=((args.datafname).split('/')[-1]).split('.')[0],  #Note - make all entries camelcase
                env_list=list_2_string(e, '--')
            )
            subsets_fname = 'subsets_{}.txt'.format(uniqueid)
            features_fname = 'feats_{}.txt'.format(uniqueid)
            rawres_fname = 'rawres_{}.json'.format(uniqueid)
            log_fname = 'log_{}.txt'.format(uniqueid)


            if args.testing:
                print(uniqueid)
                print(((args.datafname).split('/')[-1]))
                quit()


            spacing = len(list_2_string(e, ' '))

            command_str = \
                '''python main.py {alpha} {feat_eng} {data} {subsets_fname} {features_fname} {rawres_fname} {log_fname}{spacing}{env_list}\n'''
            command_str = command_str.format(
                alpha=args.alpha,
                feat_eng=args.feat_eng,
                data=args.datafname,
                subsets_fname=subsets_fname,
                features_fname=features_fname,
                rawres_fname=rawres_fname,
                log_fname=log_fname,
                spacing=(' ' * threshold(len(list_2_string(e, ' ')))),
                env_list=list_2_string(e, ' ')
            )


            f.write(command_str)








    # with open(os.path.join(args.expdir, args.cmdfile), 'w+') as cmdf:
    #     command_str = \
    #         '''
    #         python {script}
    #           --logdir {logdir}
    #         '''
    #
    #     print(command_str.format(
    #         script='my_script.py',
    #         logdir=random.random()
    #     ))


    # parser = argparse.ArgumentParser(description='Params')
    # parser.add_argument('alpha', type=float, \
    #                     help='c-interval val')
    # parser.add_argument('feat_eng', type=str, \
    #                     help='c-interval val')
    # parser.add_argument('datafname', type=str, \
    #                     help='adult.csv')
    # parser.add_argument('subsetfname', type=str, \
    #                     help='fname to write subsets')
    # parser.add_argument('featurefname', type=str, \
    #                     help='fname to write features')
    # parser.add_argument("cmdfile", type=str, \
    #                    help="filename to write all commands")
    # parser.add_argument('log_fname', type=str, \
    #                     help='fname to write log')
    # parser.add_argument("p_list", nargs='+', \
    #                     help="All the environment variables to split on")
    #
    #
    # args = parser.parse_args()
    #
    # with open(args.cmdfile, 'w+') as f:
    #     for combolength in range(0, len(args.p_list) + 1):
    #         for subset in itertools.combinations(args.p_list, combolength):
    #             #First set up the env attribute args
    #             e_args = ' '
    #             for elem in subset:
    #                 e_args = e_args + str(elem) + ' '
    #             e_args = e_args[:-1]  # CANNOT be trailing ws for xargs
    #
    #             #Now set up the proper result filenames for different env_vars
    #             new_subsetfname = args.subsetfname.split('.txt')[0] + \
    #                               e_args.replace(' ', '_') \
    #                               + '.txt'
    #
    #             new_featurefname = args.featurefname.split('.txt')[0] + \
    #                               e_args.replace(' ', '_') \
    #                               + '.txt'
    #
    #
    #             cmd = 'python main.py ' + ' ' \
    #                   + str(args.alpha) + ' ' \
    #                   + str(args.feat_eng) + ' ' \
    #                   + args.datafname + ' ' \
    #                   + new_subsetfname + ' ' \
    #                   + new_featurefname + ' ' \
    #                   + args.log_fname + ' ' \
    #                   + e_args + '\n'  #Note - no space ever allowed before \n
    #
    #             #Deal with optiona arguments
    #             # if args.log_fname is not None:
    #             #     cmd = cmd[:-1] + ' --log_fname' + ' ' + args.log_fname + '\n'
    #
    #             f.write(cmd)
    #     f.close()


        # #Consturct + write each command
    # with open(args.cmdfile, 'w+') as f:
    #     for combolength in range(0, len(args.p_list) + 1):
    #         #First set up the env attribute args
    #         e_args = ' '
    #         for elem in args.p_list:
    #             e_args = e_args + str(elem) + ' '
    #         e_args = e_args[:-1]  # CANNOT be trailing ws for xargs
    #
    #         #Now set up the proper result filenames
    #         new_subsetfname = args.subsetfname.split('.txt')[0] + \
    #                           e_args.replace(' ', '_') \
    #                           + '.txt'
    #
    #         new_featurefname = args.featurefname.split('.txt')[0] + \
    #                           e_args.replace(' ', '_') \
    #                           + '.txt'
    #         # new_featurefname = args.featurefname.split('.txt')[0] + str(
    #         #     args.p_list) \
    #         #     .replace(' ', '').replace("'", '').replace(",", "_").replace(
    #         #     '[', '_') \
    #         #     .replace(']', '_') + '.txt'
    #
    #
    #         #Now set up the rest
    #         for subset in itertools.combinations(args.p_list, combolength):
    #             cmd = 'python main.py ' + ' ' \
    #                   + str(args.alpha) + ' ' \
    #                   + args.datafname + ' ' \
    #                   + new_subsetfname + ' ' \
    #                   + new_featurefname + ' ' \
    #                   + e_args \
    #                   + '\n'
    #
    #
    #
    #             f.write(cmd)
    #     f.close()