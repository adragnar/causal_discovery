'''Given a set of parameters (python func args), generates cmdfile with 
commands with all combinations of argument inputs '''

import argparse
import itertools


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('alpha', type=float, \
                        help='c-interval val')
    parser.add_argument('datafname', type=str, \
                        help='adult.csv')
    parser.add_argument('subsetfname', type=str, \
                        help='fname to write subsets')
    parser.add_argument('featurefname', type=str, \
                        help='fname to write features')
    parser.add_argument("cmdfile", type=str, \
                       help="filename to write all commands")
    parser.add_argument("p_list", nargs='+', \
                        help="All the environment variables to split on")


    args = parser.parse_args()

    with open(args.cmdfile, 'w+') as f:
        for combolength in range(0, len(args.p_list) + 1):
            for subset in itertools.combinations(args.p_list, combolength):
                #First set up the env attribute args
                e_args = ' '
                for elem in subset:
                    e_args = e_args + str(elem) + ' '
                e_args = e_args[:-1]  # CANNOT be trailing ws for xargs

                #Now set up the proper result filenames
                new_subsetfname = args.subsetfname.split('.txt')[0] + \
                                  e_args.replace(' ', '_') \
                                  + '.txt'

                new_featurefname = args.featurefname.split('.txt')[0] + \
                                  e_args.replace(' ', '_') \
                                  + '.txt'


                cmd = 'python main.py ' + ' ' \
                      + str(args.alpha) + ' ' \
                      + args.datafname + ' ' \
                      + new_subsetfname + ' ' \
                      + new_featurefname + ' ' \
                      + e_args + '\n'

                f.write(cmd)
        f.close()


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