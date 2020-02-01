
import argparse
import itertools


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('alpha', type=float, \
                        help='c-interval val')
    parser.add_argument('subsetfname', type=str, \
                        help='fname to write subsets')
    parser.add_argument('featurefname', type=str, \
                        help='fname to write features')
    parser.add_argument("cmdfile", type=str, \
                       help="filename to write all commands")
    parser.add_argument("p_list", nargs='+', \
                        help="All the environment variables to split on")


    args = parser.parse_args()
    print(args.alpha)
    print(args.subsetfname)
    print(args.cmdfile)

    with open(args.cmdfile, 'w+') as f:
        for combolength in range(0, len(args.p_list) + 1):
            for subset in itertools.combinations(args.p_list, combolength):
                cmd = 'python main.py ' + str(args.alpha) +' ('
                for elem in subset:
                    cmd = cmd + "'" +str(elem) + "' "
                cmd = cmd + ') ' + args.subsetfname + ' ' + \
                      args.featurefname + ' ' + '\n'

                f.writelines(cmd)
        f.close()



    while(True):
        pass