import argparse
import csv
import pickle
import os

def aggregate_results(dname):
    '''Combine the associated accepted subsets and pcps into a common
    json file '''

    with open('agg_results.txt', 'w+') as f:
        wr = csv.writer(f)

        for fname in os.listdir(dname):
            result = []
            if fname.endswith('.txt') and (fname != 'agg_results.txt'):
                result.append(fname.split('_')[0])  #alpha
                result.append(fname.split('_')[2])  #r_type

                for i in range(3, len(fname.split('_'))):
                    result.append(fname.split('_')[i].replace('.txt', ''))



                result.append(pickle.load(open(os.path.join(dname, fname), 'rb')))
                wr.writerow(result)







if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filename Parameters')
    parser.add_argument("expdir", type=str, help="dirname of results")
    args = parser.parse_args()

    aggregate_results(args.expdir)

    #aggregate_results('1579749236')