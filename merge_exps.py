import argparse
import shutil
import os
from os.path import join
import pandas as pd

def merge_exps(newdir, e1, e2):
    '''Given paths to two unaggregated folders, merge them into one
    Assumes indexing for both dataframes starts at 0
    Assumes that unid is algo_id_....
    Assumes causal_discovery dir already made in e1-2
    '''

    #Assert experiments compatible
    e1_algo = [a.split('_')[0] for a in os.listdir(e1) if 'paramfile' in a]
    e2_algo = [a.split('_')[0] for a in os.listdir(e2) if 'paramfile' in a]
    assert len(e1_algo) == 1; assert len(e2_algo) == 1;
    assert e1_algo[0] == e2_algo[0]
    new_algo = e1_algo[0]

    #Make new folder
    if os.path.exists(newdir):
        shutil.rmtree(newdir)
    os.mkdir(newdir)
    new_cmdfile = join(newdir, 'cmdfile.sh')
    e1_paramfile = join(e1, '{}_paramfile.pkl'.format(new_algo))
    e2_paramfile = join(e2, '{}_paramfile.pkl'.format(new_algo))
    new_paramfile = join(newdir, '{}_paramfile.pkl'.format(new_algo))
    new_resdir = join(newdir, 'causal_discovery')
    os.mkdir(new_resdir)

    #Join the cmdfiles
    with open(new_cmdfile, 'wb') as dest:
        shutil.copyfileobj(open(join(e1, 'cmdfile.sh'), 'rb'), dest)
        shutil.copyfileobj(open(join(e2, 'cmdfile.sh'), 'rb'), dest)

    #renumber the indicies and join paramfiles
    e1_params = pd.read_pickle(e1_paramfile)
    e2_params = pd.read_pickle(e2_paramfile)
    new_params = e1_params.append(e2_params, ignore_index=True)
    new_params.index.name = 'Id'
    new_params.index = new_params.index.map(str)
    new_params.to_pickle(join(newdir, new_paramfile))

    #Renumber the files
    lastnum = int(e1_params.index.to_list()[-1])

    #Get the types of files
    uniq_ftypes = set()
    for f in os.listdir(join(e1, 'causal_discovery')):
         uniq_ftypes.add(f.split('_')[0])

    #Copy e1 files into newdir
    for f in os.listdir(join(e1, 'causal_discovery')):
        shutil.copy(join(join(e1, 'causal_discovery'), f), \
                    join(join(newdir, 'causal_discovery'), f))

    #Copy e2 modded files into newdir
    for ftype in uniq_ftypes:
        for f in os.listdir(join(e2, 'causal_discovery')):
            new_f = f.split('_')
            if ftype == new_f[0]:
                new_f[1] = str(lastnum + 1 + int(new_f[1]))
                new_f = '_'.join(new_f)
                shutil.copy(join(join(e2, 'causal_discovery'), f), \
                            join(join(newdir, 'causal_discovery'), new_f))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument("e1_dir", type=str)
    parser.add_argument("e2_dir", type=str)
    parser.add_argument("newdir", type=str)
    args = parser.parse_args()
    merge_exps(args.newdir, args.e1_dir, args.e2_dir)
