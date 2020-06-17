import argparse
import shutil
import os
from os.path import join
import pandas as pd

import aggregate as agg

def format_experiments(resdir):
    '''Given experiment folders pulled raw from server, combine into new directory)'''
    def iterate_dirs(d):
        return [r for r in os.listdir(d) if (not r.startswith('.'))]
    def iterate_files(d):
        return [r for r in os.listdir(d) if ((not r.startswith('.')) and ('.' in r))]

    algo_dict = {}
    for edir in iterate_dirs(resdir):

        #Check not final results dir
        if 'results' in edir:
            shutil.rmtree(join(resdir, edir))
            continue

        expdir = join(resdir, edir)
        cd = join(expdir, 'causal_discovery')
        if not os.path.exists(cd):
            os.mkdir(cd)

        for fname in iterate_files(expdir):

            #Copy everything into relevant internal folder
            if (fname != 'cmdfile.sh') and ('paramfile' not in fname) and ('code' not in fname):
                shutil.move(join(expdir, fname), join(cd, fname))

            #Associate algos with folders
            if ('paramfile' in fname):
                a_name = fname.split('_')[0]

                #Deal with case where aggregation before, modify filenames
                true_expdir = expdir
                if join(resdir, a_name) == expdir:
                    true_expdir = '{}Tmp'.format(expdir)
                    shutil.move(expdir, true_expdir)
                try:
                    algo_dict[a_name].append(true_expdir)
                except:
                    algo_dict[a_name] = [true_expdir]

    #Merge Experiments
    old_names = []
    for a, exp_list in algo_dict.items():
        if len(exp_list) == 1:
            os.rename(exp_list[0], join(resdir, a))
        elif len(exp_list) == 2:
            merge_exps(join(resdir, a), *exp_list)
        else:
            raise Exception('Too many folders merging')
        old_names += exp_list

    #Clean up and aggregate
    import pdb; pdb.set_trace()
    for edir in iterate_dirs(resdir):
        expdir = join(resdir, edir)
        if expdir in old_names:
            shutil.rmtree(expdir)  #Delete old folders
        else:
            agg.aggregate_loader(expdir, 'data', edir)  #Aggregate remaining ones



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

    #Deal with the code foldes in both folders
    new_code_dirname = join(newdir, 'code')
    os.mkdir(new_code_dirname)
    try:
        shutil.copytree(join(e1, 'code'), join(new_code_dirname, 'code_1'))
        shutil.copytree(join(e2, 'code'), join(new_code_dirname, 'code_2'))
    except:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    # parser.add_argument("e1_dir", type=str)
    # parser.add_argument("e2_dir", type=str)
    # parser.add_argument("newdir", type=str)
    # args = parser.parse_args()
    # merge_exps(args.newdir, args.e1_dir, args.e2_dir)
    parser.add_argument("r_dir", type=str)
    args = parser.parse_args()
    format_experiments(args.r_dir)
