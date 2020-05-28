from itertools import chain, combinations
import torch

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

def dname_from_fpath(fpath):
    '''Return name of dataset from its filepath'''
    if 'adult' in fpath:
        return 'adult'
    if 'german' in fpath:
        return 'germanCredit'

    raise Exception('Dataset Not Included')

def proc_fteng(ft):
    '''Convert the fteng string into appropiate list of modifications to make'''
    if ft == '-1':
        return []
    else:
        return [int(c) for c in ft]

def env_parser(envs):
    '''Convert a string of a list of strings without spaces into a list'''
    ret = envs.strip('[').strip(']').split(',')
    if (len(ret) == 1) and ('' in ret):
        return []
    return ret

def make_tensor(arr):
    '''Convert np array into a float tensor'''
    return torch.from_numpy(arr).float()

def merge_exps(newdir, e1, e2):
    # '''Given paths to two unaggregated folders, merge them into one'''
    # #Make new folder
    # # join()
    #
    # #renumber the indicies and join paramfiles
    # e1_params = pd.read_pickle(e1_paramfile)
    # e2_params = pd.read_pickle(e2_paramfile)
    # new_params = e1_params.append(e2_params, ignore_index=True)
    # new_params.index.name = 'Id'
    # new_params.index = new_params.index.map(str)
    #
    # #Renumber the files
    # lastnum = int(e1_params.index.to_list()[-1])
    # for

    pass
