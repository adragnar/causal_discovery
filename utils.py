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
    '''Given paths to two fully formatted folders, merge them into one'''
    #Make new folder
    # join()
    pass
