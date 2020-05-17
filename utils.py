from itertools import chain, combinations


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

def pretty(vector):
    vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.4f}".format(vi) for vi in vlist) + "]"
