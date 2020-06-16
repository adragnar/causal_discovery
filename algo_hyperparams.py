from utils import dname_from_fpath

def get_irm_args(dset_fname):
    if dname_from_fpath(dset_fname) == 'adult':
        args =  {'lr': 0.01, \
                 'n_iterations':3000, \
                 'penalty_anneal_iters':500, \
                 'l2_reg':0.0001, \
                 'pen_wgt':3000, \
                 'hid_layers':200, \
                 'verbose':True}

    elif dname_from_fpath(dset_fname) == 'german':
        args =  {'lr':0.0001, \
                 'n_iterations':2000, \
                 'penalty_anneal_iters':100, \
                 'l2_reg':0.001, \
                 'pen_wgt':3000, \
                 'hid_layers':200, \
                 'verbose':True}

    else:
        assert False

    return args

def get_linear_irm_args(dset_fname):
    args =  {'lr': 0.01, \
             'n_iterations':1000, \
             'penalty_anneal_iters':1, \
             'l2_reg':0.0, \
             'pen_wgt':1000, \
             'hid_layers':100, \
             'verbose':True}
    return args

def get_mlp_args(dset_fname):
    args =  {'lr': 0.01, \
             'n_iterations':1000, \
             'l2_reg':0.0, \
             'hid_layers':100}

    return args

def get_linreg_args(dset_fname):
    args =  {'lambda':1e-5}
    return args

def get_logreg_args(dset_fname):
    args =  {'C':0.001}
    return args
