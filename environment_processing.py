import itertools
import pandas as pd

def get_environments(df, e):
    '''Compute values of df satisfying each environment in e

    :param df: Pandas df of dataset without labels
    :param e: Dictionary of {base_cat:[all assoc df columns]} for all speicfied
              environments. Excludes columns of transformed features
    :return store: Dict of {env:e_in_values}
    :return val_store: Dict of {env:e_in_values} for the validation envs
    '''

    store = {}
    for env in itertools.product(*[e[cat] for cat in e]):
        #Get the stratification columns associated with env
        dummy_atts = []
        live_atts = []
        for att in env:
            if '_DUMmY' in att:
                dummy_atts = [a for a in e[att.split('_')[0]] if '_DUMmY' not in a]
            else:
                live_atts.append(att)

        #Compute e_in
        if not dummy_atts:
            e_in = ((df[live_atts] == 1)).all(1)
        elif not live_atts:
            e_in = ((df[dummy_atts] == 0)).all(1)
        else:
            e_in = ((df[live_atts] == 1).all(1) & (df[dummy_atts] == 0).all(1))

        store[env] = e_in
    return store

# def split_env_train_val(d, y, e_ins, val_ein):
#     '''Split data, y_all pandas dfs into train/val based on environments
#     param d: Dataset (pd df)
#     param y: labels (pd df)
#     param: e_ins: dict of e_ins for all training envs {ename:pd series}
#     param: val_ein: validation e_in (pd series)
#     return: new data, new lables
#     '''
#     pass
#
#
# def split_invar_train_val(mod, ad, ay, atts, val):
#     e_store, val_e = eproc.get_environments(ad, atts, val=val)
#     if val_e is not None:
#         train_data, train_y_all = mod.get_traindata(ad, ay, val_e)
#         val_data, val_y_all = mod.get_valdata(ad, ay, val_e)
#     else:
#         assert val_e is None
#         train_data, train_y_all = ad, ay
#         val_data, val_y_all = pd.DataFrame(), pd.DataFrame()
#
#     return train_data, train_y_all, val_data, val_y_all
