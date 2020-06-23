import itertools

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
                dummy_atts = [a for a in e[att.split('_')[0]] \
                              if '_DUMmY' not in a]
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
