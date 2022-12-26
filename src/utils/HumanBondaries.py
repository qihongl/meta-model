import pickle
import pandas as pd

HUMAN_BOUNDARIES_FPATH = '../data/high_level_events/seg_data_analysis_clean.csv'
MOVIE_POST_STR = '_C1_trim'
CONDITIONS = ['coarse', 'fine']

class HumanBondaries:
    '''event labeler - run this from src '''

    def __init__(self):
        self.df = pd.read_csv(HUMAN_BOUNDARIES_FPATH)

    def get_subdf(self, event_id, condition=None):
        assert condition in CONDITIONS or condition is None
        sub_df = self.df.loc[self.df['Movie'] == event_id + MOVIE_POST_STR]
        if condition in CONDITIONS:
            return sub_df[sub_df['Condition'] == condition]
        return sub_df

    def get_n_workers(self, event_id, condition=None):
        hb_i = hb.get_subdf(event_id, condition)
        return len(np.unique(hb_i['workerId']))

    def get_workers(self, event_id, condition=None):
        hb_i = hb.get_subdf(event_id, condition)
        return list(np.unique(hb_i['workerId']))

    def get_bond_times(self, event_id, condition=None, worker_id=None):
        hb_i = hb.get_subdf(event_id, condition)
        if worker_id is None:
            return list(hb_i['Sec'])
        return sorted(hb_i[hb_i['workerId'] == worker_id])

    # def get_bond_times(self, event_id, condition=None, worker_id=None):
    #     hbts = self._get_bond_times(event_id, condition, worker_id)
        # for
        # hbts


if __name__ == "__main__":
    '''how to use'''
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(style='white', palette='colorblind', context='poster')

    hb = HumanBondaries()

    event_id, condition = '1.1.1', 'coarse'


    workers_i = hb.get_workers(event_id, condition)
    print(workers_i)

    bts = hb.get_bond_times(event_id, condition)
    print(bts)
    np.round(bts)

    # f, ax = plt.subplots(1,1, figsize=(20, 5))
    # ax.stem(bts)
    # sns.despine()
