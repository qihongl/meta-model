import pickle
import numpy as np
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
        hb_i = self.get_subdf(event_id, condition)
        return len(np.unique(hb_i['workerId']))

    def get_workers(self, event_id, condition=None):
        hb_i = self.get_subdf(event_id, condition)
        return list(np.unique(hb_i['workerId']))

    def get_bound_times(self, event_id, condition=None, worker_id=None, to_sec=True):
        hb_i = self.get_subdf(event_id, condition)
        if worker_id is None:
            bound_times = np.array(list(hb_i['Sec']))
        else:
            bound_times = np.array(sorted(hb_i[hb_i['workerId'] == worker_id]))
        if to_sec:
            bound_times *= 3
        return bound_times


    def get_bound_prob(self, event_id, condition=None, to_sec=False, cap_at_one=True):
        boundary_times = self.get_bound_times(event_id, condition)
        if not to_sec:
            boundary_times = boundary_times * 3
        boundary_time_rounded = np.round(boundary_times)
        #
        T = int(max(boundary_time_rounded))
        n = self.get_n_workers(event_id, condition)
        prob = np.array([np.sum(boundary_time_rounded == t) / n for t in range(T)])
        if cap_at_one:
            prob[prob >= 1] = 1
        return prob


if __name__ == "__main__":
    '''how to use'''
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(style='white', palette='colorblind', context='poster')

    hb = HumanBondaries()

    event_id  = '1.1.3'
    condition = 'coarse'

    subdf = hb.get_subdf(event_id)
    subdf.head()

    boundary_times = hb.get_bound_times(event_id, condition)

    # condition = 'fine'
    alpha = .7
    p_b_c = hb.get_bound_prob(event_id, 'coarse')
    p_b_f = hb.get_bound_prob(event_id, 'fine')
    f, ax = plt.subplots(1,1, figsize=(14, 4))
    ax.plot(p_b_c, label='coarse', alpha=alpha)
    ax.plot(p_b_f, label='fine', alpha=alpha)
    ax.set_title(f'event id: {event_id} - {condition}')
    ax.set_xlabel('Time')
    ax.set_ylabel('boundary probability')
    ax.legend()
    sns.despine()
