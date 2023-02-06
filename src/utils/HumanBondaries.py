import os
import pickle
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from utils.stats import get_point_biserial
from utils import pickle_load

HUMAN_BOUNDARIES_FPATH = '../data/high_level_events/seg_data_analysis_clean.csv'
PRECOMPUTED_DIR = '../data/precomputed_hb'
MOVIE_POST_STR = '_C1_trim'
CONDITIONS = ['coarse', 'fine']



class HumanBondaries:
    '''event labeler - run this from src '''

    def __init__(self):
        self.df = pd.read_csv(HUMAN_BOUNDARIES_FPATH)

    def get_subdf(self, event_id, condition):
        assert condition in CONDITIONS
        sub_df = self.df.loc[self.df['Movie'] == event_id + MOVIE_POST_STR]
        # if condition in CONDITIONS:
        return sub_df[sub_df['Condition'] == condition]
        # return sub_df

    def get_workers(self, sub_df):
        worker_ids = np.unique(sub_df['workerId'])
        return list(worker_ids), len(worker_ids)

    def get_bound_times(self, sub_df, worker_id=None, to_sec=False):
        if worker_id is not None:
            return self._get_bound_times_1sub(sub_df, worker_id, to_sec)
        # return for all subjects
        bound_times = np.array(list(sub_df['Sec']))
        worker_ids, n = self.get_workers(sub_df)
        bound_times = [self._get_bound_times_1sub(sub_df, worker_id, to_sec) for worker_id in worker_ids]
        return bound_times

    def _get_bound_times_1sub(self, sub_df, worker_id, to_sec=True):
        bound_times = np.array(sorted(sub_df[sub_df['workerId'] == worker_id]['Sec']))
        if not to_sec:
            bound_times *= 3
        return bound_times

    def get_bound_prob(self, event_id, condition, to_sec=False, cap_at_one=True):
        sub_df = self.get_subdf(event_id, condition)
        worker_ids, n = self.get_workers(sub_df)
        boundary_times = self.get_bound_times(sub_df, to_sec=to_sec)
        all_boundary_times_rounded = np.round(np.concatenate(boundary_times))
        T = int(np.max(all_boundary_times_rounded)) + 1
        return self._get_bound_prob(all_boundary_times_rounded, n, T, cap_at_one=cap_at_one)

    def _get_bound_prob(self, all_boundary_times_rounded, n, T, cap_at_one=True):
        prob = np.array([np.sum(all_boundary_times_rounded == t) / n for t in range(T)])
        prob = gaussian_filter1d(prob, 2)
        if cap_at_one:
            prob[prob >= 1] = 1
        return prob

    def get_precomputed_hb(self, condition):
        assert condition in CONDITIONS
        fname = 'hb-' + condition + '.pkl'
        fpath = os.path.join(PRECOMPUTED_DIR, fname)
        hbdict = pickle_load(fpath)
        return hbdict

    def get_human_ceiling(self, event_id, condition):
        assert condition in CONDITIONS

        subdf = self.get_subdf(event_id, condition)
        # get boundary time for all workers
        worker_ids, n = self.get_workers(subdf)
        self.get_bound_times(subdf, to_sec=False)
        bound_times = [
            self.get_bound_times(subdf, worker_id=worker_id, to_sec=False)
            for worker_id in worker_ids
        ]
        # convert to boundary vector
        T = int(np.max(np.round(np.concatenate(bound_times))))+1
        bound_vecs = np.zeros((n, T))
        for i, worker_id in enumerate(worker_ids):
            for bound_t in bound_times[i]:
                bound_vecs[i][int(np.round(bound_t))] = 1
        r = np.zeros(n,)
        for j in range(n):
            bound_times_except_1sub = [x for i, x in enumerate(bound_times) if i!=j]
            all_bound_times_except_1sub = np.round(np.concatenate(bound_times_except_1sub))
            prob_j = self._get_bound_prob(all_bound_times_except_1sub, n, T, cap_at_one=True)
            # print(len(bound_vecs[j]))
            # print(len(prob_j))
            # print()
            r[j], _ = get_point_biserial(bound_vecs[j], prob_j)
        return r

    def get_average_human_ceiling(self, event_id_list):
        human_r_crse = np.zeros(len(event_id_list), )
        human_r_fine = np.zeros(len(event_id_list), )
        for i, event_id in enumerate(event_id_list):
            human_r_crse[i] = np.mean(self.get_human_ceiling(event_id, 'coarse'))
            human_r_fine[i] = np.mean(self.get_human_ceiling(event_id, 'fine'))
        return human_r_crse, human_r_fine


if __name__ == "__main__":
    '''how to use'''
    import seaborn as sns
    import matplotlib.pyplot as plt
    from utils import TrainValidSplit
    sns.set(style='white', palette='colorblind', context='poster')


    hb = HumanBondaries()
    tvs = TrainValidSplit()

    event_id  = '1.1.3'
    condition = 'coarse'

    csubdf = hb.get_subdf(event_id, 'coarse')
    csubdf.head()

    fsubdf = hb.get_subdf(event_id, 'fine')
    fsubdf.head()

    human_r_crse, human_r_fine = hb.get_average_human_ceiling(tvs.valid_ids)

    f, axes = plt.subplots(2, 1, figsize=(5, 8), sharex=True)
    sns.violinplot(human_r_crse, orient='v', ax=axes[0])
    sns.violinplot(human_r_fine, orient='v', ax=axes[1])
    axes[1].set_xlabel('point biserial correlation')
    axes[0].set_ylabel('coarse boundaries')
    axes[1].set_ylabel('fine boundaries')
    axes[0].set_title('mean = %.2f' % (np.mean(human_r_crse)))
    axes[1].set_title('mean = %.2f' % (np.mean(human_r_fine)))
    f.tight_layout()
    sns.despine()



    # get_point_biserial()
    # plt.imshow(bound_vecs,aspect='auto')

    # worker_ids, n_workers = hb.get_workers(subdf)
    # boundary_times = hb.get_bound_times(subdf, worker_ids[0])
    #
    # boundary_times = [hb.get_bound_times(subdf, wid) for wid in worker_ids]
    # boundary_times
    # all_boundary_times_rounded = np.round(np.concatenate(boundary_times))
    # T = int(max(all_boundary_times_rounded))


#     freq, y = np.histogram(
#         boundary_times,
#         bins=np.arange(0, max(boundary_times), 1)
#         )
#     plt.plot(freq)

    # condition = 'fine'
    alpha = .7
    p_b_c = hb.get_bound_prob(event_id, 'coarse')
    p_b_f = hb.get_bound_prob(event_id, 'fine')
    f, axes = plt.subplots(2,1, figsize=(14, 7), sharex=True, sharey=True)
    axes[0].plot(p_b_c, label='coarse', alpha=alpha)
    axes[1].plot(p_b_f, label='fine', alpha=alpha)
    axes[0].set_title(f'event id: {event_id} - {condition}')
    axes[1].set_xlabel('Time')
    # axes[0].set_ylabel('boundary probability')
    axes[0].legend()
    sns.despine()

    fhb = hb.get_precomputed_hb('fine')
    chb = hb.get_precomputed_hb('coarse')


# def get_frequency_ground_truth(second_boundaries, second_interval=1,
#                                end_second=555) -> Tuple:
#     frequency, bins = np.histogram(second_boundaries,
#                                    bins=np.arange(0, end_second + second_interval,
#                                                   second_interval))
#     return frequency, bins
#
#     frequency, bins = np.histogram(
#         boundary_times,
#         bins=np.arange(0, end_second + second_interval, 1)
#         )
#
#     second_interval, end_second = 14, 1050
