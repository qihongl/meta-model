import pickle
import numpy as np
import pandas as pd

EVENT_LABEL_FPATH = '../data/high_level_events/event_annotation_timing_average.csv'
# FPS = 25

class EventLabel:
    '''event labeler - run this from src '''

    def __init__(self):
        self.df = pd.read_csv(EVENT_LABEL_FPATH)
        self.all_subev_names = np.unique(self.df['evname'])
        self.n_subev_names = len(self.all_subev_names)

    def get_subdf(self, event_id):
        '''
        input: a string in the form of '6.1.4.pkl'
        output: the corresponding evnum
        '''
        return self.df.loc[self.df['run'] == event_id]

    def get_all_subev_nums(self, event_id):
        '''
        input: a string in the form of '6.1.4_kinect_sep_09.pkl'
        output: a list of all event numbers
        '''
        sub_df = self.get_subdf(event_id)
        return list(sub_df['evnum'])

    def get_subev_num(self, evname):
        return list(self.all_subev_names).index(evname)

    def get_bounds(self, event_id, to_sec=False):
        sub_df = self.get_subdf(event_id)
        event_boundary_times = np.array(
            list(sub_df['startsec']) + [list(sub_df['endsec'])[-1]]
        )
        # whether to keep the sec
        if not to_sec:
            event_boundary_times = event_boundary_times * 3
        # get k-hot vector of event boundary
        event_boundary_vector = vectorize_event_bounds(event_boundary_times)
        return event_boundary_times, event_boundary_vector

    def get_subev_times(self, event_id, event_num, to_sec=False):
        '''
        input: a string in the form of '6.1.4_kinect_sep_09.pkl'
        output: a tuple - (ev start times, ev end times)
        '''
        sub_df = self.get_subdf(event_id)
        event_df = sub_df.loc[sub_df['evnum'] == event_num]
        t_start, t_end = (float(event_df['startsec']), float(event_df['endsec']))
        # if to_sec:
        return t_start, t_end
        # return round(t_start / FPS), round(t_end / FPS)

    def get_start_end_times(self, event_id):
        event_bound, _ = self.get_bounds(event_id)
        return event_bound[0], event_bound[-1]


    def get_subev_labels(self, event_id, to_sec=True):
        # get event label info for event i in secs
        df_i = self.get_subdf(event_id)
        event_i_len = int(np.round(df_i['endsec'].iloc[-1]))
        if not to_sec:
            event_i_len *=3
        sub_ev_label_i = np.full(event_i_len, np.nan)
        for evname, evnum, t_start, t_end in zip(df_i['evname'], df_i['evnum'], df_i['startsec'], df_i['endsec']):
            t_start, t_end = int(t_start), int(t_end)
            if not to_sec:
                t_start, t_end = t_start * 3 , t_end * 3
            # the +1 here ensure there is no gap for the sub event label when round to sec
            sub_ev_label_i[t_start:t_end+1] = self.get_subev_num(evname)
        return sub_ev_label_i

def vectorize_event_bounds(event_bonds):
    T = int(max(np.round(event_bonds))) + 1
    event_bounds_vec = np.zeros(T)
    for evb in event_bonds:
        event_bounds_vec[int(np.round(evb))] = 1
    return event_bounds_vec


if __name__ == "__main__":
    '''how to use'''

    evlab = EventLabel()



    event_id = '1.1.1'
    sub_df = evlab.get_subdf(event_id)
    sub_df['startsec']
    print(evlab.all_subev_names)
    print(evlab.n_subev_names)


    evnums = evlab.get_all_subev_nums(event_id)

    (t_start, t_end) = evlab.get_subev_times(event_id, 1)
    print(t_start, t_end)

    event_bound, event_bound_vec = evlab.get_bounds(event_id)
    subev_labels = evlab.get_subev_labels(event_id, to_sec=False)

    print(event_bound)
    print(subev_labels)
