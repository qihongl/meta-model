import pickle
import numpy as np
import pandas as pd

EVENT_LABEL_FPATH = '../data/high_level_events/event_annotation_timing_average.csv'
# FPS = 25

class EventLabel:
    '''event labeler - run this from src '''

    def __init__(self):
        self.df = pd.read_csv(EVENT_LABEL_FPATH)

    def get_evs(self, event_id):
        '''
        input: a string in the form of '6.1.4_kinect_sep_09.pkl'
        output: the corresponding evnum
        '''
        return self.df.loc[self.df['run'] == event_id]

    def get_all_evnums(self, event_id):
        '''
        input: a string in the form of '6.1.4_kinect_sep_09.pkl'
        output: a list of all event numbers
        '''
        sub_df = self.get_evs(event_id)
        return list(sub_df['evnum'])

    def get_bounds(self, event_id, to_sec=False):
        sub_df = self.get_evs(event_id)
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
        sub_df = self.get_evs(event_id)
        event_df = sub_df.loc[sub_df['evnum'] == event_num]
        t_start, t_end = (float(event_df['startsec']), float(event_df['endsec']))
        # if to_sec:
        return t_start, t_end
        # return round(t_start / FPS), round(t_end / FPS)


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
    sub_df = evlab.get_evs(event_id)
    sub_df['startsec']


    evlab.get_all_evnums(event_id)

    (t_start, t_end) = evlab.get_subev_times(event_id, 1)
    print(t_start, t_end)

    event_bound, event_bound_vec = evlab.get_bounds(event_id)



    print(event_bonds)
