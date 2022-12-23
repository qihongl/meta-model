import pickle
import pandas as pd
from utils.io import split_video_id

EVENT_LABEL_FPATH = '../data/high_level_events/event_annotation_timing_average.csv'

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


    def get_evtimes(self, event_id, event_num):
        '''
        input: a string in the form of '6.1.4_kinect_sep_09.pkl'
        output: a tuple - (ev start times, ev end times)
        '''
        sub_df = self.get_evs(event_id)
        event_df = sub_df.loc[sub_df['evnum'] == event_num]
        return (float(event_df['startsec']), float(event_df['endsec']))

if __name__ == "__main__":
    '''how to use'''

    evlab = EventLabel()

    event_id = '1.1.1'
    sub_df = evlab.get_evs(event_id)
    sub_df

    evlab.get_all_evnums(event_id)

    (t_start, t_end) = evlab.get_evtimes(event_id, 1)
    (t_start, t_end)
