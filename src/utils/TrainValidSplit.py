import numpy as np

DATA_ROOT = '../data/output/'
PROBLEMATIC_TRAIN_VIDEOS = ['4.4.1']
class TrainValidSplit:

    def __init__(self, feature_tag='sep_09'):
        self.feature_tag = feature_tag
        self.train_fpath = f'{DATA_ROOT}train_{self.feature_tag}.txt'
        self.valid_fpath = f'{DATA_ROOT}valid_{self.feature_tag}.txt'

        self.train_fnames = get_all_lines(self.train_fpath)
        self.valid_fnames = get_all_lines(self.valid_fpath)
        self.all_fnames = self.train_fnames + self.valid_fnames

        self.train_ids = [event_id(fn) for fn in self.train_fnames]
        self.valid_ids = [event_id(fn) for fn in self.valid_fnames]
        self.all_ids = self.train_ids + self.valid_ids

        self.n_train_files = len(self.train_fnames)
        self.n_valid_files = len(self.valid_fnames)
        self.n_files = self.n_train_files + self.n_valid_files

    def remove_problematic_videos(self):
        for event_id in PROBLEMATIC_TRAIN_VIDEOS:
            self.n_train_ids.remove(event_id)
            self.all_ids.remove(event_id)
            self.n_train_files -=1
            self.n_files -=1

    def is_train_file(self, event_id_str):
        """
        event_id_str: str in the form of 'x.y.z'
        """
        fname = event_id_str + '_kinect'
        assert fname in self.all_fnames, 'must be a train file or a valid file'
        if fname in self.train_fnames:
            return True
        return False

    def is_valid_file(self, event_id_str):
        return not is_train_file(event_id_str)

def event_id(fname):
    return fname.split('_')[0]


def get_all_lines(txt_fname):
    txt_file = open(txt_fname, 'r')
    return [line.strip() for line in txt_file.readlines()]

if __name__ == "__main__":
    tvs = TrainValidSplit()
    print(tvs.train_fnames)
    print(tvs.train_ids)
    print(tvs.n_train_files)
    print(tvs.valid_fnames)
    print(tvs.valid_ids)
    print(tvs.n_valid_files)
    print(tvs.all_fnames)
    print(tvs.n_files)
    print(tvs.is_train_file('1.1.1'))


# '4.4.1' in tvs.train_ids
