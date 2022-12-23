import os
import glob
import pickle


# ID2CHAPTER = bidict({
#     '1' : 'breakfast',
#     '2' : 'exercising',
#     '3' : 'cleaning',
#     '4' : 'bathroom'
# })
#

def split_video_id(video_id, to_int=False):
    actor_id, chapter_id, run_id = video_id.split('.')
    if to_int:
        return int(actor_id), int(chapter_id), int(run_id)
    return actor_id, chapter_id, run_id

def pickle_load(data_path):
    df = pickle.load(open(data_path, 'rb'))
    return df

def list_fnames(data_dir, fpattern):
    '''
    list all fnames/fpaths with a particular fpattern (e.g. *pca.pkl)
    '''
    fpaths = glob.glob(os.path.join(data_dir, fpattern))
    n_data_files = len(fpaths)
    fnames = [None] * n_data_files
    for i, fpath in enumerate(fpaths):
        # get file info
        fnames[i] = os.path.basename(fpath)
    return fpaths, fnames
