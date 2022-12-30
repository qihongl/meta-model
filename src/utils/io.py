import os
import glob
import pickle


def pickle_load(data_path):
    df = pickle.load(open(data_path, 'rb'))
    return df


def pickle_save_dict(input_dict, save_path):
    """Save the dictionary

    Parameters
    ----------
    input_dict : type
        Description of parameter `input_dict`.
    save_path : type
        Description of parameter `save_path`.

    """
    with open(save_path, 'wb') as handle:
        pickle.dump(input_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load_dict(fpath):
    """load the dict

    Parameters
    ----------
    fpath : type
        Description of parameter `fpath`.

    Returns
    -------
    type
        Description of returned object.

    """
    return pickle.load(open(fpath, "rb"))


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
