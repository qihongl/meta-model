import os
import glob
import pickle
import torch

CKPT_FTEMP = 'ckpt-ep-%d.pt'

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



def save_ckpt(cur_epoch, log_path, agent, optimizer, verbose=False):
    # compute fname
    ckpt_fname = CKPT_FTEMP % cur_epoch
    log_fpath = os.path.join(log_path, ckpt_fname)
    torch.save({
        'network_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, log_fpath)
    if verbose:
        print(f'model saved at epoch {cur_epoch}')


def load_ckpt(epoch_load, log_path, agent, optimizer=None):
    # compute fname
    ckpt_fname = CKPT_FTEMP % epoch_load
    log_fpath = os.path.join(log_path, ckpt_fname)
    if os.path.exists(log_fpath):
        # load the ckpt back
        checkpoint = torch.load(log_fpath)
        # unpack results
        agent.load_state_dict(checkpoint['network_state_dict'])
        if optimizer is None:
            optimizer = torch.optim.Adam(agent.parameters())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.train()
        # msg
        print(f'network weights - epoch {epoch_load} loaded')
        return agent, optimizer
    print('ERROR: ckpt DNE')
    return None, None


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
