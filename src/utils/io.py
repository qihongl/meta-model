import os
import glob
import pickle
import torch
# import numpy as np

CKPT_FTEMP = 'ckpt-ep-%d.pt'
SC_FTEMP = 'sc-ep-%d.pkl'

def pickle_load(data_path):
    loaded_obj = pickle.load(open(data_path, 'rb'))
    return loaded_obj

def pickle_save(input_obj, save_path):
    with open(save_path, 'wb') as handle:
        pickle.dump(input_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_ckpt(cur_epoch, log_path, agent, optimizer, context_module_dict, verbose=False):
    # compute fname
    ckpt_fname = CKPT_FTEMP % cur_epoch
    sc_fname = SC_FTEMP % cur_epoch
    torch.save(
        {
            'network_state_dict': agent.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        },
        os.path.join(log_path, ckpt_fname)
    )
    pickle_save(context_module_dict, os.path.join(log_path, sc_fname))
    if verbose:
        print(f'model saved at epoch {cur_epoch}')


def load_ckpt(epoch_load, log_path, agent, optimizer, verbose=True):
    # compute fname
    ckpt_fname = CKPT_FTEMP % epoch_load
    sc_fname = SC_FTEMP % epoch_load
    ckpt_fpath = os.path.join(log_path, ckpt_fname)
    sc_fpath = os.path.join(log_path, sc_fname)
    if os.path.exists(ckpt_fpath) and os.path.exists(sc_fpath):
        # load the ckpt back
        checkpoint = torch.load(ckpt_fpath)
        # unpack results
        agent.load_state_dict(checkpoint['network_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.train()
        context_module_dict = pickle_load(sc_fpath)
        # msg
        if verbose:
            print(f'{ckpt_fname} loaded')
            print(f'{sc_fname} loaded')
        return agent, optimizer, context_module_dict
    print('ERROR: ckpt DNE')
    return None, None, None


def list_fnames(data_dir, fpattern, verbose=False):
    '''
    list all fnames/fpaths with a particular fpattern (e.g. *pca.pkl)
    '''
    fpaths = glob.glob(os.path.join(data_dir, fpattern))
    n_data_files = len(fpaths)
    fnames = [None] * n_data_files
    for i, fpath in enumerate(fpaths):
        # get file info
        fnames[i] = os.path.basename(fpath)
    if verbose:
        print(fnames)
    return fpaths, fnames


# if __name__ == "__main__":
#     # training param
