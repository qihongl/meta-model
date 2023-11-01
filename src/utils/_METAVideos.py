import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

from utils import to_pth
from utils.io import pickle_load
from utils import TrainValidSplit
from scipy.cluster.hierarchy import dendrogram, linkage
from utils._METAConstants import hier_struct, hlc2hlcid, mlc2mlcid, llc_names, llc2llcid, llc2mlcid, llc2hlcid

sns.set(style='white', palette='colorblind', context='talk')
cpal = sns.color_palette('colorblind')


class METAVideos:

    '''definitions
    hlc2hlcid id = the ids for the 4 chapters + multi-chapter actions
    mlc2mlcid id = the mid level (in the hier struct) id
    llc2llcid id = the low level (leafs in the hier struct) id
    instance id = there are multiple instances for each llc2llcid...

    '''

    def __init__(self, min_len=None, exlude_multi_chapter=0, verbose=1):
        self.dim = 30
        data_dict = pickle_load('../data/meta-videos.pkl')
        self.data = data_dict['video_data']
        self.data_llc = data_dict['video_data_llc']
        self.llc_mean_pattern = pickle_load('../data/meta-llc-mean-pattern.pkl')
        # constants
        self.hier_struct = hier_struct
        # xlc -> id mapping dicts
        self.llc2llcid = llc2llcid
        self.hlc2hlcid = hlc2hlcid
        self.mlc2mlcid = mlc2mlcid
        #
        self.llc2mlcid = llc2mlcid
        self.llc2hlcid = llc2hlcid
        #
        self.all_llc = llc_names
        self.all_llc_rmca = [llc_i for llc_i in llc_names if llc2hlcid[llc_i] !=0]
        self.n_llc = len(llc_names)
        self.n_llc_rmca = len(self.all_llc_rmca)
        self.n_hlc = len(hlc2hlcid)
        #
        self.hlc_cpal = sns.color_palette('colorblind', n_colors=self.n_hlc)
        self.hlc_cpal_rmca = np.roll(self.hlc_cpal, shift=1, axis=0)
        # options
        self.exlude_multi_chapter = exlude_multi_chapter
        self.min_len = min_len
        #
        self.verbose = verbose
        # pre processing
        if min_len is not None: self.remove_tiny_events()
        if exlude_multi_chapter: self.remove_multi_chapter_actions()

        # precompute useful stats
        self.llc_counts = self.get_llc_counts()
        self.llc_p = self.llc_counts / np.sum(self.llc_counts)
        self.n_ll_events_per_video = [len(v) for v in self.data_llc.values()]
        self.n_total_ll_events = sum([len(v) for v in self.data_llc.values()])
        # meta.n_total_ll_events
        self.tvs = TrainValidSplit()
        # log str
        self.log_str = f'meta/min_len={min_len}-rm_mca-{exlude_multi_chapter}'


    '''pre processing'''
    def remove_tiny_events(self):
        '''remove events that are too short'''
        print(f'REMOVE events that are shorter than {self.min_len} ...')
        counts = 0
        for k in self.data.keys():
            for i, ll_event in enumerate(self.data[k]):
                if len(ll_event) < self.min_len:
                    if self.verbose:
                        print(f'video id = {k} \t {self.data_llc[k][i]} : {np.shape(ll_event)}')
                    # pop both the label and the data if the data is too short
                    self.data[k].pop(i)
                    self.data_llc[k].pop(i)
                    counts+=1
        print(f'Done, # events removed = {counts}')

    def remove_multi_chapter_actions(self):
        counts = 0
        print('REMOVE multichapter actions ...')
        for k in self.data.keys():
            for i, llc in enumerate(self.data_llc[k]):
                if self.llc2hlcid[llc] == 0:
                    if self.verbose:
                        print(f'video id = {k} \t {llc}')
                    self.data[k].pop(i)
                    self.data_llc[k].pop(i)
                    counts+=1
        print(f'Done, # events removed = {counts}')

    '''data generator'''
    def get_data(self, to_torch=True):
        '''return a permutation of all data'''
        all_llc_events = concat_lists([video_i for video_i in self.data.values()])
        all_llc_event_names = concat_lists([video_llcs for video_llcs in self.data_llc.values()])
        perm_op = np.random.permutation(len(all_llc_events))
        all_llc_events = [all_llc_events[i] for i in perm_op]
        all_llc_event_names = [all_llc_event_names[i] for i in perm_op]
        if to_torch:
            all_llc_events = [to_pth(x) for x in all_llc_events]
        return all_llc_events, all_llc_event_names

    def get_video(self, video_id, to_torch=True):
        # video_id = '4.4.4'
        event_length = [len(event_i) for event_i in self.data[video_id]]
        llc_ids = np.concatenate([[self.llc2llcid[llc_i]] * evn_len_i
            for (llc_i, evn_len_i) in zip(self.data_llc[video_id], event_length)])
        # reformat
        X = np.concatenate(self.data[video_id])
        if to_torch: X = to_pth(X)
        return X, llc_ids

    '''utils'''

    def get_llc_counts(self):
        all_llc_events = concat_lists(list(self.data_llc.values()))
        return [all_llc_events.count(llc_i) for llc_i in self.all_llc]

    def llc_RDM(self):
        return np.corrcoef(self.llc_mean_pattern)

    def get_mid_level_cats(self):
        second_level_nodes = []
        for value in self.hier_struct.values():
            if isinstance(value, dict):  # Check if the value is a dictionary
                second_level_nodes.extend(value.keys())
        print(second_level_nodes)

    def get_low_level_cats(self):
        return get_leaves(hier_struct)

    def compute_validation_set_mean_pattern(self):
        valid_llc_ids = {self.llc2llcid[llc_ids]: 0 for llc_ids in self.all_llc}
        valid_llc_mean_pattern = {self.llc2llcid[llc_ids]: [] for llc_ids in self.all_llc}
        for valid_id_i in self.tvs.valid_ids:
            for i, llc_i in enumerate(self.data_llc[valid_id_i]):
                valid_llc_ids[self.llc2llcid[llc_i]] +=1
                valid_llc_mean_pattern[self.llc2llcid[llc_i]].append(np.mean(self.data[valid_id_i][i],axis=0))
        for i in range(self.n_llc):
            if len(np.shape(valid_llc_mean_pattern[i])) == 1:
                valid_llc_mean_pattern[i] = None
            else:
                valid_llc_mean_pattern[i] = np.mean(valid_llc_mean_pattern[i],axis=0)
        # make label of which llc is in the validation set
        llc_in_valid_set = np.array([valid_llc_mean_pattern[i] is not None for i in range(self.n_llc)])
        return valid_llc_mean_pattern, llc_in_valid_set

    def validation_set_llc_RDM(self):
        valid_set_mean_pattern, llc_in_valid_set = self.compute_validation_set_mean_pattern()
        valid_set_mean_pattern_rmnone = np.array([valid_set_mean_pattern[i] for i in range(len(llc_in_valid_set)) if llc_in_valid_set[i]])
        return np.corrcoef(valid_set_mean_pattern_rmnone)

    '''plot utils'''

    def print_all_event_shapes(self):
        for k in self.data.keys():
            print(f'%3d\t{k}' % len(self.data[k]))
            for i, ll_event in enumerate(self.data[k]):
                print(f'\t\t{np.shape(ll_event)}')
        print('\n' + '-'*80)


    def show_llc_frequency(self):
        f, ax = plt.subplots(1,1, figsize=(5,14))
        ax.barh(range(self.n_llc), self.llc_counts, color='grey')
        ax.set_yticks(range(self.n_llc))
        ax.set_yticklabels(self.all_llc)
        ax.set_ylim((-1, self.n_llc))
        ax.set_xlabel('# event instances')
        # c_cpal = sns.color_palette('colorblind', n_colors=5)
        for ytick in ax.get_yticklabels():
            ytick.set_color(self.hlc_cpal[llc2hlcid[ytick.get_text()]])
        ax.set_title('Low-level event intance counts')
        sns.despine()

    def make_dendrogram(self, linked, yticklabels):
        c_cpal = sns.color_palette('colorblind', n_colors=5)
        c_cpal = np.roll(c_cpal,shift=1,axis=0)
        def color_func(*args, **kwargs): return 'k'

        f, ax = plt.subplots(1,1, figsize=(10, 14))
        dendrogram(linked,
                   orientation='left',
                   labels=yticklabels,
                   distance_sort='descending',
                   link_color_func=color_func,
                   show_leaf_counts=False, ax=ax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        for ytick in ax.get_yticklabels():
            ytick.set_color(c_cpal[self.llc2hlcid[ytick.get_text()]])
        ax.tick_params(axis='y', which='major', labelsize=18)
        return f, ax

    def make_sns_clustermap(self, mean_pattern, yticklabels, metric):
        c_cpal = sns.color_palette('colorblind', n_colors=5)
        c_cpal = np.roll(c_cpal,shift=1,axis=0)
        cg = sns.clustermap(
            mean_pattern,
            col_cluster=0,
            row_cluster=1,
            metric=metric,
            yticklabels=yticklabels,
            dendrogram_ratio=[.5, 0],
            figsize=(12, 12),
            cbar_pos=(0.05, 0.05, .03, .15),
            )
        cg.ax_heatmap.set_xlabel("Feature dimension")
        cg.ax_heatmap.set_xticks([0, 10, 20])
        cg.ax_heatmap.set_xticklabels([0, 10, 20])
        cg.ax_heatmap.set_title(f'metric={metric}')
        # color the text label
        for ytick in cg.ax_heatmap.get_yticklabels():
            ytick.set_color(c_cpal[self.llc2hlcid[ytick.get_text()]])

'''utils'''
def concat_lists(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))

def get_leaves(node):
    if isinstance(node, dict):
        leaves = []
        for key, value in node.items():
            leaves.extend(get_leaves(value))
        return leaves
    elif isinstance(node, list):
        return node
    else:
        return []

def compute_llc_mean_pattern(meta):
    mean_pattern = np.zeros((meta.n_llc, meta.dim))
    for k, llc_name in enumerate(meta.data.keys()):
        mean_pattern_llc = []
        for j in range(len(meta.data[llc_name])):
            mean_pattern_llc.append(np.mean(meta.data[llc_name][j],axis=0))
        mean_pattern[k] = np.mean(mean_pattern_llc, axis=0)
    return mean_pattern


if __name__ == "__main__":
    '''how to use'''
    np.random.seed(0)
    min_len = 30
    exlude_multi_chapter = 1
    verbose = 0
    meta = METAVideos(min_len=min_len, exlude_multi_chapter=exlude_multi_chapter, verbose=verbose)

    video_id = '4.4.4'
    print(type(meta.data))
    print(meta.data.keys())
    print(np.shape(meta.data[video_id]))
    print(meta.data[video_id])
    np.shape(meta.data[video_id])
    np.shape(np.concatenate(meta.data[video_id]))

    print(type(meta.data_llc))
    print(meta.data_llc.keys())
    print(np.shape(meta.data_llc[video_id]))
    print(meta.data_llc[video_id])
    # cmeta.get_video(video_id




    '''print out the number of event instances for each llc2llcid '''
    # meta.print_all_event_shapes()

    '''how to use sample_event_info and get_data'''

    X_list, llc_name_list = meta.get_data(to_torch=True)

    # visualize
    for i in range(len(llc_name_list)):
        llc_name = llc_name_list[i]
        llc_id = meta.llc2llcid[llc_name]
        hlc2hlcid = meta.llc2hlcid[llc_name]
        print(f'%3d %3d\t{llc_name}'%(hlc2hlcid, llc_id))

    '''low level event class frequency'''
    meta.show_llc_frequency()

    '''plot hier'''
    # remove multi-chapter action (mca)
    mask_rm_mca = np.array([v for v in meta.llc2hlcid.values()]) !=0
    mean_pattern = meta.llc_mean_pattern[mask_rm_mca]
    yticklabels = np.array(meta.all_llc)[mask_rm_mca]
    metric = 'correlation'
    meta.make_sns_clustermap(mean_pattern, yticklabels, metric)

    linked = linkage(mean_pattern, method='average', metric=metric)
    f, ax  = meta.make_dendrogram(linked, yticklabels)

    '''save mean patterns'''
    # from utils import pickle_save_dict
    # mean_pattern = compute_llc_mean_pattern(meta)
    # pickle_save_dict(mean_pattern, '../data/meta-llc-mean-pattern.pkl')
    # print(np.shape(mean_pattern))

    '''show RDM '''
    f, ax = plt.subplots(1,1, figsize=(5,4))
    sns.heatmap(np.corrcoef(meta.llc_mean_pattern), square=True, ax=ax)
    ax.set_xlabel('low level event type')
    ax.set_ylabel('low level event type')


    # '''distance between the mean pattern tree vs. true tree'''
    # import zss
    # from copy import deepcopy
    # from scipy.cluster.hierarchy import linkage, to_tree
    # from task._Node import Node, convert_to_zss_format, create_tree, print_tree
    #
    # # Root node label can be a name representing the entire hierarchy or simply "root"
    # hier_struct_rmca = deepcopy(hier_struct)
    # del hier_struct_rmca['multichapter_actions']
    # true_tree = create_tree("meta-root", hier_struct_rmca)
    # print_tree(true_tree)
    # method = 'average'
    # # Generate two sample linkages (as an example)
    # np.random.seed(42)
    # n_rand_trees = 100
    # distance_null = [None] * n_rand_trees
    # for i in range(n_rand_trees):
    #     X = np.random.random((41, 2))
    #     T_rand = to_tree(linkage(X, method=method, metric=metric))
    #     T_rand_zss = convert_to_zss_format(T_rand)
    #     distance_null[i] = zss.simple_distance(T_rand_zss, true_tree, Node.get_children, Node.get_label)
    #
    # distance_perm = [None] * n_rand_trees
    # for i in range(n_rand_trees):
    #     pi = np.random.permutation(len(mean_pattern))
    #     linked = linkage(mean_pattern[pi,:], method=method, metric=metric)
    #     T_perm = to_tree(linked)
    #     T_perm_zss = convert_to_zss_format(T_perm)
    #     distance_perm[i] = zss.simple_distance(T_perm_zss, true_tree, Node.get_children, Node.get_label)
    #
    # # compare the mean pattern tree vs. truth
    # T_hat = to_tree(linked)
    # T_hat_zss = convert_to_zss_format(T_hat)
    # distance = zss.simple_distance(T_hat_zss, true_tree, Node.get_children, Node.get_label)
    # print("Tree Edit Distance:", distance)
    #
    # f, ax = plt.subplots(1,1)
    # sns.histplot(distance_null, bins=10, ax=ax)
    # # sns.histplot(distance_perm, bins=10, ax=ax)
    # ax.axvline(distance, ls='--', color='k')
