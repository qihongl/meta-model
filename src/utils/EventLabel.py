import pickle
import numpy as np
import pandas as pd

EVENT_LABEL_FPATH = '../data/high_level_events/event_annotation_timing_average.csv'
# FPS = 25

subevn2cat = {
    'apply_chapstick': 4,
    'apply_lotion': 4,
    'bicep_curls': 1,
    'blowdry_hair': 4,
    'brush_hair': 4,
    'brush_teeth': 4,
    'clean_a_surface': 0,
    'comb_hair': 4,
    'do_stair_steps': 1,
    'drink_sport_drink': 1,
    'drink_water': 1,
    'eat_a_granola_bar': 1,
    'floss_teeth': 4,
    'fold_blanket_or_comforter': 3,
    'fold_shirts_or_pants': 3,
    'fold_socks': 3,
    'fold_towels': 3,
    'jump_rope': 1,
    'look_at_text_message': 0,
    'perform_jumping_jacks': 1,
    'prepare_a_bagel': 2,
    'prepare_cereal': 2,
    'prepare_fresh_fruit': 2,
    'prepare_hot_oatmeal': 2,
    'prepare_instant_coffee': 2,
    'prepare_milk': 2,
    'prepare_orange_juice': 2,
    'prepare_tea': 2,
    'prepare_toast': 2,
    'prepare_yogurt_with_granola': 2,
    'push_ups': 1,
    'put_cases_on_pillows': 3,
    'put_objects_in_drawers': 0,
    'put_on_bed_sheets': 3,
    'shave_face': 4,
    'shoulder_press': 1,
    'sit_ups': 1,
    'take_a_pill': 0,
    'take_objects_out_of_drawers': 0,
    'torso_rotations': 1,
    'use_hair_gel': 4,
    'use_hand_duster': 3,
    'use_mouthwash': 4,
    'use_vacuum_attachment': 3,
    'vacuum_floor': 3,
    'wash_face': 4
}

hierarchical_structure = {
    "exercising": {
        "performing_cardio_exercises": [
            "jump_rope",
            "perform_jumping_jacks",
            "do_stair_steps"
        ],
        "performing_resistance_exercises": [
            "bicep_curls",
            "shoulder_press",
            "push_ups",
            "sit_ups"
        ],
        "taking_a_snack_break": [
            "drink_sport_drink",
            "eat_a_granola_bar",
            "drink_water"
        ]
    },
    "making_breakfast": {
        "preparing_main_items": [
            "prepare_cereal",
            "prepare_a_bagel",
            "prepare_hot_oatmeal"
        ],
        "preparing_side_items": [
            "prepare_fresh_fruit",
            "prepare_toast",
            "prepare_yogurt_with_granola"
        ],
        "preparing_beverages": [
            "prepare_orange_juice",
            "prepare_tea",
            "prepare_instant_coffee"
        ]
    },
    "cleaning_a_room": {
        "removing_dust": [
            "vacuum_floor",
            "use_hand_duster",
            "use_vacuum_attachment"
        ],
        "preparing_a_bed": [
            "put_on_bed_sheets",
            "put_cases_on_pillows",
            "fold_blanket_or_comforter"
        ],
        "folding_laundry": [
            "fold_shirts_or_pants",
            "fold_socks",
            "fold_towels"
        ]
    },
    "bathroom_grooming": {
        "grooming_mouth": [
            "brush_teeth",
            "floss_teeth",
            "use_mouthwash",
            "apply_chapstick"
        ],
        "grooming_hair": [
            "blowdry_hair",
            "brush_hair",
            "comb_hair",
            "use_hair_gel"
        ],
        "grooming_face": [
            "wash_face",
            "shave_face",
            "apply_lotion"
        ]
    },
    "multichapter_actions": [
        "take_a_pill",
        "torso_rotations",
        "put_objects_in_drawers",
        "take_objects_out_of_drawers",
        "clean_a_surface",
        "look_at_text_message"
    ]
}

class EventLabel:
    '''event labeler - run this from src '''

    def __init__(self):
        self.df = pd.read_csv(EVENT_LABEL_FPATH)
        self.all_subev_names = np.unique(self.df['evname'])
        self.n_subev_names = len(self.all_subev_names)
        self.subevn2cat = subevn2cat
        self.hierarchical_structure = hierarchical_structure

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

    (t_start, t_end) = evlab.get_subev_times(event_id, 1)
    print(t_start, t_end)

    event_bound, event_bound_vec = evlab.get_bounds(event_id)
    subev_labels = evlab.get_subev_labels(event_id, to_sec=False)

    print(event_bound)
    print(subev_labels)


    # import networkx as nx
    # import matplotlib.pyplot as plt

    # hierarchical_structure = {
    #     # ... (your hierarchical structure goes here)
    # }

    def add_edges(graph, node, parent=None):
        if parent is not None:
            graph.add_edge(parent, node)
        for k, v in hierarchical_structure.get(node, {}).items():
            if isinstance(v, dict):
                add_edges(graph, k, node)
            else:
                for item in v:
                    graph.add_edge(node, item)

    G = nx.DiGraph()
    add_edges(G, list(hierarchical_structure.keys())[0])

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10)
    plt.title("Hierarchical Structure Visualization")
    plt.show()
