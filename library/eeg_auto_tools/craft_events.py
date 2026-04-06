# Copyright 2025 Sear Gamemode
import numpy as np 
import mne 
import re 

def get_ANT_ROI():
    P1_ROI = ['P3', 'Pz', 'P4', 'POz', 'P1', 'P2']
    N1_ROI = ['P3', 'Pz', 'P4', 'O1', 'Oz', 'O2']
    CNV_ROI = ['FCz', 'Cz']
    N2_ROI = ['P3', 'Pz', 'P4', 'POz', 'P1', 'P2']
    P3_ROI = ['P3', 'Pz', 'P4', 'POz', 'P1', 'P2']
    COMPONENTS_ROI = {'P1': P1_ROI,
           'N1': N1_ROI,
           'CNV': CNV_ROI,
           'N2': N2_ROI,
           'P3': P3_ROI}
    return COMPONENTS_ROI

def make_ANT_events(raw, target_stimulus):
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    desired_stimulus = 'Stimulus/s200'

    target_stimuli_codes = dict(zip(event_id.values(), event_id.keys()))
    filtered_events = []

    for i in range(0, len(events) - 1):
        cur = events[i]
        nxt = events[i + 1]
        if target_stimuli_codes[cur[2]] in target_stimulus and target_stimuli_codes[nxt[2]] == desired_stimulus:
            filtered_events.append(cur)

    filtered_events = np.array(filtered_events, dtype=int).reshape(-1, 3)
    filtered_event_id = {stim: event_id[stim] for stim in target_stimulus if stim in event_id}

    return filtered_events, filtered_event_id


def make_CB_events(raw, mode):
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    event_id_rev = {v: k for k, v in event_id.items()}

    if mode == "Rest-IAT":
        reg = "cb"
        cb_events = [event for event in events if reg in event_id_rev[event[2]]]

    else:
        reg = "tar"
        cb_events = []
        seen_rn_values = set()

        for event in events:
            event_code = event_id_rev[event[2]]
            if (reg in event_code):
                rn_value = None
                match = re.search(r"rn(\d+)", event_code)
                if match:
                    rn_value = match.group(1)

                if rn_value and rn_value not in seen_rn_values:
                    cb_events.append(event)
                    seen_rn_values.add(rn_value)

    cb_events = np.array(cb_events)

    unique_cb_labels = list(set(event_id_rev[event[2]] for event in cb_events))
    cb_event_id = {label: event_id[label] for label in unique_cb_labels}
    return cb_events, cb_event_id

def make_RiTi_events(raw, stimulus_list, filt=False):
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    inverted_dict = {v: k for k, v in event_id.items()}  #  event_id  
    if filt:
        filtered_events = []
        for event in events:
            stimulus = inverted_dict[event[2]]
            if stimulus in stimulus_list:
                filtered_events.append(event)
        filtered_events = np.array(filtered_events, dtype=int)
    else:    
        filtered_events = np.array(events, dtype=int)

    duplet_events = []
    duplet_names = {}
    duplet_code = 1  #    

    for i in range(1, len(filtered_events)):
        current_stimulus = inverted_dict[filtered_events[i, 2]]  #  
        previous_stimulus = inverted_dict[filtered_events[i - 1, 2]]  #  
        if (current_stimulus in stimulus_list) and (previous_stimulus in stimulus_list) and (current_stimulus != previous_stimulus):
            duplet_name = f'{previous_stimulus[:]}_{current_stimulus[:]}'
            if duplet_name not in duplet_names:
                duplet_names[duplet_name] = duplet_code
                duplet_code += 1
            duplet_events.append([filtered_events[i, 0], 0, duplet_names[duplet_name]])
    duplet_events = np.array(duplet_events, dtype=int)
    event_id_duplets = {name: code for name, code in duplet_names.items()}
    return duplet_events, event_id_duplets

