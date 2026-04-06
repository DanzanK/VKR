# Copyright 2025 Sear Gamemode
import re 
from datetime import datetime
import os 
import mne 
import numpy as np 
from typing import Optional, Tuple, Dict, Any

_SCENARIO_ALIASES = {
    "ants": "ANT",
    "ant": "ANT",
    "mmns": "MMN",
    "mmn": "MMN",
    
    "riti": "RiTi",
    "rise": "RiTi",
    "risetime": "RiTi",
    "rise_time": "RiTi",
    "rise-time": "RiTi",

    "ssds": "SST",
    "sst": "SST",
    "speech": "speech",
    "spch": "speech",

    "rest": "Rest",
    "resting": "Rest",
    "restingstate": "Rest",
    "restingstateeeg": "Rest",
    "resting_state": "Rest",
    "resting-state": "Rest",
}

def extract_visit_num_from_path(path: str):
    m = re.search(r"посещение\s*(\d+)", path, flags=re.I)
    return int(m.group(1)) if m else None

def canonical_scenario(name: str) -> str:
    """
    Приводим 'ANTs', 'ant', 'rs11' -> каноническому ключу.
    """
    if not name:
        return "UNKNOWN"
    s = str(name).strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    # rs11/rs12/rs13/rs21/rs22/rs23 -> Rest
    if re.fullmatch(r"rs\d\d", s):
        return "Rest"
    return _SCENARIO_ALIASES.get(s, name.strip())

    mapping = {
        "ants": "ant",
        "ant": "ant",
        "mmns": "mmn",
        "mmn": "mmn",
        "ssds": "sst",
        "sst": "sst",
        "spch": "speech",
        "speech": "speech",
        "riti": "riti",
    }
    return mapping.get(s, s)




def _event_code_from_desc(desc: str) -> Optional[int]:
    """
    Из 'Stimulus/s140' или 'Stimulus/S 140' достаём 140.
    """
    if not desc:
        return None
    m = re.search(r"(\d{1,4})\s*$", str(desc))
    return int(m.group(1)) if m else None


def preprocessing_events(raw, scenarious='RiTi'):
    scen = canonical_scenario(scenarious)

    try:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
    except Exception:
        events, event_id = None, None

    if events is None:
        events = np.zeros((0, 3), dtype=int)
    if event_id is None:
        event_id = {}

    reverse_event_id = {v: k for k, v in event_id.items()}

    base_repo = {
        'Result_of_quality_checking': 'N/A',
        'Initial_Events': int(len(events)),
        'Filtered_Events': int(len(events)),
        'Expected_Events': 'N/A',
        'SubSeq_Flag': 'N/A',
        'Match_Indexes': 'N/A',
        'Transitions': 'N/A',
    }

    # --- RiTi: оставляем seq-проверку ---
    if scen == 'RiTi':
        seq_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RiTi_seq.txt')

        if not os.path.exists(seq_path):
            repo = dict(base_repo)
            repo['Filtered_Events'] = 0
            repo['Expected_Events'] = 0
            repo['SubSeq_Flag'] = 'SEQ_FILE_NOT_FOUND'
            repo['Result_of_quality_checking'] = 'FAIL: SEQ_FILE_NOT_FOUND'
            return np.zeros((0, 3), dtype=int), reverse_event_id, repo

        seq_list = []
        transform = {'Stimulus/s115': 1, 'Stimulus/s120': 4, 'Stimulus/s130': 2, 'Stimulus/s160': 3, 'Stimulus/s240': 5}
        reverse_transform = dict(zip(transform.values(), transform.keys()))

        with open(seq_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    seq_list.append(reverse_transform[int(line)])

        new_events = verificate_events(raw, seq_list)
        if new_events is None or len(new_events) == 0:
            repo = dict(base_repo)
            repo['Filtered_Events'] = 0
            repo['Expected_Events'] = int(len(seq_list))
            repo['SubSeq_Flag'] = False
            repo['Result_of_quality_checking'] = 'FAIL: NO_MATCHED_EVENTS'
            return np.zeros((0, 3), dtype=int), reverse_event_id, repo

        filtered_seq = [
            reverse_event_id.get(int(new_events[j, 2]), str(int(new_events[j, 2])))
            for j in range(len(new_events))
        ]
        subseq, match_idxs, tr = is_subsequence(filtered_seq, seq_list)

        repo = dict(base_repo)
        repo.update({
            'Filtered_Events': int(len(new_events)),
            'Expected_Events': int(len(seq_list)),
            'SubSeq_Flag': bool(subseq),
            'Match_Indexes': int(match_idxs),
            'Transitions': int(tr),
        })
        if subseq and int(len(new_events)) >= int(len(seq_list)):
            repo['Result_of_quality_checking'] = 'OK'
        else:
            repo['Result_of_quality_checking'] = f'FAIL: SEQ_MISMATCH ({len(new_events)}/{len(seq_list)})'
        return new_events.astype(int), reverse_event_id, repo

    # --- общие сценарии: фильтруем по кодам ---
    # коды из ТЗ: Rest: 1,2,3,4,11,12; ANT: 140-155 + 200-202; MMN: 161-163; N400: 203-204
    rules = {
        'ANT':  {'target': set(range(140, 156)), 'need_any': set(range(140, 156)), 'resp': {200, 201, 202}},
        'MMN':  {'target': {161, 162, 163}, 'need_any': {161, 162, 163}, 'resp': set()},
        'Rest': {'target': {11, 12}, 'need_any': {11, 12}, 'resp': set(), 'need_markers': {1, 2}},
        'N400': {'target': {203, 204}, 'need_any': {203, 204}, 'resp': set()},
        # SST пока без точных кодов — не ломаемся
        'SST':  {'target': set(), 'need_any': set(), 'resp': set()},
    }

    rule = rules.get(scen, None)
    if rule is None:
        # неизвестный сценарий: просто не падаем
        repo = dict(base_repo)
        repo['Result_of_quality_checking'] = 'N/A'
        return events.astype(int), reverse_event_id, repo

    # вычислим коды событий
    codes = []
    for e in events:
        desc = reverse_event_id.get(int(e[2]), "")
        codes.append(_event_code_from_desc(desc))

    target = rule.get('target', set())
    if target:
        mask = [(c in target) for c in codes]
        new_events = events[np.array(mask, dtype=bool)]
    else:
        new_events = events

    repo = dict(base_repo)
    repo['Filtered_Events'] = int(len(new_events))

    # QC-логика: минимальные sanity-check условия
    codes_set = {c for c in codes if c is not None}
    need_any = rule.get('need_any', set())
    need_markers = rule.get('need_markers', set())

    missing_any = []  # группы по "хотя бы один"
    if need_any and len(codes_set.intersection(need_any)) == 0:
        missing_any.append(f"need_any={sorted(list(need_any))}")

    missing_markers = []
    for mk in need_markers:
        if mk not in codes_set:
            missing_markers.append(mk)

    if scen == 'ANT':
        # доп.статистика по ответам
        resp = rule.get('resp', set())
        resp_cnt = sum((c in resp) for c in codes if c is not None)
        stim_cnt = sum((c in need_any) for c in codes if c is not None)
        repo['Expected_Events'] = f">=1 stimulus ({min(need_any)}-{max(need_any)})"
        if stim_cnt > 0 and resp_cnt > 0:
            repo['Result_of_quality_checking'] = 'OK'
        else:
            repo['Result_of_quality_checking'] = f'FAIL: stim={stim_cnt}, resp={resp_cnt}'

    elif scen == 'MMN':
        repo['Expected_Events'] = ">=1 (161/162/163)"
        has_std = 161 in codes_set
        has_dev = (162 in codes_set) or (163 in codes_set)
        repo['Result_of_quality_checking'] = 'OK' if (has_std and has_dev) else f'FAIL: std={has_std}, dev={has_dev}'

    elif scen == 'Rest':
        repo['Expected_Events'] = ">=1 (11/12) + markers 1&2"
        if missing_markers:
            repo['Result_of_quality_checking'] = f'FAIL: missing markers {missing_markers}'
        elif missing_any:
            repo['Result_of_quality_checking'] = f'FAIL: no eyes-open/closed (11/12)'
        else:
            repo['Result_of_quality_checking'] = 'OK'

    elif scen == 'N400':
        repo['Expected_Events'] = ">=1 (203/204)"
        repo['Result_of_quality_checking'] = 'OK' if not missing_any else 'FAIL: no 203/204'

    else:
        # SST / неизвестное: хотя бы не падаем
        repo['Result_of_quality_checking'] = 'N/A'

    return new_events.astype(int), reverse_event_id, repo


    

def get_meta(file_name: str) -> Optional[Dict[str, Any]]:
    """
    Понимает имена вида:
      INP0008_v1.4_ANTs_R003_27.10.23.vhdr
      RNS060_v1_ANTs_Op013_29.11.2025.vhdr
    """
    base = os.path.basename(file_name)

    # prefix + id (INP#### или RNS###/RNS####)
    # v<something> where visit may be "1.4" or "4" etc.
    pat = re.compile(
        r'(?P<prefix>INP|RNS)(?P<id>\d{3,4})_v(?P<vraw>\d+(?:\.(?P<visit_num>\d+))?)_'
        r'(?P<experiment>[^_]+)_(?P<operator_code>[^_]+)_(?P<date>\d{2}\.\d{2}\.\d{2,4})'
        r'\.(?P<format>vhdr|vmrk|eeg)$',
        flags=re.IGNORECASE
    )

    m = pat.match(base)
    if not m:
        return None

    meta = m.groupdict()

    # visit_num: если есть "v1.4" -> 4, иначе "v4" -> 4, иначе "v1" -> 1
    if meta.get("visit_num") is None:
        try:
            meta["visit_num"] = int(meta["vraw"])
        except Exception:
            meta["visit_num"] = None
    else:
        meta["visit_num"] = int(meta["visit_num"])

    meta["prefix"] = meta["prefix"].upper()
    meta["id"] = str(meta["id"])
    meta["experiment"] = str(meta["experiment"])
    meta["operator_code"] = str(meta["operator_code"])
    return meta




def get_brainvision_files(vhdr_path):
    eeg_fname = None
    vmrk_fname = None
    with open(vhdr_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith('DataFile='):
            eeg_fname = line.split('=')[1].strip()
        elif line.startswith('MarkerFile='):
            vmrk_fname = line.split('=')[1].strip()
    return (eeg_fname, vmrk_fname)

def extract_visit_num_from_visit_folder(visit_folder_name: str) -> Optional[int]:
    """
    'посещение 5' -> 5
    'посещение 5 визит 2' -> 5 (визит игнорируем)
    """
    if not visit_folder_name:
        return None
    s = str(visit_folder_name).lower()
    m = re.search(r"посещ\w*\s*(\d+)", s)
    if m:
        return int(m.group(1))
    # fallback: просто первое число
    m2 = re.search(r"(\d+)", s)
    return int(m2.group(1)) if m2 else None


def extract_prefix_id_from_participant_folder(participant_folder: str) -> Tuple[Optional[str], Optional[str]]:
    """
    'RNS060' -> ('RNS', '060'), 'INP0008' -> ('INP','0008')
    """
    if not participant_folder:
        return None, None
    m = re.match(r"^(INP|RNS)(\d{3,4})$", participant_folder.strip(), flags=re.IGNORECASE)
    if not m:
        return None, None
    return m.group(1).upper(), m.group(2)


def get_file_info(file_path, elc_path):
    raw = mne.io.read_raw(file_path, preload=False, verbose=False)
    other_files = ()
    if os.path.splitext(file_path)[1] == '.vhdr':
        other_files = get_brainvision_files(file_path)
    eeg_files = list(map(os.path.basename, (file_path,) + other_files))

    # meta теперь всегда dict (в худшем случае пустой {})
    meta = get_meta(eeg_files[0]) or {}

    # fallback из структуры папок: .../<participant>/<visit>/<experiment>/Raw/<file>
    RAW_PATH = os.path.dirname(file_path)
    EXPERIMENT_PATH = os.path.dirname(RAW_PATH)
    VISIT_PATH = os.path.dirname(EXPERIMENT_PATH)
    PARTICIPANT_PATH = os.path.dirname(VISIT_PATH)

    participant_folder = os.path.basename(PARTICIPANT_PATH)
    visit_folder = os.path.basename(VISIT_PATH)
    experiment_folder = os.path.basename(EXPERIMENT_PATH)

    prefix_f, id_f = extract_prefix_id_from_participant_folder(participant_folder)
    visit_f = extract_visit_num_from_visit_folder(visit_folder)
    scen_f = canonical_scenario(experiment_folder)

    # добиваем meta отсутствующие поля
    if "prefix" not in meta or not meta.get("prefix"):
        meta["prefix"] = prefix_f
    if "id" not in meta or not meta.get("id"):
        meta["id"] = id_f
    if "visit_num" not in meta or meta.get("visit_num") in (None, "N/A"):
        meta["visit_num"] = visit_f
    if "experiment" not in meta or not meta.get("experiment"):
        meta["experiment"] = scen_f


    visit_from_path = extract_visit_num_from_path(file_path)
    visit_num = str(visit_from_path) if visit_from_path is not None else str(meta.get("visit_num", "N/A"))

    scenario_raw = meta.get("experiment", "N/A")
    scenario = canonical_scenario(scenario_raw)
    meas_date = raw.info['meas_date']

    data = {
        "id": meta.get("id", "N/A"),
        "visit_num": visit_num,
        "scenario": scenario,
        "scenario_raw": scenario_raw,
        "operator_code": ', '.join(meta.get('operator_code', 'N/A').split('_')),
        "eeg_files": ' '.join(eeg_files),
        "elc_file": os.path.basename(elc_path) if elc_path else 'Not found',
        "raw_type": type(raw).__name__,
        "duration": f"{raw.times[-1] - raw.times[0]:.2f}",
        "nchan": raw.info['nchan'],
        "electrodes_of_montages": [
            raw.info['ch_names'][i] for i in mne.pick_types(raw.info, eeg=True)
        ],
        # заодно поправлю логику — раньше там было странное сравнение с {}
        "ecg_channel": any('BIP' in ch for ch in raw.ch_names),
        "eog_channel": any('EOG' in ch for ch in raw.ch_names),
        "sfreq": raw.info['sfreq'],
        "measurement_datetime": (
            meas_date.strftime('%Y-%m-%d %H:%M:%S')
            if isinstance(meas_date, datetime) else 'Unknown'
        ),
        "highpass": raw.info['highpass'],
        "lowpass": raw.info['lowpass'],
        "custom_ref_applied": raw.info.get('custom_ref_applied', 'Not available'),
        
    }
    return data



def is_subsequence(subseq, seq):
    subseq_index = 0
    subseq_len = len(subseq)
    flag = 0
    transitions = 0
    for elem in seq:
        copy_flag = flag
        if subseq_index < subseq_len and elem == subseq[subseq_index]:
            subseq_index += 1
            flag = 1
        else:
            flag = 0
        if flag != copy_flag:
            transitions += 1
    return subseq_index == subseq_len, subseq_index, transitions


def verificate_events(raw, seq_list, shift=1.85, var=0.35):
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    reverse_event_id = dict(zip(event_id.values(), event_id.keys()))
    tar_idx = 0
    new_events = []
    n_events = len(events)
    n_seq = len(seq_list)
    events_times = events[:, 0]/raw.info['sfreq']
    events_seq = np.array([reverse_event_id[events[i, 2]] for i in range(len(events))])
    current_time = events_times[0] - shift

    while not (tar_idx > n_seq - 1 or current_time > events_times[-1]):
        
        next_time_l = current_time + shift - var
        next_time_r = current_time + shift + var

        mask = (events_times >= next_time_l) & (events_times <= next_time_r)
        filtered_seq = events_seq[mask]
        filtered_events = events[mask]
        filtered_times = events_times[mask]
        if filtered_seq.size > 0:
            unique_elements = np.unique(filtered_seq)
            if len(unique_elements) == 1:
                new_events.append(filtered_events[0])
                tar_idx += 1
                current_time = filtered_times[0]
            else:
                matches = seq_list[tar_idx] == filtered_seq
                if matches.all():
                    new_events.append(filtered_events[matches[0]])
                    tar_idx += 1
                    current_time = filtered_times[matches[0]]
                else:    
                    tar_idx += 1
                    current_time += shift
        else:
            tar_idx += 1
            current_time += var
    
    return np.array(new_events)
