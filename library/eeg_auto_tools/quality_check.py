# Copyright 2025 Sear Gamemode
import mne 
import os
import numpy as np
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from mne.preprocessing import compute_bridged_electrodes
from itertools import chain
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import json
import re


from autoreject import Ransac 
from joblib import Parallel, delayed
from .savers import bridge_save, event_saver, get_clusters
from .metrics import isolation_forest, check_volt_of_epochs 
from .scenarious import verificate_events, is_subsequence
from .montages import create_custom_montage, read_elc, align_head

def compute_bad_epochs(epochs, snr_matrix, roi_channels=None, thr_auto=True):
    n_trials, n_channels, n_times = epochs._data.shape
    ch_names = epochs.ch_names 
    
    if thr_auto:
        threshold_ep=np.percentile(snr_matrix, 20, axis=(0, 1))
        threshold_ch=np.percentile(snr_matrix, 0.5, axis=(0, 1))
        volt_max=dict(eeg=np.percentile(np.abs(epochs._data), 99.999, axis=(0, 1, 2)))
        volt_min=dict(eeg=np.percentile(np.abs(epochs._data), 10, axis=(0, 1, 2)))
    else:
        threshold_ep=-22
        threshold_ch=-30
        volt_max = dict(eeg=130e-6)
        volt_min = dict(eeg=1e-6)

    #--------------SNR---------------#
    rej_dict = {}
    
    rej_dict['SNR_over_channels'] = np.where(np.mean(snr_matrix, axis=1) < threshold_ep)[0].tolist() #   ,      
    rej_dict['SNR_channel'] = []
    for epoch_idx in range(n_trials):
        channel_names_above_threshold = [ch_names[ch] for ch in np.where(snr_matrix[epoch_idx] < threshold_ch)[0]] #     
        if channel_names_above_threshold:
            rej_dict['SNR_channel'].append((epoch_idx, channel_names_above_threshold)) #         

    #-------ISOLATION_FOREST---------#
    rej_dict['Isol_forest_all_chns'] = isolation_forest(epochs.copy(), mode='ep')
    if roi_channels:
        rej_dict['Isol_forest_tar_chns'] = isolation_forest(epochs.copy().pick(roi_channels), mode='ep')
    #------------VOLT----------------#
    rej_dict['Volt_max'] = check_volt_of_epochs(epochs, reject=volt_max, flat=None)
    rej_dict['Volt_flat'] = check_volt_of_epochs(epochs, reject=None, flat=volt_min)

    #--------------FINAL-------------#
    snr_all_channels_indices = set(rej_dict['SNR_over_channels'])
    snr_channel_indices = set(epoch for epoch, _ in rej_dict['SNR_channel'])
    volt_max_indices = set(rej_dict['Volt_max'])
    volt_min_indices = set(rej_dict['Volt_flat'])
    isol_forest_all_indices = set(rej_dict['Isol_forest_all_chns'])

    combined_indices = snr_all_channels_indices | snr_channel_indices | volt_max_indices | isol_forest_all_indices | volt_min_indices
    if roi_channels:
        isol_forest_tar_indices = set(rej_dict['Isol_forest_tar_chns'])
        combined_indices = combined_indices | isol_forest_tar_indices
        
    rej_dict['FINAL'] = sorted(list(combined_indices))
    rej_dict['Percentage_removed_trials'] = len(rej_dict['FINAL'])/epochs._data.shape[0]*100
    #for key in rej_dict.keys():
    #    print(f'rej_dict[{key}] = {rej_dict[key]}')
    return rej_dict

def set_montage(raw, montage, elc_file, mode, threshold, verbose=False, interpolate=None, vis=None):
    if montage=='waveguard64':
        montage = create_custom_montage(montage)
    elif montage == 'personal':
        ch_dict, nasion, lpa, rpa, hsp = read_elc(elc_file)
        if interpolate:
            ch_dict, nasion, lpa, rpa, hsp = align_head(ch_dict, nasion, np.array(lpa), np.array(rpa), np.array(hsp), standard='waveguard64', 
                                                    mode=mode, threshold=threshold)
        montage = mne.channels.make_dig_montage(ch_pos=ch_dict, nasion=nasion, lpa=lpa, rpa=rpa, hsp=hsp, coord_frame='head')
    else:
        montage = mne.channels.make_standard_montage(montage)
    
# ---- Fix channel names that don't exist in the montage (e.g., Oz_1) ----
    montage_chs = set(getattr(montage, "ch_names", []))

    # 1) Если есть каналы вида *_1 и есть базовое имя в монтаже — переименуем
    rename_map = {
        ch: ch[:-2]
        for ch in raw.ch_names
        if ch.endswith("_1") and (ch[:-2] in montage_chs) and (ch[:-2] not in raw.ch_names)
    }
    if rename_map:
        raw.rename_channels(rename_map)

        # 1.5) RNS: каналы BIP* не являются scalp EEG и обычно не имеют позиций
    bip_chs = [ch for ch in raw.ch_names if re.fullmatch(r"BIP\d+", ch, flags=re.IGNORECASE)]
    if bip_chs:
        raw.set_channel_types({ch: "misc" for ch in bip_chs})


    # 2) Если после этого остались *_1, которых нет в монтаже — делаем их misc
    leftover = [ch for ch in raw.ch_names if ch.endswith("_1") and ch not in montage_chs]
    if leftover:
        raw.set_channel_types({ch: "misc" for ch in leftover})

        # 2.5) Любые EEG-каналы, которых нет в монтаже → в misc (иначе MNE/топомапы ломаются)
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    missing_eeg = [raw.ch_names[p] for p in eeg_picks if raw.ch_names[p] not in montage_chs]
    if missing_eeg:
        raw.set_channel_types({ch: "misc" for ch in missing_eeg})

    try:
        raw.set_montage(montage, verbose=verbose, match_case=False, on_missing="ignore")
    except TypeError:
        # на случай старой версии mne без match_case
        raw.set_montage(montage, verbose=verbose, on_missing="ignore")

    if vis:
        fig = montage.plot(show_names=True, kind='3d')
        for ax in fig.get_axes():
            if ax.name == '3d':
                ax.set_xlim([-0.1, 0.1])
                ax.set_ylim([-0.1, 0.1])
                ax.set_zlim([-0.1, 0.1])
    return raw


def detect_bad_channels(raw, method='auto'):
    raw = raw.copy().pick_types(eeg=True, exclude=[])
    if len(raw.ch_names) < 3:
        return {"HighAmp": [], "LowAmp": [], "Bridged": [], "Noise_Rate": []}, [], [], []


    bad_channels = []
    scores = []
    electrodesD = {}
    flat_chans, noisy_channels, _, _ = get_flat_channels(raw)
    clusters, bridge_figs = search_bridge_cluster(raw)
    bridged_electrodes = list(set(list(chain.from_iterable(clusters))))
    raw = raw.drop_channels(np.concatenate([flat_chans, noisy_channels, bridged_electrodes]))
    if method == 'ransac':
        bad_channels, scores, noised_fig = DNC_ransac(raw)
    elif method == 'ed':
        bad_channels, scores, noised_fig = DNC_electrical_distance(raw)
    elif method == 'corr':
        bad_channels, scores, noised_fig = DNC_corr(raw)
    elif method == 'lof':
        bad_channels, scores = mne.preprocessing.find_bad_channels_lof(raw, metric="euclidean", return_scores=True, verbose=False)
    elif method in ['auto', 'SN_ratio']:
        bad_channels, scores, noised_fig = DNC_SN_ratio(raw)
    else:
        raise ValueError(f"Unknown method '{method}'. Please use 'ransac', 'neighbours', 'psd', 'ed', 'corr' or 'auto'")
    
    electrodesD['HighAmp'] = noisy_channels
    electrodesD['LowAmp'] = flat_chans
    electrodesD['Bridged'] = bridged_electrodes
    electrodesD['Noise_Rate'] = bad_channels
    return electrodesD, clusters, bridge_figs, noised_fig
    

def get_flat_channels(raw, threshold_min=3e-6, threshold_max=300e-6, threshold_length=0.5):
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(picks) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    data = raw.get_data(picks=picks)
    ch_names = np.array([raw.ch_names[p] for p in picks])

    empty_scores = np.array([np.mean(np.abs(data[idx]) < threshold_min) for idx, ch in enumerate(ch_names)])
    empty_channels = np.array(ch_names[empty_scores > threshold_length])
    max_scores = np.array([np.mean(np.abs(data[idx]) > threshold_max) for idx, ch in enumerate(ch_names)])
    max_channels = ch_names[max_scores > threshold_length] 
    return empty_channels, max_channels, empty_scores, max_scores

def search_bridge_cluster(raw, threshold=0.99):
    # Берём только EEG-каналы (misc / stim / eog и т.п. исключаются)
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(picks) < 2:
        return [], []

    ch_names = [raw.ch_names[p] for p in picks]
    data = raw.get_data(picks=picks)
    corr_matrix = np.corrcoef(data)

    # heuristics
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    np.fill_diagonal(corr_matrix, 0)

    binary_matrix = (corr_matrix > threshold).astype(int)
    G = nx.from_numpy_array(binary_matrix)
    cliques = list(nx.find_cliques(G))

    clusters_indices = [clique for clique in cliques if len(clique) >= 2]
    clusters_names = [[ch_names[i] for i in comp] for comp in clusters_indices]

    clusters_stats = []
    for comp in clusters_indices:
        comp_names = [ch_names[i] for i in comp]
        corr_vals = []
        for i in range(len(comp)):
            for j in range(i + 1, len(comp)):
                corr_vals.append(corr_matrix[comp[i], comp[j]])
        corr_vals = np.asarray(corr_vals, dtype=float)
        if corr_vals.size == 0:
            continue

        clusters_stats.append({
            "channels": comp_names,
            "max_corr": float(np.max(corr_vals)),
            "min_corr": float(np.min(corr_vals)),
            "avg_corr": float(np.mean(corr_vals)),
        })

    # Координаты для топомапа: только по picks
    ch_pos = np.array([raw.info["chs"][p]["loc"][:2] for p in picks], dtype=float)

    # Фильтруем NaN/Inf, иначе plot_topomap падает
    good = np.isfinite(ch_pos).all(axis=1)

    # Если точек мало — топомап нельзя построить (но QC пусть продолжается)
    if good.sum() < 3:
        return clusters_names, []

    ch_names_plot = [n for n, g in zip(ch_names, good) if g]
    ch_pos_plot = ch_pos[good]
    corr_plot = corr_matrix[np.ix_(good, good)].flatten()

    # Подчищаем stats: убираем каналы, которых нет в plot-наборе
    keep = set(ch_names_plot)
    clusters_stats_plot = []
    for st in clusters_stats:
        chs = [c for c in st["channels"] if c in keep]
        if len(chs) >= 2:
            st2 = dict(st)
            st2["channels"] = chs
            clusters_stats_plot.append(st2)

    try:
        figs = get_clusters(clusters_stats_plot, ch_names_plot, ch_pos_plot, corr_plot)
    except Exception:
        figs = []

    return clusters_names, figs



def search_bridge_cluster_with_times(raw, window_size=2.0, overlap=0.5, threshold=0.99):
    sfreq = raw.info['sfreq']
    window_samples = int(window_size * sfreq)
    step_samples = int(window_samples * (1 - overlap))

    clusters_list = []
    bridge_times = []

    n_samples = raw.n_times
    for start in range(0, n_samples - window_samples + 1, step_samples):
        end = start + window_samples
        tmin = start / sfreq
        tmax = (end - 1) / sfreq
        raw_win = raw.copy().crop(tmin=tmin, tmax=tmax)
        clusters, _ = search_bridge_cluster(raw_win, threshold=threshold)
        clusters_list.append(clusters)
        bridge_times.append(start)

    return clusters_list, bridge_times



def event_check(raw, mind_stimulus, proc_stimulus, saving_dir=None, vis=False):
    proc_count = 0
    mind_count = 0
    other_count = 0
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    rev_event_id = dict(list(zip(event_id.values(), event_id.keys())))
    mind_counter = {stimulus: 0 for stimulus in mind_stimulus}
    proc_counter = {stimulus: 0 for stimulus in proc_stimulus}
    strange_list = set()
    for event in events:
        if rev_event_id[event[2]] in proc_counter.keys():
            for key in proc_counter.keys():
                if rev_event_id[event[2]] == key:
                    proc_counter[key]+=1
                    proc_count += 1
        elif rev_event_id[event[2]] in mind_counter.keys():
            for key in mind_counter.keys():
                if rev_event_id[event[2]] == key:
                    mind_counter[key]+=1
                    mind_count += 1
        else:
            strange_list.add(rev_event_id[event[2]])
            other_count += 1
    event_saver(proc_counter, mind_counter, saving_dir, vis)

def bridging_test(raw, saving_dir=None, vis=False):
    raw = raw.copy()
    bridged_idx, ed_matrix = compute_bridged_electrodes(raw, verbose=False)
    correlation_matrix = np.corrcoef(raw.get_data())
    bridge_save(raw, correlation_matrix, bridged_idx, ed_matrix, saving_dir, vis=vis)
    return bridged_idx

def find_adj_neighbors(raw, ch_name):
    montage = raw.get_montage()
    adjacency, ch_names = mne.channels.find_ch_adjacency(raw.info, 'eeg')
    adjacency = adjacency.toarray()
    idx = raw.ch_names.index(ch_name)
    neighbors_idx = np.where(adjacency[idx])[0]
    neighbors = [ch_names[i] for i in neighbors_idx]
    return neighbors

def calculate_correlations(raw, duration=1, overlap=0.5):
    epochs = mne.make_fixed_length_epochs(raw, duration=duration, overlap=overlap, verbose=False)
    data = epochs.get_data()
    n_trials, n_channels, n_times = epochs.get_data().shape

    adjacency, ch_names = mne.channels.find_ch_adjacency(raw.info, ch_type="eeg")

    neighbors_indices = [np.where(adjacency[i])[0] for i in range(n_channels)]
    correlations = np.zeros((n_trials, n_channels))  
    
    for i in range(n_trials):
        epoch_data = data[i] 
        for j in range(n_channels):
            neighbor_data = np.mean(epoch_data[neighbors_indices[j]], axis=0)
            channel_data = epoch_data[j]
            corr = np.corrcoef(channel_data, neighbor_data)[0, 1]
            correlations[i, j] = corr 
    return correlations

def compute_snr(signal, noise, eps=1e-12):
    power_signal = np.mean(signal ** 2)
    power_noise = np.mean(noise ** 2)
    snr_linear = power_signal / (power_noise + eps) # denominator is greater than zero
    snr_db = 10 * np.log10(snr_linear)
    return snr_linear, snr_db

def q_normalization(x, q1=75, q2=25):
    q75 = np.percentile(x, q1, axis=(0, 1), keepdims=True)
    q25 = np.percentile(x, q2, axis=(0, 1), keepdims=True)
    iqr = q75 - q25
    x_norm = x/iqr
    return x_norm 

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def plot_topomap(variable, raw, title, label, vlim):
    montage = raw.get_montage()
    fig, ax = plt.subplots()
    if montage is None:
        ax.set_title(f"{title}\n(no montage)")
        return fig

    pos_dict = montage.get_positions().get("ch_pos", {})
    variable = np.asarray(variable)

    names, pos, vals = [], [], []
    for i, ch in enumerate(raw.ch_names):
        if ch not in pos_dict:
            continue
        xy = np.asarray(pos_dict[ch])[:2]
        if not np.isfinite(xy).all():
            continue
        if i >= len(variable):
            continue
        names.append(ch)
        pos.append(xy)
        vals.append(variable[i])

    if len(pos) < 3:
        ax.set_title(f"{title}\n(not enough valid positions)")
        return fig

    im, _ = mne.viz.plot_topomap(vals, np.array(pos), axes=ax, show=False, names=names, vlim=vlim)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(label)
    return fig




class ENDetector():
    def __init__(self, k=2.0):
        self.model_params = None
        self.k = k

    def load(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, 'model_params.json')
        with open(file_path, 'r') as f:
            self.model_params = json.load(f)
            self.k = self.model_params["k"]
    
    def update(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, 'model_params.json')
        with open(file_path, 'w') as f:
            json.dump(self.model_params, f, indent=2)

    def fit(self, X):
        median_global = np.median(X)
        mad_global = np.median(np.abs(X - median_global))
        global_threshold = median_global - self.k * mad_global
        print(f"dB Threshold: {global_threshold:.2f} dB")
        self.model_params = {
                "median_global": float(median_global),
                "mad_global": float(mad_global),
                "k": self.k,
                "snr_db_ref": X.tolist()}
        print(self.model_params)

    def predict(self, X):
        median_g = self.model_params["median_global"]
        mad_g = self.model_params["mad_global"]
        k_g = self.model_params["k"]
        threshold = median_g - k_g * mad_g
        noisy_mask = (X < threshold)
        noisy_indices = np.where(noisy_mask)[0]
        return noisy_indices, threshold
    
    def predict_proba(self, X):
        sorted_ref = np.sort(self.model_params["snr_db_ref"])
        n = len(sorted_ref)
        
        snr_db_new = np.atleast_1d(np.array(X, dtype=float))
        prob_noises = []
        
        for x in snr_db_new:
            idx = np.searchsorted(sorted_ref, x, side='right')
            cdf_val = idx / n
            prob_noise = 1 - cdf_val
            prob_noises.append(prob_noise)

        return np.array(prob_noises).flatten()



def DNC_SN_ratio(raw, noise_threshold=0.8, optimized=False):  
    ch_names = raw.ch_names
    original_data = raw.get_data()

    def process_channel(ch_idx):
        ch_name = ch_names[ch_idx]
        raw_channel = raw.copy().load_data()
        raw_channel.pick(ch_names)
        raw_channel.info['bads'] = [ch_name]
        noised_data = original_data[ch_idx, :].copy()
        raw_channel.interpolate_bads(reset_bads=False, verbose=False)
        interpolated_data = raw_channel.get_data(picks=[ch_name])[0] # signal
        noise_data = noised_data - interpolated_data # noise
        _, snr_db = compute_snr(interpolated_data, noise_data)
        return ch_name, snr_db

    if optimized:
        results = Parallel(n_jobs=4)(
            delayed(process_channel)(ch_idx) for ch_idx in range(len(ch_names))
        )
    else:
        results = []
        for ch_idx in range(len(ch_names)):
            result = process_channel(ch_idx)
            results.append(result)
    
    snr_probabilities = {}
    bad_channels = []
    scores = []
    detector = ENDetector(k=2)
    detector.load()

    for ch_name, snr_db in results:
        proba = float(detector.predict_proba(snr_db).item())

        snr_probabilities[ch_name] = proba
        if proba > noise_threshold:
            bad_channels.append(ch_name)
            scores.append(proba)

    variable = np.array(list(snr_probabilities.values())).flatten().tolist()
    title = 'SNR Probaility topomap'
    label = 'Probability Coefficient'
    vlim=(None, None)
    fig = plot_topomap(variable, raw, title, label, vlim)
    noisy_text = "Noisy Electrodes:\n" + "\n".join(
        [f"{ch_name}: {prob:.2f}" for ch_name, prob in zip(bad_channels, scores)]
    )
    fig.text(1.1, 0.5, noisy_text, va='center', ha='left', fontsize=10)
    plt.close(fig)
    return bad_channels, scores, fig


def DNC_corr(raw, threshold_corr=0.65, threshold_perc=5):
    correlations = calculate_correlations(raw)
    variable=np.abs(np.mean(correlations, axis=0))
    title='Correlation topomap'
    label = 'Correlation Coefficient'
    vlim=(0.5, 1)
    fig = plot_topomap(variable, raw, title, label, vlim)
    plt.close(fig)

    result = np.mean(correlations, axis=0)
    p5 = np.percentile(correlations, threshold_perc, axis=(0, 1))
    artefacts_indexes = result < p5

    binary_matrix = correlations < threshold_corr
    probs = binary_matrix.mean(axis=0)
    noised_channels = ((np.array(raw.ch_names)[artefacts_indexes]).tolist())
    score = probs[artefacts_indexes]
    return noised_channels, score, fig



def DNC_electrical_distance(raw, vis=False, thr_std=3):
    
    bridged_idx, ed_matrix = compute_bridged_electrodes(raw, verbose=False)
  
    ed_matrix = ed_matrix.copy()
    picks = mne.pick_types(raw.info, eeg=True)
    tril_idx = np.tril_indices(picks.size)
    for epo_idx in range(ed_matrix.shape[0]):
        ed_matrix[epo_idx][tril_idx] = ed_matrix[epo_idx].T[tril_idx]
    channel_names = np.array([raw.ch_names[i] for i in picks])

    ed_matrix = np.nanmin(ed_matrix, axis=1)

    ed_matrix_norm = q_normalization(ed_matrix)
    elec_dists = np.median(ed_matrix_norm, axis=0)
    
    picks = mne.pick_types(raw.info, eeg=True)
    ch_names = raw.ch_names

    results = (ed_matrix_norm - elec_dists).mean(axis=0)
    probs = sigmoid(results)
    artefact_indices = results > thr_std
    variable = probs
    title = "Electrical Distance Probability topomap"
    label = 'Probability Coefficient'
    vlim = (0, 1.0)
    raw = raw.copy().pick_types(eeg=True, exclude=[])
    #...
    fig = plot_topomap(variable, raw, title, label, vlim)

    plt.close(fig)
    return (channel_names[artefact_indices]).tolist(), probs[artefact_indices], fig


def DNC_ransac(raw):  
    ransac = Ransac(verbose=False)
    epochs = mne.make_fixed_length_epochs(raw, duration=10, overlap=0.5, preload=True)
    _ = ransac.fit_transform(epochs)
    probabilities = ransac.bad_log.mean(axis=0)
    bad_channels = ransac.bad_chs_
    scores = probabilities[mne.pick_channels(raw.info['ch_names'], bad_channels)]
    variable = probabilities
    title = "RANSAC Probability topomap"
    label = 'Probability Coefficient'
    vlim = (0, 1.0)
    raw = raw.copy().pick_types(eeg=True, exclude=[])
    #...
    fig = plot_topomap(variable, raw, title, label, vlim)

    plt.close(fig)
    return bad_channels, scores, fig


def compared_spectrum(inst1, inst2, fmin=0, fmax=100):
    psd_before = inst1.compute_psd(fmin=fmin, fmax=fmax, remove_dc=False, verbose=False)
    psd_after = inst2.compute_psd(fmin=fmin, fmax=fmax, remove_dc=False, verbose=False)

    if isinstance(inst1, mne.Epochs) and isinstance(inst2, mne.Epochs):
        psd_before = psd_before.average()
        psd_after = psd_after.average()

    bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13),
            'Beta': (13, 30), 'Gamma': (30, 80)}
    
    mean_psd_before = 10*np.log10(psd_before.get_data() * 1e12)
    mean_psd_after = 10*np.log10(psd_after.get_data() * 1e12)
    freqs = psd_before.freqs
    psd_list = [mean_psd_before, mean_psd_after]
    titles = ['Spectrum before', 'Spectrum after']
    colors = ['b', 'g']

    ymin = np.floor(min(mean_psd_before.min(), mean_psd_after.min()) / 10) * 10
    ymax = np.ceil(max(mean_psd_before.max(), mean_psd_after.max()) / 10) * 10
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
  
    for idx, (mean_psd, title, color) in enumerate(zip(psd_list, titles, colors)):
        ax = axs[idx]
        y_mean = np.mean(mean_psd, axis=0)
        y_std = np.std(mean_psd, axis=0)
        
        ax.plot(freqs, y_mean, color=color)
        ax.fill_between(freqs, y_mean - y_std, y_mean + y_std, color=color, alpha=0.3)
        ax.set_title(title)
        ax.set_ylabel('PSD (dB/V/Hz)')
        ax.grid(True)
          
        band_powers = {}
        for band_name, (fmin, fmax) in bands.items():
            idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
            band_power = y_mean[idx_band].mean()
            band_powers[band_name] = band_power
                 
            ax.axvline(fmin, color='red', linestyle='--', linewidth=1)
            ax.axvline(fmax, color='red', linestyle='--', linewidth=1)
            ax.fill_betweenx([ymin, ymax], fmin, fmax, color='grey', alpha=0.05)
              
            ax.text((fmin + fmax) / 2, ymax - 2, f"{band_name}\n{band_power:.1f} dB",
                    horizontalalignment='center', verticalalignment='top', fontsize=9, zorder=4)
        
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(freqs.min(), freqs.max())
        ax.set_xticks(np.arange(freqs.min(), freqs.max()+1, 2))
        ax.set_yticks(np.arange(ymin, ymax+1, 5))
        ax.set_xlabel('Frequency (Hz)')

    plt.tight_layout()
    plt.close(fig)
    return fig
