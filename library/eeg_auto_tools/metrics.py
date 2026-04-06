# Copyright 2025 Sear Gamemode
import mne 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, normaltest, chisquare, norm
from sklearn.ensemble import IsolationForest
from scipy.signal import correlate, welch
from mne.channels import find_ch_adjacency
from .craft_events import make_RiTi_events


def get_neighbors(inst):
    adjacency, ch_names = find_ch_adjacency(info=inst.info, ch_type='eeg')
    adjacency = adjacency.toarray()
    adjacencies = {i: adjacency[i].nonzero()[0].tolist() for i in range(len(ch_names))}
    return adjacencies

def check_volt_of_epochs(epochs, reject, flat):
    epochs_cleaned = epochs.copy().drop_bad(reject=reject, flat=flat, verbose=False)
    count = 0
    rejected_epochs = []
    for i, log in enumerate(epochs_cleaned.drop_log): 
        if log==('NO_DATA',) or log==('MERGE DUPLICATE',):
            continue
        if log:
            rejected_epochs.append(count)
        count += 1
    return rejected_epochs

def calculate_SN_ratio(epochs, option='neighbours', mode='log'):
    epochs_data = epochs.get_data(copy=True, verbose=False)
    n_epochs, n_channels, n_times = epochs_data.shape
    clean_signal = np.zeros((n_epochs, n_channels, n_times))
    
    if option == 'neighbours':
        adjacencies = get_neighbors(epochs)
        for ch in range(n_channels):
            neighbor_indices = adjacencies.get(ch)
            neighbor_indices.remove(ch)
            if neighbor_indices:
                clean_signal[:, ch, :] = np.mean(epochs_data[:, neighbor_indices, :], axis=1)

    elif option == 'mean_epochs':
        clean_signal = np.mean(epochs_data, axis=0, keepdims=True).repeat(n_epochs, axis=0)
    elif option == 'median_epochs':
        clean_signal = np.median(epochs_data, axis=0, keepdims=True).repeat(n_epochs, axis=0)
    else:
        raise ValueError("Invalid option. Choose from 'neighbours', 'mean_epochs', 'median_epochs', 'mean_channels', 'median_channels'.")

    noise_signal = epochs_data - clean_signal

    if mode == 'log':
        snr = 10 * np.log10(np.mean(clean_signal ** 2, axis=2) / np.mean(noise_signal ** 2, axis=2))
    elif mode == 'linear':
        snr = np.mean(clean_signal ** 2, axis=2) / np.mean(noise_signal ** 2, axis=2)
    else:
        raise ValueError("Invalid mode. Choose 'log' or 'linear'.")
    return snr
    
def isolation_forest(epochs, mode='ep'):
    data = epochs.get_data(copy=True)
    n_trials, n_channels, n_times = data.shape
    if mode == 'ep':
        clf = IsolationForest(n_estimators=20, random_state=42, max_features=int(n_channels * n_times / 2))
        preds = clf.fit_predict(data.reshape(n_trials, -1))
        return [i for i, pred in enumerate(preds) if pred == -1]
    elif mode == 'ch':
        clf = IsolationForest(n_estimators=20, random_state=42, max_features=int(n_trials * n_times / 2))
        preds = clf.fit_predict(data.reshape(n_channels, -1))
        return [i for i, pred in enumerate(preds) if pred == -1]
    else:
        return None


def mae(y_true, y_pred):  
    return np.mean(np.abs(y_true - y_pred))

def mape(y_true, y_pred):
    eps=1e-8
    return (np.abs(y_true - y_pred)/(y_true+eps)).mean()*100

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def rmspe(y_true, y_pred):
    eps=1e-8
    return np.sqrt(np.mean(((y_true - y_pred)/(y_true+eps))**2))*100

def cos_distance(y_true, y_pred):
    cos_alpha = np.dot(y_true, y_pred)/(np.linalg.norm(y_true)*np.linalg.norm(y_pred))
    return cos_alpha

def compute_psd(signal, sfreq):
    freqs = np.fft.rfftfreq(len(signal), d=1./sfreq)
    spectrum = np.abs(np.fft.rfft(signal))**2  
    #  PSD      
    psd = spectrum / (len(signal) * sfreq)
    return freqs, psd

def calculate_statistics(signals, z_score=1.96):
    mean = np.mean(signals, axis=(0, 1))
    std = np.std(signals, axis=0)
    se = np.mean(std / np.sqrt(signals.shape[0]), axis=0)
    ci_95 = z_score * se  # 95%  
    return mean, ci_95

def rythm_metric(epochs, picks):
    brain_bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (14, 30),
        'gamma': (30, 90)
        }
    sfreq = epochs.info['sfreq']
 
    def filter_band(epochs, band):
        return epochs.copy().filter(band[0], band[1])
 
    def get_signals_and_stats(epochs, picks, band):
        filtered_epochs = filter_band(epochs, band)
        signals = filtered_epochs.pick_channels(picks).get_data(copy=True) * 1e6
        mean_signal = signals.mean(axis=(0, 1))
        mean, ci = calculate_statistics(signals)
        freqs, spectrum = compute_psd(mean_signal, sfreq)
        return mean_signal, mean, ci, freqs, spectrum
 
    results = {}
    for band_name, band in brain_bands.items():
        results[band_name] = get_signals_and_stats(epochs, picks, band)

    times = epochs.times

    fig, axs = plt.subplots(5, 2, figsize=(15, 15), sharex='col')

    colors = {
        'delta': 'purple',
        'theta': 'red',
        'alpha': 'blue',
        'beta': 'orange',
        'gamma': 'green'
    }

    for i, (band_name, (signal, mean_signal, ci, freqs, spectrum)) in enumerate(results.items()):
        label = f'{band_name.capitalize()} ({brain_bands[band_name][0]}-{brain_bands[band_name][1]} Hz)'
        color = colors[band_name]
        
        axs[i, 0].plot(times, signal, label=f'{label} SE={np.mean(ci):.2f}$\mu$V', color=color)
        axs[i, 0].set_ylabel('Amplitude ($\mu$V)')
        axs[i, 0].set_title(label)
        axs[i, 0].set_xlabel('Time (s)')
        axs[i, 0].set_xlim(-0.2, 1.0)
        axs[i, 0].fill_between(times, mean_signal - ci, mean_signal + ci, color=color, alpha=0.3)
        axs[i, 0].axvline(x=0, color='black', linestyle='--')
        
        axs[i, 1].plot(freqs, spectrum, label=f'{label} Spectrum', color=color)
        axs[i, 1].set_ylabel('Power/Hz ($\mu V^2$/Hz)')
        axs[i, 1].set_title(f'{label} Spectrum')
        axs[i, 1].set_xlabel('Frequency (Hz)')
        axs[i, 1].set_xlim(0, 90)

    for ax in axs.flat:
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()
    delta_signal = results['delta'][0]
    theta_signal = results['theta'][0]
    alpha_signal = results['alpha'][0]
    beta_signal = results['beta'][0]
    gamma_signal = results['gamma'][0]

    return delta_signal, theta_signal, alpha_signal, beta_signal, gamma_signal

def get_epochs(raw, chns):
    epochs = make_RiTi_events(raw)
    mne.viz.plot_epochs_image(epochs, picks=chns, combine="mean", vmin=-30, vmax=30)
    return epochs

def mahal(x, y, cov=None, reg=1e-20):
   cov = cov if cov else np.cov(np.vstack([x, y]), rowvar=False)
   invcov = np.linalg.pinv(cov, rcond=reg)
   return np.sqrt((x-y)@invcov@(x-y).T)

def mean_amplitude_ratio(signal1, signal2):
    return (np.mean(np.abs(signal1)) - np.mean(np.abs(signal2))) / np.mean(np.abs(signal2)) * 100


def tetta_betta_ratio(tetta, beta):
    return (tetta**2).mean()/(beta**2).mean()

def compare_epochs(inst1, inst2, picks):
    if isinstance(inst1, mne.io.brainvision.brainvision.RawBrainVision) and isinstance(inst2, mne.io.brainvision.brainvision.RawBrainVision):
        epochs1 = get_epochs(inst1, picks)
        epochs2 = get_epochs(inst2, picks)
    else:
        epochs1 = inst1
        epochs2 = inst2
    delta_signals_1, theta_signals_1, alpha_signals_1, beta_signals_1, gamma_signals_1 = rythm_metric(epochs1, picks)
    delta_signals_2, theta_signals_2, alpha_signals_2, beta_signals_2, gamma_signals_2 = rythm_metric(epochs2, picks)
    results = {'Delta':{}, 'Theta':{}, 'Alpha':{}, 'Beta':{}, 'Gamma':{}}
    for i, (signal1, signal2, band) in enumerate([(delta_signals_1, delta_signals_2, 'Delta'), (theta_signals_1, theta_signals_2, 'Theta'),
                                            (alpha_signals_1, alpha_signals_2, 'Alpha'), (beta_signals_1, beta_signals_2, 'Beta'), 
                                            (gamma_signals_1, gamma_signals_2, 'Gamma')]):
        results[band]['MAE'] = mae(signal1, signal2)
        results[band]['MAPE(%)'] = mape(signal1, signal2)
        results[band]['RMSE'] = rmse(signal1, signal2)
        results[band]['RMSPE(%)'] = rmspe(signal1, signal2)
        results[band]['CosA'] = cos_distance(signal1, signal2)
        results[band]['Mahal'] = mahal(signal1, signal2)
        results[band]['Mean-A'] = mean_amplitude_ratio(signal1, signal2)

    df_results = pd.DataFrame(results)
    
    print(df_results, f'\nAplitude~ V*1e-6')
    print('Tetta Beta Ratio', tetta_betta_ratio(theta_signals_1, beta_signals_1)/tetta_betta_ratio(theta_signals_2, beta_signals_2))

def cross_correlation_score(data1, data2, sf):
    data1 = np.mean(data1, axis=0)
    data2 = np.mean(data2, axis=0)
    corr = correlate(data1, data2)
    lags = np.arange(-len(data1) + 1, len(data2))
    lags_in_seconds = lags / sf
    plt.plot(lags_in_seconds, corr)
    plt.title(f'-')
    plt.xlabel(' (s)')
    plt.ylabel('')
    plt.show()


def test_chi_square_mormal(epochs):
    epochs_data = epochs.get_data(copy=True)
    results = []
    for epoch in epochs_data:
        pvals = []
        for channel in epoch:
            mean, std = norm.fit(channel)
            observed_freqs, bins = np.histogram(channel, bins='auto')
            expected_freqs = norm.cdf(bins[1:], mean, std) - norm.cdf(bins[:-1], mean, std)
            expected_freqs *= observed_freqs.sum() / expected_freqs.sum() 
            chi_stat, p_chi_square = chisquare(f_obs=observed_freqs, f_exp=expected_freqs)
            pvals.append(p_chi_square)
        results.append(pvals)
    results = np.array(results)
    cmap = plt.cm.RdYlGn_r 
    fig = plt.figure(figsize=(64, 8))
 
    cax2 = plt.imshow(results.T, cmap=cmap, vmin=0, vmax=0.2, aspect=9.0)
    fig.colorbar(cax2)
    plt.yticks(ticks=np.arange(results.shape[1]), labels=epochs.ch_names, fontsize=12)
    plt.title("Chi-square Test P-values", fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Channels', fontsize=12)
    plt.show()

def test_Shapiro_mormal(epochs):
    epochs_data = epochs.get_data(copy=True)  
    results = []
    for epoch in epochs_data:
        pvals = []
        for channel in epoch:
            stat, p_dagostino = shapiro(channel)
            pvals.append(p_dagostino)
        results.append(pvals)
    results = np.array(results)
    cmap = plt.cm.RdYlGn_r  
    fig = plt.figure(figsize=(64, 8))
    cax2 = plt.imshow(results.T, cmap=cmap, vmin=0, vmax=0.2, aspect=9.0)
    fig.colorbar(cax2)
    plt.yticks(ticks=np.arange(results.shape[1]), labels=epochs.ch_names, fontsize=12)
    plt.title("D'Shapiro Test P-values", fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Channels', fontsize=12)
    plt.show()

def test_Dagostino_mormal(epochs):
    epochs_data = epochs.get_data(copy=True)
    results = []
    for epoch in epochs_data:
        pvals = []
        for channel in epoch:
            stat, p_dagostino = normaltest(channel)
            pvals.append(p_dagostino)
        results.append(pvals)
    results = np.array(results)
    cmap = plt.cm.RdYlGn_r 
    fig = plt.figure(figsize=(64, 8))
    cax2 = plt.imshow(results.T, cmap=cmap, vmin=0, vmax=0.2, aspect=9.0)
    fig.colorbar(cax2)
    plt.yticks(ticks=np.arange(results.shape[1]), labels=epochs.ch_names, fontsize=12)
    plt.title("D'Agostino-K^2 Test P-values", fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Channels', fontsize=12)
    plt.show()
