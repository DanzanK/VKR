# Copyright 2025 Sear Gamemode
import emd
import mne
import numpy as np 
from scipy.signal import stft, istft
from sklearn.decomposition import PCA
from itertools import combinations


def info_suppression(epochs, n_components=10):
    n_epochs, n_channels, n_times = epochs.get_data().shape
    reshaped_data = epochs.get_data().reshape(-1, n_times)
    fft_data = np.fft.fftshift(reshaped_data)
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(fft_data)
    inverse_pca_data = pca.inverse_transform(pca_data)
    inverse_fft_data = np.fft.ifftshift(inverse_pca_data)
    filtered_data = np.array(inverse_fft_data).reshape(n_epochs, n_channels, n_times)
    filtered_epochs = mne.EpochsArray(filtered_data, epochs.info, tmin=epochs.tmin)
    return filtered_epochs


def spectr_filter(epochs):
    data = epochs.get_data()
    frequencies, times, Z = stft(data)
    power = np.abs(Z)**2
    threshold = np.percentile(power, 99.99)   
    mask = power < threshold
    clean_Z = Z * mask
    _, clean_data = istft(clean_Z)
    clean_epochs = epochs.copy()._data = clean_data
    return clean_epochs


def emd_decompose(epochs, evoked):
    EEG = epochs.copy().get_data() * 10**6
    ERP = evoked.copy().get_data() * 10**6
    n_epochs, n_channels, n_times = EEG.shape
    sfreq = int(epochs.info['sfreq'])

    hilbert_erp = []
    ch_names = epochs.ch_names

    for pick_idx, ch_name in enumerate(ch_names):
        filtered_electrode_data = []
        max_corr_data = []
        for ep in range(n_epochs):
            data = EEG[ep][pick_idx]
            erp_data = ERP[pick_idx]
            imfs = emd.sift.mask_sift(data, max_imfs=5)

            def find_optimal_imfs(imfs, erp_data):
                n_imfs = imfs.shape[1]
                max_corr = -1
                best_combination = None
                for r in range(1, n_imfs + 1):
                    for combination in combinations(range(n_imfs), r):
                        combined_imf = np.sum(imfs[:, combination], axis=1)
                        corr = np.corrcoef(erp_data, combined_imf)[0, 1]
                        if corr > max_corr:
                            max_corr = corr
                            best_combination = combination
                return best_combination, max_corr

            optimal_imfs, max_correlation = find_optimal_imfs(imfs, erp_data)
            filtered_signal = np.sum(imfs[:, optimal_imfs], axis=1)
            filtered_electrode_data.append(filtered_signal)
            max_corr_data.append(max_correlation)
  
        std_deviation = np.std(filtered_electrode_data, axis=0)
        standard_error = std_deviation / np.sqrt(n_epochs)
        mean_standard_error = np.mean(standard_error)
        print(f'Electrode {ch_names[pick_idx]}')
        print(f"\tMean Standard Error={mean_standard_error:.3f}")
        print(f'\tCorrelation: mean={np.mean(max_corr_data):.3f}, std={np.std(max_corr_data, ddof=1):.3f}')
        hilbert_erp.append(filtered_electrode_data)
     
    hilbert_erp = np.array(hilbert_erp)
    hilbert_erp = hilbert_erp.transpose(1, 0, 2) / 10**6 
    epochs_denoised = mne.EpochsArray(hilbert_erp, epochs.info, event=epochs.events, tmin=epochs.tmin, event_id=epochs.event_id,
                                      baseline=epochs.baseline, tmax=epochs.tmax, raw_sfreq=epochs.sfreq, verbose=False)
    return epochs_denoised
