# Copyright 2025 Sear Gamemode
import mne 
import numpy as np 


def calculate_tfr(epochs, method, fmin=2, fmax=40, n_coef=3.5, average=True, return_itc=False):
    sfreq=epochs.info['sfreq']
    num=int(n_coef*(fmax-fmin))
    freqs = np.logspace(*np.log10([fmin, fmax]), num=num)  
    n_cycles = (freqs*np.pi*(len(epochs.times)-1))/(5*sfreq)
    if return_itc:
        tfr, itc = epochs.compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, average=average, 
                                      return_itc=True, n_jobs=7, use_fft=False, zero_mean=True)
        return tfr, itc, freqs
    else:
        tfr = epochs.compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, average=average, 
                                 return_itc=False, n_jobs=7, use_fft=False, zero_mean=True)
        return tfr, freqs
