# Copyright 2025 Sear Gamemode
import mne 
import torch
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .metrics import calculate_SN_ratio

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, latent_dim)  #  
        self.fc_logvar = nn.Linear(64, latent_dim)  #  

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = self.fc3(z)
        return z

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mu, logvar, z

def loss_function(reconstructed_x, x, mu, logvar, z, alpha=0, beta=0.5):
    recon_loss = F.mse_loss(reconstructed_x, x, reduction='mean')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    infomax_loss = alpha * (z.pow(2).mean(dim=0) - 1).abs().sum()
    mae_loss = F.l1_loss(reconstructed_x, x, reduction='mean')
    return recon_loss + kld_loss + infomax_loss + beta*mae_loss

class Feature_Extractor():
    def __init__(self, epochs, latent_dim=5, lr=1e-3, alpha=1.0):
        # 
        self.alpha=alpha  #   infomax
        self.lr = lr
        input_dim = epochs.get_data().shape[1] * epochs.get_data().shape[2]  #      
        self.latent_dim = latent_dim #   
        #    
        self.vae = VAE(input_dim=input_dim, latent_dim=latent_dim)
        self.optimizer = optim.Adam(self.vae.parameters(), lr=self.lr)
        #  
        epochs_data = torch.tensor(epochs.get_data(), dtype=torch.float32)
        self.epochs_data = epochs_data.view(epochs_data.shape[0], -1)  #   

    def train(self,):
        n_epochs = 100
        for epoch in range(n_epochs):
            self.vae.train()
            self.optimizer.zero_grad()
            reconstructed_x, mu, logvar, z = self.vae(self.epochs_data)
            loss = loss_function(reconstructed_x, self.epochs_data, mu, logvar, z, alpha=self.alpha)
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f" [{epoch+1}/{n_epochs}], : {loss.item():.4f}")
    
    def extract(self, epochs):
        epochs_data = torch.tensor(epochs.get_data(), dtype=torch.float32)
        epochs_data = epochs_data.view(epochs_data.shape[0], -1) 
        self.vae.eval()
        with torch.no_grad():
            mu, logvar = self.vae.encoder(epochs_data)
            z = self.vae.reparameterize(mu, logvar)  
        return z

def get_VAE_features(epochs, vae):
    z = vae.extract(epochs)
    return z

def snr(epochs):
    snr_matrix = calculate_SN_ratio(epochs)
    return {'SNR': snr_matrix}

def mean(epochs, mode='zero'):
    data = epochs.get_data()
    if mode == 'zero':
        tmin = epochs.tmin
        sfreq = epochs.info['sfreq']
        result = data[:, :, int(abs(tmin*sfreq)):].mean(axis=2)
    else:
        result = data.mean(axis=2)
    return {"Mean": result}

def std(epochs, mode='zero'):
    data = epochs.get_data()
    if mode == 'zero':
        tmin = epochs.tmin
        sfreq = epochs.info['sfreq']
        result = data[:, :, int(abs(tmin*sfreq)):].std(axis=2)
    else:
        result = data.std(axis=2)
    return {"Std": result}

def median(epochs, mode='zero'):
    data = epochs.get_data()
    if mode == 'zero':
        tmin = epochs.tmin
        sfreq = epochs.info['sfreq']
        result = np.median(data[:, :, int(abs(tmin*sfreq)):], axis=2)
    else:
        result = data.std(axis=2)
    return {"Std": result}

def bands_spectrum_power(epochs):
    results = {}
    bands =  {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (14, 30), 'gamma': (30, 90)}
    data = epochs.get_data()
    sfreq = epochs.info['sfreq']
    for band_name, (fmin, fmax) in bands.items():
        psds, _ = mne.time_frequency.psd_array_multitaper(data, sfreq=sfreq, fmin=fmin, fmax=fmax, verbose=False)
        band_power = np.mean(psds, axis=2)
        results[band_name] = band_power
    return results

def derivative(epochs):
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    deriv = (data[1:] - data[:-1])/sfreq
    return {"1-Deriv": deriv}

def standard_error_of_the_mean(epochs):
    data = epochs.get_data()
    n_trials = data.shape[0]
    sem = np.std(data, axis=0) / np.sqrt(n_trials)
    result = {'SEM':sem}
    return result

def latency_and_peaks(epochs, component):
    peak_windows = {
        'P1': (0.08, 0.13),
        'N1': (0.10, 0.15),
        'P2': (0.15, 0.25),
        'N2': (0.20, 0.30),
        'P3': (0.30, 0.50)
    }
    if component not in peak_windows:
        raise ValueError(f" : {component}.  : {list(peak_windows.keys())}")
    
    tmin, tmax = peak_windows[component]
    evoked = epochs.average()
    
    results = {
        f"{component}_latency": [],
        f"{component}_amplitude": []
    }
    for ch_name in evoked.ch_names:
        latency, amplitude = evoked.get_peak(
            tmin=tmin, tmax=tmax, mode='pos' if 'P' in component else 'neg', 
            ch_type='eeg', picks=[ch_name], return_amplitude=True
        )
        
    results[f"{component}_latency"].append(latency)
    results[f"{component}_amplitude"].append(amplitude)

    return results


def feature_selector(epochs, list_features=[]):
    values = {}
    for metric in list_features:
        values[metric.__name__] = metric(epochs)
    return values


