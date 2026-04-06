# Copyright 2025 Sear Gamemode
import seaborn as sns
import matplotlib.pyplot as plt
import mne 
import pandas as pd 
import os 
import numpy as np 
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D


def get_clusters(clusters_stats, ch_names, ch_pos, correlation_values):
    figs = []

    # --- sanitize inputs: remove channels with NaN/Inf positions ---
    pos = np.asarray(ch_pos, dtype=float)
    if pos.ndim != 2 or pos.shape[1] < 2:
        # некорректные координаты — просто не рисуем топомап
        pos = None
    else:
        good = np.isfinite(pos[:, 0]) & np.isfinite(pos[:, 1])
        if good.sum() >= 3:
            pos = pos[good, :2]
            ch_names = [n for n, g in zip(ch_names, good) if g]
        else:
            pos = None  # слишком мало точек для Delaunay

    # Дальше строим data только для отфильтрованных ch_names
    data = np.full(len(ch_names), np.nan)
    mask = np.zeros(len(ch_names), dtype=bool)
    ch_name_to_idx = {ch: idx for idx, ch in enumerate(ch_names)}

    clusters = [cluster_stat['channels'] for cluster_stat in clusters_stats]

    for cluster_idx, cluster in enumerate(clusters):
        cluster_value = cluster_idx + 1
        for ch_name in cluster:
            if ch_name in ch_name_to_idx:
                idx = ch_name_to_idx[ch_name]
                data[idx] = cluster_value
                mask[idx] = True

    N_clusters = len(clusters)
    cmap = plt.get_cmap('tab20', max(N_clusters, 1))
    cluster_colors = cmap(np.arange(max(N_clusters, 1)) % 20)
    cluster_cmap = ListedColormap(cluster_colors)

    # --- Topomap (optional) ---
    if pos is not None and N_clusters > 0:
        fig, ax = plt.subplots(figsize=(8, 8))
        ch_pos = np.asarray(ch_pos)
        valid = np.isfinite(ch_pos).all(axis=1)

        valid = np.isfinite(ch_pos).all(axis=1)
        data_v = data[valid]
        pos_v = ch_pos[valid]
        names_v = [n for i, n in enumerate(ch_names) if valid[i]]

        fig, ax = plt.subplots(figsize=(8, 8))
        im, cn = mne.viz.plot_topomap(
            data_v,
            pos=pos_v,
            axes=ax,
            cmap=cluster_cmap,
            names=names_v,
            show=False,
            vlim=(0.5, N_clusters + 0.5),
            contours=0,
            image_interp='nearest',
            extrapolate='head',
            outlines='head',
            sphere=(0., 0., 0., 0.095)
        )


        legend_elements = []
        for idx, cluster_stat in enumerate(clusters_stats):
            avg_corr = cluster_stat['avg_corr']
            max_corr = cluster_stat['max_corr']
            min_corr = cluster_stat['min_corr']
            color = cluster_colors[idx % 20]
            label = f'Cluster {idx+1}: Correlation (avg={avg_corr:.3f}, max={max_corr:.3f}, min={min_corr:.3f})'
            legend_elements.append(Line2D(
                [0], [0],
                marker='s',
                color='w',
                label=label,
                markerfacecolor=color,
                markersize=10
            ))

        ax.set_title('Bridged Clusters')
        ax.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.,
            fontsize=8
        )
        figs.append(fig)
        plt.close(fig)

    # --- Histogram (оставляем всегда, даже если topomap пропустили) ---
    fig = plt.figure(figsize=(8, 6))
    plt.hist(np.abs(np.asarray(correlation_values, dtype=float)), bins=100, edgecolor='black')
    plt.title('Histogram of Pairwise Electrode Correlations')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Number of Electrode Pairs')
    figs.append(fig)
    plt.close(fig)

    return figs



def ASSR_erp_plot(ERP_c, ERP_n, noised_c_idx, noised_n_idx, roi, comp, baseline=0.2, save=None, vis=False):
    electrodes_to_plot = roi[comp]
    def get_picks_and_data(ERP, electrodes):
        picks = mne.pick_channels(ERP.epochs.info['ch_names'], include=electrodes)
        epochs = ERP.epochs.pick(picks)
        return epochs 
    
    epochs_c = get_picks_and_data(ERP_c, electrodes_to_plot)
    epochs_n = get_picks_and_data(ERP_n, electrodes_to_plot)

    n_epochs_c = len(epochs_c)
    n_epochs_n = len(epochs_n)

    sf = int(epochs_c.info['sfreq'])
    fig_erp = plt.figure(figsize=(16, 12))
    gs_erp = GridSpec(4, 2, figure=fig_erp, hspace=0.7)
    fig_erp.suptitle(f'Event-Related Potencial (ERP), \nCongruent epochs:{n_epochs_c}/{n_epochs_c+len(noised_c_idx)}, Incongruent epochs:{n_epochs_n}/{n_epochs_n+len(noised_n_idx)}')
    t = epochs_c.times
    for i, electrode in enumerate(electrodes_to_plot):
        row, col = divmod(i, 2)
        ax_erp = fig_erp.add_subplot(gs_erp[row, col])
        data_c = epochs_c.get_data(picks=electrode, verbose=False)*1e6
        data_n = epochs_n.get_data(picks=electrode, verbose=False)*1e6
        mean_data_c = np.mean(data_c, axis=0).squeeze()
        mean_data_n = np.mean(data_n, axis=0).squeeze()
        SE_c = np.std(data_c, axis=0, ddof=1).squeeze()/ np.sqrt(data_c.shape[0])
        SE_n = np.std(data_n, axis=0, ddof=1).squeeze()/ np.sqrt(data_n.shape[0])
        
        ax_erp.plot(t, mean_data_c, label='con', color='blue')
        ax_erp.plot(t, mean_data_n, label='incon', color='red')
        ax_erp.fill_between(t, mean_data_c - SE_c, mean_data_c + SE_c, color='blue', alpha=0.3)
        ax_erp.fill_between(t, mean_data_n - SE_n, mean_data_n + SE_n, color='red', alpha=0.3)
        ax_erp.set_ylim([-20, 20])
        ax_erp.set_title(f'{electrode} electrode',  fontsize=10)
        ax_erp.set_xlabel('Time, sec', fontsize=10)
        ax_erp.set_ylabel('Amplitude, $\mu$V', fontsize=10)
        ax_erp.axhline(0, color='black', linewidth=1)
        ax_erp.axvline(0, color='black', linewidth=1)
        ax_erp.grid(True)
        ax_erp.legend(fontsize=8)
        ax_erp.set_box_aspect(0.2) 

    electrode_c_data = [epochs_c.average(picks=electrode, method='mean').data *1e6 for electrode in electrodes_to_plot]
    electrode_n_data = [epochs_n.average(picks=electrode, method='mean').data *1e6 for electrode in electrodes_to_plot]
    mean_data_c_all = np.mean(electrode_c_data, axis=0).squeeze()
    mean_data_n_all = np.mean(electrode_n_data, axis=0).squeeze()

    ax_erp_avg = fig_erp.add_subplot(gs_erp[3, 0])
    ax_erp_avg.plot(t, mean_data_c_all, label='con', color='blue', linestyle='--')
    ax_erp_avg.plot(t, mean_data_n_all, label='incon', color='red', linestyle='--')
    ax_erp_avg.set_ylim([-20, 20])
    ax_erp_avg.set_title(f'Average over epochs-averaged electrodes', fontsize=10)
    ax_erp_avg.set_xlabel('Time, sec', fontsize=10)
    ax_erp_avg.set_ylabel('Amplitude, $\mu$V', fontsize=10)
    ax_erp_avg.axhline(0, color='black', linewidth=1)
    ax_erp_avg.axvline(0, color='black', linewidth=1)
    ax_erp_avg.grid(True)
    ax_erp_avg.legend(fontsize=8)
    ax_erp_avg.set_box_aspect(0.2)

    epochs_c_data = np.mean(epochs_c.get_data(picks=electrodes_to_plot, verbose=False), axis=1)*1e6
    epochs_n_data = np.mean(epochs_n.get_data(picks=electrodes_to_plot, verbose=False), axis=1)*1e6
    mean_data_c_all_epochs = np.mean(epochs_c_data, axis=0)
    mean_data_n_all_epochs = np.mean(epochs_n_data, axis=0)
    SD_c_all_epochs = np.std(epochs_c_data, axis=0, ddof=1)/ np.sqrt(epochs_c_data.shape[0])
    SD_n_all_epochs = np.std(epochs_n_data, axis=0, ddof=1)/ np.sqrt(epochs_n_data.shape[0])
    t_epochs = np.arange(-baseline, len(mean_data_c_all_epochs) / sf - baseline, 1 / sf)

    ax_erp_avg_epochs = fig_erp.add_subplot(gs_erp[3, 1])
    ax_erp_avg_epochs.plot(t_epochs, mean_data_c_all_epochs, label='con', color='blue', linestyle='--')
    ax_erp_avg_epochs.plot(t_epochs, mean_data_n_all_epochs, label='incon', color='red', linestyle='--')
    ax_erp_avg_epochs.fill_between(t_epochs, mean_data_c_all_epochs - SD_c_all_epochs, mean_data_c_all_epochs + SD_c_all_epochs, color='blue', alpha=0.3)
    ax_erp_avg_epochs.fill_between(t_epochs, mean_data_n_all_epochs - SD_n_all_epochs, mean_data_n_all_epochs + SD_n_all_epochs, color='red', alpha=0.3)
    ax_erp_avg_epochs.set_ylim([-20, 20])
    ax_erp_avg_epochs.set_title(f'Average over electrodes and epochs', fontsize=10)
    ax_erp_avg_epochs.set_xlabel('Time, sec', fontsize=10)
    ax_erp_avg_epochs.set_ylabel('Amplitude, $\mu$V', fontsize=10)
    ax_erp_avg_epochs.axhline(0, color='black', linewidth=1)
    ax_erp_avg_epochs.axvline(0, color='black', linewidth=1)
    ax_erp_avg_epochs.grid(True)
    ax_erp_avg_epochs.legend(fontsize=8)
    ax_erp_avg_epochs.set_box_aspect(0.2)
    plt.show()
    if not vis:
        plt.close(fig_erp)
    if save:
        fig_erp.savefig(save)

def ASSR_psd_plot(ERP_c, ERP_n, noised_c_idx, noised_n_idx, roi, comp, save=None, vis=False):
    electrodes_to_plot = roi[comp]
    def get_picks_and_data(ERP, electrodes):
        picks = mne.pick_channels(ERP.epochs.info['ch_names'], include=electrodes)
        epochs = ERP.epochs.pick(picks)
        return epochs 
    
    epochs_c = get_picks_and_data(ERP_c, electrodes_to_plot)
    epochs_n = get_picks_and_data(ERP_n, electrodes_to_plot)

    sf = int(epochs_c.info['sfreq'])
    fig_psd = plt.figure(figsize=(16, 12))
    gs_psd = GridSpec(4, 2, figure=fig_psd, hspace=0.7)
    fig_psd.suptitle('PSD')
    
    for i, electrode in enumerate(electrodes_to_plot):
        row, col = divmod(i, 2)
        ax_psd = fig_psd.add_subplot(gs_psd[row, col])

        mean_data_c = np.mean(epochs_c.average(picks=electrode, method='mean').data, axis=0)
        mean_data_n = np.mean(epochs_n.average(picks=electrode, method='mean').data, axis=0)
        
        psd_c, freqs_c = mne.time_frequency.psd_array_welch(mean_data_c, sfreq=sf, fmin=0, fmax=40, n_fft=sf, verbose=False)
        psd_n, freqs_n = mne.time_frequency.psd_array_welch(mean_data_n, sfreq=sf, fmin=0, fmax=40, n_fft=sf, verbose=False)

        ax_psd.plot(freqs_c, 10 * np.log10(psd_c), label='con', color='blue')
        ax_psd.plot(freqs_n, 10 * np.log10(psd_n), label='n-con', color='red')
        ax_psd.set_title(f'{electrode} electrode', fontsize=10)
        ax_psd.set_xlabel('Frequency (Hz)', fontsize=10)
        ax_psd.set_ylabel('Power (dB)', fontsize=10)
        #ax_psd.axhline(-100, color='black', linewidth=1)
        ax_psd.axvline(0, color='black', linewidth=1)
        ax_psd.grid(True)
        ax_psd.legend(fontsize=8)
        ax_psd.set_box_aspect(0.2)  # ,      

    electrode_c_data = [epochs_c.average(picks=electrode, method='mean').data *1e6 for electrode in electrodes_to_plot]
    electrode_n_data = [epochs_n.average(picks=electrode, method='mean').data *1e6 for electrode in electrodes_to_plot]
    mean_data_c_all = np.mean(electrode_c_data, axis=0).squeeze()
    mean_data_n_all = np.mean(electrode_n_data, axis=0).squeeze()

    #    PSD
    psd_c_all, freqs_c_all = mne.time_frequency.psd_array_welch(mean_data_c_all, sfreq=sf, fmin=0, fmax=40, n_fft=sf, verbose=False)
    psd_n_all, freqs_n_all = mne.time_frequency.psd_array_welch(mean_data_n_all, sfreq=sf, fmin=0, fmax=40, n_fft=sf, verbose=False)

    ax_psd_avg = fig_psd.add_subplot(gs_psd[3, :])
    ax_psd_avg.plot(freqs_c_all, 10 * np.log10(psd_c_all), label='con', color='blue', linestyle='--')
    ax_psd_avg.plot(freqs_n_all, 10 * np.log10(psd_n_all), label='n-con', color='red', linestyle='--')
    ax_psd_avg.set_title(f'Average over electrodes', fontsize=10)
    ax_psd_avg.set_xlabel('Frequency (Hz)', fontsize=10)
    ax_psd_avg.set_ylabel('Power (dB)', fontsize=10)
    #ax_psd_avg.axhline(-100, color='black', linewidth=1)
    ax_psd_avg.axvline(0, color='black', linewidth=1)
    ax_psd_avg.grid(True)
    ax_psd_avg.legend(fontsize=8)
    ax_psd_avg.set_box_aspect(0.1)
    plt.show()
    if not vis:
        plt.close(fig_psd)
    if save:
        fig_psd.savefig(save)


def snr_plot(snr_matrix, ch_names):
    snr_fig, ax = plt.subplots(figsize=(8, 6))
    ep_ind = np.arange(0, snr_matrix.shape[0], 1)
    ch_ind = np.arange(0, snr_matrix.shape[1], 1)
    xticklabels = np.array(ep_ind)[::max(1, len(ep_ind) // 10)]
    yticklabels = np.array(ch_names)[ch_ind]
    
    sns.heatmap(snr_matrix.T, cmap='magma', xticklabels=xticklabels, yticklabels=yticklabels,
                vmin=-30, vmax=-10, ax=ax)
    
    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_xticklabels(xticklabels)
    ax.set_title('SNR Heatmap')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Channels')
    ax.tick_params(axis='y', labelsize=8)
    plt.close(snr_fig)

    snr_values = snr_matrix.flatten()
    hist_fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(snr_values, bins=30, color='blue', edgecolor='black', ax=ax, kde=True, stat='density')

    ax.set_title('SNR Value Frequency Histogram')
    ax.set_xlabel('SNR Value')
    ax.set_ylabel('Probability')
    ax.grid(axis='y', alpha=0.75)
    ax.set_xlim(-30, 0)
    plt.close(hist_fig)
    return snr_fig, hist_fig


def compared_snr(snr_matrices, names):
    kde_fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("husl", len(snr_matrices))
    for snr_matrix, name, color in zip(snr_matrices, names, colors):
        snr_values = snr_matrix.flatten()
        mean_snr = np.median(snr_values)
        label_with_mean = f"{name} (Median: {mean_snr:.2f})"
        kde = sns.kdeplot(snr_values, ax=ax, color=color, fill=True, bw_adjust=0.5, label=label_with_mean, alpha=0.5)
        
        color = kde.collections[-1].get_facecolor()[0]  #    RGBA
        ax.axvline(mean_snr, linestyle='--', color=color, linewidth=1.5)
    ax.set_title('SNR KDE')
    ax.set_xlabel('SNR Value')
    ax.set_ylabel('Density')
    ax.grid(axis='y', alpha=0.75)
    ax.set_xlim(-35, 0)
    ax.set_ylim(0, 0.15)
    ax.grid()
    ax.legend(title='SNR Matrices', loc='upper right')
    plt.close(kde_fig)
    return kde_fig


def bridge_save(raw, correlation_matrix, bridged_idx, ed_matrix, saving_dir=None, vis=False):
    channels = raw.ch_names
    bridge_fig = mne.viz.plot_bridged_electrodes(raw.info, bridged_idx, ed_matrix)
    fig_corr = plt.figure(figsize=(8, 6))
    ax = sns.heatmap(correlation_matrix, cmap='viridis', xticklabels=channels, yticklabels=channels) #np.mean(ed_matrix, axis=0)
    ax.set_xticklabels(channels, rotation=90, fontsize=7)
    ax.set_yticklabels(channels, rotation=0, fontsize=7)
    plt.title('Correlation of Channel Connections (R)')
    plt.xlabel('Channels')
    plt.ylabel('Channels')
    if saving_dir:
        plt.savefig(os.path.join(saving_dir, f'orrelation_electrodes.png'))
        bridge_fig.savefig(os.path.join(saving_dir, f'bridged_electrodes.png'))
    if not vis:
        plt.close(fig_corr)
        plt.close(bridge_fig)


def components_save(raw, ica, exclude_idx, labels, probas, saving_dir=None, vis=False):
    duration = raw.times[-1]*1000
    start_time = 0 
    stop_time = min(10*1000, duration)  
    ica_fig = ica.plot_components(title='ICA Components', show=vis)
    if isinstance(raw, mne.io.Raw) or isinstance(raw,mne.io.brainvision.brainvision.RawBrainVision):
        overlay_fig = ica.plot_overlay(raw, title='Overlay', start=start_time, stop=stop_time, 
                                    exclude=exclude_idx, show=vis)
    else:
        overlay_fig = ica.plot_overlay(raw.average(), title='Overlay', start=start_time, stop=stop_time, 
                                    exclude=exclude_idx, show=vis)
    if saving_dir:
        ica_fig.savefig(os.path.join(saving_dir, 'ica_components.png'))
        overlay_fig.savefig(os.path.join(saving_dir, 'overlay.png'))
    if not vis:
        plt.close(ica_fig)
        plt.close(overlay_fig)
    for idx in exclude_idx:
        ica_exclude_fig = ica.plot_properties(raw, picks=idx, show=vis, verbose=False)[0]
        if saving_dir:
            if labels is None:
                ica_exclude_fig.savefig(os.path.join(saving_dir, f'ica_exclude_components_{idx}_label:hand.png'))
            else:
                ica_exclude_fig.savefig(os.path.join(saving_dir, f'ica_exclude_components_{idx}_{labels[idx]}_{probas[idx]}.png'))
        if not vis:
            plt.close(ica_exclude_fig)


def event_saver(proc_counter, mind_counter, saving_dir=None, vis=False):
    df = pd.read_csv('tools/ASSR.csv', index_col=None)
    combined_dict = {**proc_counter, **mind_counter}
    combined_dict = {key.replace('Stimulus/', ''): value for key, value in combined_dict.items()}
    df.columns = df.columns.str.replace(' ', '')
    new_row = {column: combined_dict[column] for column in df.columns}
    new_row_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_row_df], ignore_index=True)
    target_row = df.iloc[0].copy()
    target_row_numeric = pd.to_numeric(target_row, errors='coerce')
    new_row_numeric = pd.to_numeric(pd.Series(new_row), errors='coerce')
    difference_row = target_row_numeric - new_row_numeric
    difference_row_df = pd.DataFrame([difference_row])
    df = pd.concat([df, difference_row_df], ignore_index=True)

    df.insert(0, 'Cond', ['Target', 'Real', 'Diff'])
    if saving_dir:
        df.to_csv(os.path.join(saving_dir, 'stimulus_counts.csv'), index=False)
    fig_table, ax = plt.subplots(figsize=(12, 2))  
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title('  ', fontsize=14, pad=1)
    if saving_dir:
        fig_table.savefig(os.path.join(saving_dir, 'stimulus_counts.png'), bbox_inches='tight', pad_inches=0.1)
    if not vis:
        plt.close(fig_table)


def tfr_plot(tfr, itc, picks, fmin, fmax, vmin=None, vmax=None, baseline=(None, 0), combine='mean', 
             baseline_mode='logratio', saving_dir=None, vis=False, 
             tfr_title='Event Related Spectral Perturbation (ERSP)',
             itc_title='Inter-Trial Coherence (ITC)'):  
    tfr_fig = tfr.plot(title=tfr_title, fmin=fmin, fmax=fmax, vlim=(vmin,vmax), 
             picks=picks, combine=combine, baseline=baseline, mode=baseline_mode)[0] 
    itc_fig = itc.plot(title=itc_title, fmin=fmin, fmax=fmax, 
             picks=picks, combine=combine, baseline=baseline, mode=baseline_mode)[0]
    if not vis:
        plt.close(tfr_fig)
        plt.close(itc_fig)
    if saving_dir:
        tfr_fig.savefig(os.path.join(saving_dir, f'ERSP.png'))
        itc_fig.savefig(os.path.join(saving_dir, f'ITC.png'))


def plot_noise_data(raw, noisy_channel_flags, n_channels, threshold_noise, integral_curve):
    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(1, 3, width_ratios=[5, 1, 3], figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    sns.heatmap(noisy_channel_flags, vmin=0, vmax=1, cmap='viridis', 
                cbar_kws={'label': '  PSD'}, ax=ax0)
    ax0.set_xlabel('')
    ax0.set_ylabel('')
    ax0.set_title('   PSD    ')
    ax0.set_yticks(np.arange(n_channels) + 0.5)
    ax0.set_yticklabels(raw.ch_names, rotation=0)
    
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    ax1.plot(integral_curve, np.arange(n_channels)+0.5, color='blue', label='  ')
    ax1.axvline(threshold_noise, color='red', linestyle='--', label=f' - {threshold_noise}')
    ax1.set_xlim((0, 1))
    ax1.set_xlabel(' ')
    ax1.set_title(' ')
    ax1.legend(loc='right', fontsize='small')
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[0, 2])  #    
    pos = np.array([raw.info['chs'][i]['loc'][:2] for i in range(raw.info['nchan'])], dtype=float)
    good = np.isfinite(pos).all(axis=1)

    if good.sum() >= 3:
        pos_g = pos[good]
        names_g = [ch for ch, g in zip(raw.ch_names, good) if g]
        vals_g = np.asarray(integral_curve, dtype=float)[good]

        mne.viz.plot_topomap(
            vals_g, pos_g, axes=ax2, show=False, cmap='RdBu_r', vlim=(0, 1),
            names=names_g
        )
    else:
        ax2.text(0.5, 0.5, "Not enough valid channel positions\nfor topomap", ha="center", va="center")

    ax2.set_title('  ')
    plt.tight_layout()
    plt.show()


def plot_topomap(coherence, band_name, info, title):
    coherence_mean = coherence.mean(axis=1)
    mask = np.zeros((len(info['ch_names']), len(info['ch_names'])), dtype=bool)
    for i in range(len(info['ch_names'])):
        for j in range(i+1, len(info['ch_names'])):
            mask[i, j] = True
    mne.viz.plot_topomap(coherence_mean, info, mask=mask, show=False)
    plt.title(title)
    plt.show()


def char_plot(spec_con, epochs, method, mode, cwt_freqs):
    con = spec_con.get_data()
    if np.iscomplexobj(con):
        con = np.abs(con)
    con_mean = con.mean(axis=0)
    times = epochs.times
    plt.figure(figsize=(12, 6))
    plt.imshow(con_mean, aspect=1/40, cmap='inferno', extent=[times[0], times[-1], cwt_freqs[0], cwt_freqs[-1]], origin='lower') #vmin=0, vmax=1
    plt.colorbar(label=f' {method} {mode}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'  {method}   CWT Morlet')
    plt.show()
    plt.savefig(f'{method} {mode}.png')


def plot_circular_connectivity(con, epochs, threshold=0.5, title='Circular Connectivity Diagram'):
    con = np.mean(np.abs(con), axis=2)  
    con[con < threshold] = 0  
    node_names = epochs.ch_names
    mne.viz.plot_connectivity_circle(con, node_names, title=title, vmin=0, vmax=1, colormap='inferno', show=True)
