# Copyright 2025 Sear Gamemode
from .transforms import Interpolate, AutoICA, Sequence, Rereference, Resample, CheckEvents, Cropping, FilterBandpass
from .transforms import SetMontage, BadChannelsDetector, Raw2Epoch, BadEpochsDetector, DetrendEpochs, BaselineEpochs, ChannelSelector
from .scenarious import get_file_info
import mne 
import numpy as np 


class EpochsAnalysier():
    def __init__(self, scenarious_name, stimulus_list, roi, tmin, tmax, 
                 baseline=(None, 0), detrend_type='linear'):
        self.stimulus_list = stimulus_list
        self.scenarious_name = scenarious_name
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        self.roi = roi
        self.detrend_type = detrend_type
        self.epochs_store = []
        self.pathes = {}

    def compute(self, record, epoch_path, progress_bar=None):
        self.epoch_path = epoch_path
        mne.set_log_level('ERROR')
        raw = mne.io.read_raw(record, preload=True)
        mne.set_log_level('WARNING')
        self.precompute_epochs = Sequence(
                        checker = CheckEvents(scenarious_name=self.scenarious_name, report=True),
                        raw2epoch = Raw2Epoch(self.stimulus_list, self.scenarious_name, tmin=self.tmin, tmax=self.tmax, baseline=None),
                        detector = BadEpochsDetector(self.roi),
                        baseliner = BaselineEpochs(baseline=self.baseline),
                        detrender = DetrendEpochs(detrend_type=self.detrend_type),
                        )
        self.precompute_epochs(raw, progress_bar)
        self.pathes['detector'] = self.precompute_epochs.detector.save_report(self.epoch_path, pref='detector_')
        self.pathes['detrender'] = self.precompute_epochs.detrender.save_report(self.epoch_path, pref='detrender_')
        self.pathes['baseliner'] = self.precompute_epochs.baseliner.save_report(self.epoch_path, pref='baseliner_')

        return [self.precompute_epochs.insts[1], self.precompute_epochs.insts[-2], self.precompute_epochs.insts[-1]]
    
    def get_report(self):
        report_images = {
            'detector_SNR_hist_image': self.pathes['detector'][0],
            'detrend_SNR_hist_image': self.pathes['detrender'][0],
            'baseline_SNR_hist_image': self.pathes['baseliner'][0],
            'detector_spectrum_hist_image': self.pathes['detector'][1],
            'detrend_spectrum_hist_image': self.pathes['detrender'][1],
            'baseline_spectrum_hist_image': self.pathes['baseliner'][1],
        }
        raw2epoch_report, _  = self.precompute_epochs.raw2epoch.get_transform_report()
        detector_report, _ = self.precompute_epochs.detector.get_transform_report()
        detrender_report, _ = self.precompute_epochs.detrender.get_transform_report()
        baseliner_report, _ = self.precompute_epochs.baseliner.get_transform_report()
        merged_dict = {**raw2epoch_report, **detector_report, **detrender_report, **baseliner_report, **report_images} 
        return merged_dict


class AutoCleaner():
    def __init__(self, scenarious_name, reref, n_components, l_freq,  
                 h_freq, notch_freq, down_sfreq, excluded_channels, 
                 stimulus, output_path=None):
        
        self.scenarious_name = scenarious_name
        self.reref = reref
        self.n_components = n_components
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.down_sfreq = down_sfreq
        self.excluded_channels = excluded_channels
        self.pathes = {}
        self.output_path = output_path
        self.stimulus = stimulus

    def clean(self, file_path, elc_file, bad_channels, progress_bar=None):
        self.file_path = file_path
        raw = mne.io.read_raw(self.file_path, preload=True, verbose=False)
        raw.info['bads'] = bad_channels

        self.prep_pipe = Sequence(
                chselector = ChannelSelector(self.excluded_channels),
                setter_montage = SetMontage('waveguard64', elc_file, mode='Cz', 
                                            threshold=0.019, interpolate=False),
                ffilter = FilterBandpass(l_freq=self.l_freq, h_freq=self.h_freq, 
                                         notch_freq=self.notch_freq, report=True),
                cropper = Cropping(stimulus=self.stimulus, report=False),
                rerefer = Rereference(self.reref, exclude='bads', report=True),
                resampler = Resample(sfreq=self.down_sfreq),
                ica = AutoICA(n_components=self.n_components, method='picard', report=True),
                interpolater = Interpolate()
                )
        self.prep_pipe(raw, progress_bar)
        self.pathes['filter'] = self.prep_pipe.ffilter.save_report(self.output_path, pref='filter_')
        self.pathes['reref'] = self.prep_pipe.rerefer.save_report(self.output_path, pref='reref_')
        self.pathes['ica'] = self.prep_pipe.ica.save_report(self.output_path, pref='ica_')
        return self.prep_pipe.insts
    
    def get_report(self,):
        ica_info, _ = self.prep_pipe.ica.get_transform_report()
        report_images = {
            'filter_spectrum_image': self.pathes['filter'][0],
            'reref_spectrum_image': self.pathes['reref'][0],
            'ica_all_comp_image': self.pathes['ica'][-2],
            'ica_each_comp_images': self.pathes['ica'][0:-2],
            'ica_spectrum_image':self.pathes['ica'][-1]
        }
        merged_dict = {**{'Record':self.file_path}, **ica_info, **report_images}
        return merged_dict
    

class QualityChecker():
    def __init__(self, excluded_channels, l_freq, h_freq, notch_freq,
                 noise_detector, stimulus):
        self.excluded_channels = excluded_channels
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.noise_detector = noise_detector
        self.file_path = None
        self.elc_path = None
        self.scenarious_name = None
        self.qc_path = None
        self.stimulus = stimulus
        self.pathes = {}
        self.qc_pipe = None

    def check(self, file_path, elc_path, qc_path, scenarious_name, progress_bar=None):
        self.file_path = file_path
        self.elc_path = elc_path
        self.scenarious_name = scenarious_name
        self.qc_path = qc_path

        if progress_bar:
            progress_bar.set_postfix(status="Reading file...")

        raw = mne.io.read_raw(self.file_path, preload=True, verbose=False)

        self.qc_pipe = Sequence(
            chselector=ChannelSelector(self.excluded_channels),
            setter_montage=SetMontage("waveguard64", self.elc_path, mode="Cz", threshold=0.019, interpolate=False),
            autofilter=FilterBandpass(l_freq=self.l_freq, h_freq=self.h_freq, notch_freq=self.notch_freq, report=True),
            checker=CheckEvents(scenarious_name=self.scenarious_name, report=True),
            detector=BadChannelsDetector(method=self.noise_detector, report=True),
        )

        self.qc_pipe(raw, progress_bar)

        if progress_bar:
            progress_bar.set_postfix(status="Saving reports...")

        self.pathes["filter"] = self.qc_pipe.autofilter.save_report(self.qc_path, pref="filter_")
        self.pathes["events"] = self.qc_pipe.checker.save_report(self.qc_path, pref="events_")
        self.pathes["detector"] = self.qc_pipe.detector.save_report(self.qc_path, pref="detector_")

    def get_report(self):
        data = get_file_info(self.file_path, self.elc_path)

        events_info, _ = self.qc_pipe.checker.get_transform_report()       # <-- ВКЛЮЧИЛИ
        detector_info, _ = self.qc_pipe.detector.get_transform_report()

        QC_images = {
            'filter_image': self.pathes['filter'][0],
            'events_image': self.pathes['events'][0] if self.pathes.get('events') else None,
            'clusters_image': self.pathes['detector'][0],
            'hist_bridges_image': self.pathes['detector'][1],
            'Noised_channels_image': self.pathes['detector'][2],
        }

        events_info, _ = self.qc_pipe.checker.get_transform_report()
        detector_info, _ = self.qc_pipe.detector.get_transform_report()

        merged_dict = {
            **data,              # сначала мета/файловая инфа (там заглушки)
            **events_info,       # потом события (перезапишут N/A реальными числами)
            **detector_info,
            **QC_images,
            "Record": self.file_path,
    }


        bad = set(merged_dict.get('HighAmp', [])) | set(merged_dict.get('LowAmp', [])) | \
        set(merged_dict.get('Bridged', [])) | set(merged_dict.get('Noise_Rate', []))
        merged_dict['N_bad_channels'] = len(bad)

        subseq = merged_dict.get('SubSeq_Flag', 'N/A')
        if isinstance(subseq, bool):
            merged_dict['Result_of_quality_checking'] = 'OK' if (subseq and merged_dict['N_bad_channels'] == 0) else 'CHECK'
        else:
            merged_dict['Result_of_quality_checking'] = 'OK' if merged_dict['N_bad_channels'] == 0 else 'CHECK'

        return merged_dict




def compute_snr_from_channels(raw, signal_channel, noisy_channel, eps=1e-12):
    data = raw.get_data(picks=[signal_channel, noisy_channel])
    signal = data[0]
    noisy = data[1]
    noise = noisy - signal
    power_signal = np.mean(signal ** 2)
    power_noise = np.mean(noise ** 2) + eps
    snr_linear = power_signal / power_noise
    snr_db = 10 * np.log10(snr_linear)
    probability = 1 / (1 + snr_linear)
    return snr_db, probability


def create_virtual_channel(raw, channel_list, name='AVG'):
    data = raw.get_data(picks=channel_list)  
    avg_data = np.mean(data, axis=0, keepdims=True)
    sfreq = raw.info['sfreq']
    info_avg = mne.create_info([name], sfreq, ch_types=['eeg'])
    info_avg._unlocked = True
    info_avg['highpass'] = None
    info_avg['lowpass'] = None
    info_avg._unlocked = False
    raw_avg = mne.io.RawArray(avg_data, info_avg)
    raw_copy = raw.copy()
    raw_copy.info._unlocked = True
    raw_copy.info['highpass'] = None
    raw_copy.info['lowpass'] = None
    raw_copy.info._unlocked = False
    raw_copy.add_channels([raw_avg], force_update_info=True)
    return raw_copy


def local_noise_tester(raw, target_channel="P1", neigh=["CP1", "P3", "PO3"], plot=True):
    desired = [target_channel, "AVG"]
    virtual_raw = create_virtual_channel(raw.copy(), neigh, "AVG")
    all_channels = neigh + desired
    new_order = [ch for ch in desired if ch in all_channels] + [ch for ch in all_channels if ch not in desired]
    order_indices = np.array([raw.ch_names.index(ch) for ch in new_order])
    if plot:
        virtual_raw.plot(picks=all_channels, order=order_indices)
    snr_value, probs = compute_snr_from_channels(virtual_raw, signal_channel="AVG", noisy_channel=target_channel)
    print(f"SNR: {snr_value:.2f} dB, Proba: {probs:.2f}")
