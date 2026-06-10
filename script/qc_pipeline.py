# qc_pipeline.py
# -*- coding: utf-8 -*-
"""
Единый QC-конвейер EEG.

Файл объединяет функциональность бывших `architecture.py` и `improved_qc.py`:
- поиск BrainVision-записей и чтение конфигов;
- legacy QualityChecker / preprocessing / processing helpers;
- improved QC: metadata, montage fallback, continuous QC, FASTER, spectral QC, epoch QC, ICA + ICLabel, HTML/MNE/CSV reports.

Положите этот файл в ту же папку `script`, где раньше лежали `architecture.py` и `improved_qc.py`.
"""

# ============================================================
# БЛОК 1. Infrastructure / legacy architecture.py
# ============================================================

import os
import re
import glob
import shutil
import json
import sqlite3
import ast
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
from datetime import datetime
import traceback
import matplotlib.pyplot as plt
import mne
from mne.report import Report
#from mne.time_frequency import psd_welch

# Добавляем mne-faster (требуется установка: pip install git+https://github.com/wmvanvliet/mne-faster.git)
HAS_FASTER = False
find_bad_channels = None

try:
    from mne_faster import find_bad_channels
    HAS_FASTER = True
except Exception:
    print("Установите mne-faster: pip install git+https://github.com/wmvanvliet/mne-faster.git")

    #raise ImportError("Установите mne-faster: pip install git+https://github.com/wmvanvliet/mne-faster.git")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "library"))

import eeg_auto_tools  # type: ignore[reportMissingImports]
from eeg_auto_tools.developments import QualityChecker, AutoCleaner, EpochsAnalysier  # type: ignore[reportMissingImports]
from eeg_auto_tools.scenarious import canonical_scenario, extract_visit_num_from_visit_folder  # type: ignore[reportMissingImports]

SCENARIO_ALIASES = {
    # Attention Network Test
    "ant": "ANT", "ants": "ANT", "attentionnetworktest": "ANT",
    "attention_network_test": "ANT", "attention network test": "ANT",

    # Rise Time / Amplitude Rise Time
    "riti": "RiTi", "rit": "RiTi", "risetime": "RiTi", "rise_time": "RiTi",
    "rise time": "RiTi", "art": "RiTi", "frt": "RiTi",

    # Mismatch negativity
    "mmn": "MMNs", "mmns": "MMNs", "mismatchnegativity": "MMNs",
    "mismatch negativity": "MMNs",

    # Resting-state scenarios are collapsed to Rest for QC; scenario_raw keeps exact folder/file label.
    "rest": "Rest", "resting": "Rest", "restingstate": "Rest",
    "restingstateeeg": "Rest", "rs": "Rest", "rs11": "Rest", "rs12": "Rest", "rs13": "Rest",

    # Other project scenarios
    "assr": "ASSR", "auditorysteadystateresponse": "ASSR",
    "vft": "VFT", "vft6": "VFT6", "visualfrequencytagging": "VFT",
    "visual frequency tagging": "VFT",
    "n400": "N400",
    "picturematch": "PictureMatch", "picture-match": "PictureMatch",
    "picture_match": "PictureMatch", "picture match": "PictureMatch",
    "picturematchparadigm": "PictureMatch", "picture match paradigm": "PictureMatch",
}

def canonical_scenario(name: str) -> str:
    if not name:
        return name
    s = name.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if re.fullmatch(r"rs\d+", s):
        return "Rest"
    return SCENARIO_ALIASES.get(s, name)

def extract_visit_num_from_visit_folder(visit_folder_name: str):
    if not visit_folder_name:
        return None
    m = re.search(r"(?:^|\s)посещение\s*(\d+)", visit_folder_name, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

def extract_participant_from_folder(folder_name: str):
    m = re.match(r"(?P<prefix>[A-Za-z]+)(?P<id>\d+)", folder_name)
    if not m:
        return None, None
    return m.group("prefix"), m.group("id")

def get_local_veriable(path: str):
    path = os.path.normpath(path)
    FILE_PATH = path
    RAW_PATH = os.path.dirname(FILE_PATH)
    EXPERIMENT_PATH = os.path.dirname(RAW_PATH)
    VISIT_PATH = os.path.dirname(EXPERIMENT_PATH)
    PARTICIPANT_PATH = os.path.dirname(VISIT_PATH)

    visit_name = os.path.basename(VISIT_PATH)
    experiment_name = os.path.basename(EXPERIMENT_PATH)

    PREPROCESSED_PATH = os.path.join(EXPERIMENT_PATH, 'Preprocessed_2')
    PROCESSED_PATH = os.path.join(EXPERIMENT_PATH, 'Processed')

    return (
        PARTICIPANT_PATH,
        VISIT_PATH,
        EXPERIMENT_PATH,
        FILE_PATH,
        RAW_PATH,
        PREPROCESSED_PATH,
        PROCESSED_PATH,
        visit_name,
        experiment_name,
    )

def extract_file_info(file_name: str):
    file_pattern = re.compile(
        r'(?P<prefix>[A-Za-z]+)'
        r'(?P<id>\d{3,4})_'
        r'v(?P<ver>\d+)(?:\.(?P<visit_num>\d+))?_'
        r'(?P<experiment>[^_]+)_'
        r'(?P<operator_code>[^_]+)_'
        r'(?P<date>\d{2}\.\d{2}\.\d{2,4})$'
    )
    m = file_pattern.match(file_name)
    if not m:
        return None
    d = m.groupdict()
    return d

def extract_preprocessed_file_info(file_name: str):
    file_pattern = re.compile(
        r'(?P<prefix>[A-Za-z]+)'
        r'(?P<id>\d{3,4})_'
        r'v(?P<ver>\d+)(?:\.(?P<visit_num>\d+))?_'
        r'(?P<experiment>[^_]+)_'
        r'(?P<operator_code>[^_]+)_'
        r'(?P<date>\d{2}\.\d{2}\.\d{2,4})_f_r_i$'
    )
    m = file_pattern.match(file_name)
    return m.groupdict() if m else None

def get_params_config(config_dir: str):
    with open(config_dir, "r", encoding='utf-8') as file:
        params = json.load(file)
    return params

def extract_eeg_and_vmrk_filenames_from_vhdr(vhdr_file_path: str):
    eeg_file = None
    vmrk_file = None
    try:
        with open(vhdr_file_path, 'r', encoding='utf-8') as vhdr_file:
            for line in vhdr_file:
                if line.startswith("DataFile="):
                    eeg_file = line.split('=')[1].strip()
                elif line.startswith("MarkerFile="):
                    vmrk_file = line.split('=')[1].strip()
        return eeg_file, vmrk_file
    except Exception as e:
        print(f"Error reading {vhdr_file_path}: {str(e)}")
        return None, None

def extract_eeg_filenames_from_vmrk(vmrk_file_path: str):
    eeg_file = None
    try:
        with open(vmrk_file_path, 'r', encoding='utf-8') as vmrk_file:
            for line in vmrk_file:
                if line.startswith("DataFile="):
                    eeg_file = line.split('=')[1].strip()
        return eeg_file
    except Exception as e:
        print(f"Error reading {vmrk_file_path}: {str(e)}")
        return None

def get_records(DATA_PATH, analysis_visits, analysis_experiments, analysis_ids):
    """
    Ищет BrainVision-записи.

    Поддерживаются две структуры:
    1) <project>/DATA/INP0005/посещение 4/ANTs/Raw/*.vhdr
    2) <project>/INP0005/посещение 4/ANTs/Raw/*.vhdr

    Функция сохранена под тем же именем, но стала устойчивее к ситуации,
    когда в data_config.json указано "DATA", а фактической папки DATA нет.
    """
    def _resolve_data_root(data_path):
        candidates = []
        if data_path:
            candidates.append(Path(data_path))

        script_dir = Path(__file__).resolve().parent
        cwd = Path.cwd()

        if data_path and not Path(data_path).is_absolute():
            candidates.extend([
                cwd / data_path,
                script_dir / data_path,
                script_dir.parent / data_path,
            ])

        candidates.extend([cwd, script_dir, script_dir.parent])

        for c in candidates:
            try:
                c = c.resolve()
            except Exception:
                continue
            if not c.exists() or not c.is_dir():
                continue
            try:
                has_participant_dirs = any(
                    p.is_dir() and re.match(r"^[A-Za-z]+\d{3,4}$", p.name)
                    for p in c.iterdir()
                )
                if has_participant_dirs:
                    return str(c)
            except Exception:
                continue

        return data_path

    DATA_PATH = _resolve_data_root(DATA_PATH)
    records = []

    if not DATA_PATH or not os.path.isdir(DATA_PATH):
        print(f"Data path not found: {DATA_PATH}.")
        return records

    ignored_dirs = {
        "script", "scripts", "library", "templates", "__pycache__",
        ".git", ".venv", "venv", "env", "results", "reports",
    }

    if len(analysis_ids) == 1 and analysis_ids[0] == '*':
        participants = [
            p for p in os.listdir(DATA_PATH)
            if os.path.isdir(os.path.join(DATA_PATH, p))
            and p.lower() not in ignored_dirs
            and re.match(r"^[A-Za-z]+\d{3,4}$", p)
        ]
    else:
        participants = analysis_ids

    for participant in participants:
        PARTICIPANT_PATH = os.path.join(DATA_PATH, participant)
        if not os.path.isdir(PARTICIPANT_PATH):
            print(f"Participant path not found: {PARTICIPANT_PATH}. Skipping.")
            continue

        if len(analysis_visits) == 1 and analysis_visits[0] == '*':
            visit_list = [v for v in os.listdir(PARTICIPANT_PATH) if os.path.isdir(os.path.join(PARTICIPANT_PATH, v))]
        else:
            wanted = set(str(x) for x in analysis_visits)
            visit_list = []
            for v in os.listdir(PARTICIPANT_PATH):
                full = os.path.join(PARTICIPANT_PATH, v)
                if not os.path.isdir(full):
                    continue
                vnum = extract_visit_num_from_visit_folder(v)
                if (v in wanted) or (vnum is not None and str(vnum) in wanted):
                    visit_list.append(v)

        for visit in visit_list:
            VISIT_PATH = os.path.join(PARTICIPANT_PATH, visit)

            if len(analysis_experiments) == 1 and analysis_experiments[0] == '*':
                experiments = [e for e in os.listdir(VISIT_PATH) if os.path.isdir(os.path.join(VISIT_PATH, e))]
            else:
                requested = {canonical_scenario(x) for x in analysis_experiments}
                experiments = []
                visit_contents = os.listdir(VISIT_PATH)
                for folder in visit_contents:
                    folder_path = os.path.join(VISIT_PATH, folder)
                    if not os.path.isdir(folder_path):
                        continue
                    if canonical_scenario(folder) in requested:
                        experiments.append(folder)
                experiments = list(dict.fromkeys(experiments))

            for experiment in experiments:
                EXPERIMENT_PATH = os.path.join(VISIT_PATH, experiment)
                RAW_PATH = os.path.join(EXPERIMENT_PATH, 'Raw')
                if not os.path.exists(RAW_PATH):
                    continue

                vhdr_files = glob.glob(os.path.join(RAW_PATH, "*.vhdr"))
                eeg_files = glob.glob(os.path.join(RAW_PATH, "*.eeg"))
                vmrk_files = glob.glob(os.path.join(RAW_PATH, "*.vmrk"))

                if len(vhdr_files) != 1 or len(eeg_files) != 1 or len(vmrk_files) != 1:
                    print(f"Неправильное количество файлов в {RAW_PATH}. Пропуск.")
                    continue

                eeg_file0, vmrk_file = extract_eeg_and_vmrk_filenames_from_vhdr(vhdr_files[0])
                if vmrk_file != os.path.basename(vmrk_files[0]):
                    print(f"Несоответствие .vmrk в {RAW_PATH}. Пропуск.")
                    continue

                eeg_file1 = extract_eeg_filenames_from_vmrk(vmrk_files[0])
                if eeg_file0 != eeg_file1:
                    print(f"Несоответствие .eeg в {RAW_PATH}. Пропуск.")
                    continue

                if os.path.basename(eeg_files[0]) != eeg_file0:
                    print(f"Неправильное имя .eeg файла в {RAW_PATH}. Пропуск.")
                    continue

                records.append(vhdr_files[0])

    return records

# Новые функции для оценки качества (Mohamed 2017 + визуализация)
freq_bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 50)}

def _trapz(y, x):
    # В новых NumPy может не быть np.trapz, зато есть np.trapezoid
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def compute_band_power_scores(raw):
    # Новый API вместо psd_welch (MNE >= 1.3)
    spectrum = raw.compute_psd(
        method="welch",
        fmin=0.5,
        fmax=50.0,
        n_fft=2048,
        n_overlap=1024,
        verbose=False,
    )
    psds, freqs = spectrum.get_data(return_freqs=True)  # psds: (n_channels, n_freqs)
    psds_mean = psds.mean(axis=0)

    band_power = {}
    total_power = _trapz(psds_mean, freqs)
    for band, (fmin, fmax) in freq_bands.items():
        idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        band_power[band] = _trapz(psds_mean[idx], freqs[idx]) if len(idx) > 0 else 0

    data = raw.get_data()
    scores = {
        "mean_amplitude_uv": float(np.abs(data).mean() * 1e6),
        "max_amplitude_uv": float(np.ptp(data, axis=1).max() * 1e6),
        "dominant_frequency_hz": float(freqs[np.argmax(psds_mean)]),
        "total_power": float(total_power),
        "alpha_beta_ratio": band_power["Alpha"] / (band_power["Beta"] + 1e-12),
        "theta_alpha_ratio": band_power["Theta"] / (band_power["Alpha"] + 1e-12),
    }
    scores.update({f"power_{band.lower()}": float(p) for band, p in band_power.items()})
    return scores, band_power, (psds, freqs)


def plot_legacy_band_power(band_power, freqs, psds_mean, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.semilogy(freqs, psds_mean, color='black')

    colors = plt.cm.viridis(np.linspace(0, 1, len(band_power)))
    for i, band in enumerate(band_power.keys()):
        ax.axvspan(*freq_bands[band], color=colors[i], alpha=0.3, label=band)

    ax.set_xlabel('Частота (Гц)')
    ax.set_ylabel('Мощность (V²/Hz)')
    ax.set_title('Спектр мощности EEG с выделенными полосами')
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    # НЕ закрываем здесь
    return fig


def render_page_QC(data, output_path, fs_path='templates', template_path='QC_template.html'):
    # Добавляем новые изображения в шаблон
    for key in ['filter_image', 'clusters_image', 'hist_bridges_image', 'Noised_channels_image',
                'band_power_image', 'faster_topomap_image']:
        data[key] = os.path.basename(data.get(key, ''))
    env = Environment(loader=FileSystemLoader(fs_path))
    template = env.get_template(template_path)
    rendered_html = template.render(data)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(rendered_html)

def render_page_prep(data, output_path, template_path='Prep_template.html'):
    data['filter_spectrum_image'] = os.path.basename(data.get('filter_spectrum_image', ''))
    data['reref_spectrum_image'] = os.path.basename(data.get('reref_spectrum_image', ''))
    data['ica_spectrum_image'] = os.path.basename(data.get('ica_spectrum_image', ''))
    data['ica_all_comp_image'] = os.path.basename(data.get('ica_all_comp_image', ''))
    data['ica_each_comp_images'] = [os.path.basename(p) for p in data.get('ica_each_comp_images', [])]

    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template(template_path)
    rendered_html = template.render(data)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(rendered_html)

def get_bad_chs(df, record):
    record = df[df['Record'] == record]
    record.loc[:, 'HighAmp'] = record['HighAmp'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    record.loc[:, 'LowAmp'] = record['LowAmp'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    record.loc[:, 'Bridged'] = record['Bridged'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    record.loc[:, 'Noise_Rate'] = record['Noise_Rate'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    high_amp = record['HighAmp'].iloc[0]
    low_amp = record['LowAmp'].iloc[0]
    bridged = record['Bridged'].iloc[0]
    noise_rate = record['Noise_Rate'].iloc[0]

    bad_channels = list(set(high_amp + low_amp + bridged + noise_rate))
    return bad_channels

# Основная функция QC с новыми добавлениями
def compute_qc(records, qc_dataframe_file, config_dir, hot_qc=True, exist_ok=True):
    qc_params = get_params_config(config_dir)['Quality_Check']

    with tqdm(records, total=len(records)) as progress_bar:
        for idx, record in enumerate(progress_bar):
            try:
                PARTICIPANT_PATH, VISIT_PATH, EXPERIMENT_PATH, FILE_PATH, _, _, _, visit_name, experiment = get_local_veriable(record)

                elc_files = glob.glob(os.path.join(VISIT_PATH, "*.elc"))
                ELC_PATH = elc_files[0] if elc_files else None

                qc_path = os.path.join(EXPERIMENT_PATH, 'QC')
                if exist_ok and glob.glob(f'{qc_path}/**/dossier.html', recursive=True):
                    continue
                if hot_qc and os.path.exists(qc_path):
                    shutil.rmtree(qc_path)
                os.makedirs(qc_path, exist_ok=True)

                folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                folder_path = os.path.join(qc_path, folder_name)
                os.makedirs(folder_path, exist_ok=True)

                file_stem = os.path.splitext(os.path.basename(FILE_PATH))[0]
                info = extract_file_info(file_stem) or {}
                scenario_key = canonical_scenario(experiment)

                participant_folder = os.path.basename(os.path.dirname(VISIT_PATH))
                prefix_f, id_f = extract_participant_from_folder(participant_folder)
                visit_from_folder = extract_visit_num_from_visit_folder(visit_name)

                id_val = info.get("id") or id_f
                visit_val = info.get("visit_num") or visit_from_folder

                meta_dict = {
                    "prefix": info.get("prefix") or prefix_f,
                    "id": str(id_val) if id_val is not None else None,
                    "visit_num": visit_val,
                    "scenario": scenario_key,
                    "scenario_raw": experiment,
                    "operator_code": info.get("operator_code"),
                    "date": info.get("date"),
                    "Record": FILE_PATH,
                }

                # Загрузка данных
                progress_bar.set_description('Загрузка raw...')
                raw = mne.io.read_raw_brainvision(FILE_PATH, preload=True)

                if ELC_PATH:
                    montage = mne.channels.read_custom_montage(ELC_PATH)

                    # BIP/EOG/ECG часто отсутствуют в .elc → делаем их не-EEG
                    misc_chs = [ch for ch in raw.ch_names if ch.upper().startswith("BIP")]
                    misc_chs += [ch for ch in raw.ch_names if ch.upper().startswith("ECG")]
                    misc_chs += [ch for ch in raw.ch_names if ch.upper().startswith("EOG")]

                    if misc_chs:
                        raw.set_channel_types({ch: "misc" for ch in misc_chs})

                    # главное: не падать, если монтаж не содержит позиции для некоторых каналов
                    try:
                        raw.set_montage(montage, match_case=False, on_missing="ignore")
                    except TypeError:
                        # на случай очень старого MNE без match_case
                        raw.set_montage(montage, on_missing="ignore")


                # FASTER (Nolan 2010) — Step 1: bad channels (нужны Epochs)
                faster_bad_channels = []

                if HAS_FASTER and find_bad_channels is not None:
                    try:
                        progress_bar.set_description('FASTER: bad channels.')
                        raw_for_faster = raw.copy().filter(1, 40, verbose=False)

                        # делаем фиксированные эпохи (FASTER работает по эпохам)
                        epochs = mne.make_fixed_length_epochs(
                            raw_for_faster,
                            duration=2.0,
                            overlap=0.0,
                            preload=True,
                            reject_by_annotation=True,
                            verbose=False,
                        )

                        if len(epochs) > 0:
                            faster_bad_channels = find_bad_channels(epochs, thres=5) or []
                        else:
                            faster_bad_channels = []

                        # применяем к raw (чтобы дальше PSD/репорт учитывали bads)
                        raw.info["bads"] = sorted(set(raw.info.get("bads", [])) | set(faster_bad_channels))

                    except Exception as e:
                        print(f"[WARNING] FASTER пропущен для {FILE_PATH}: {e}")
                        faster_bad_channels = []
                else:
                    print("[WARNING] FASTER отключён — продолжаю без него.")



                # Band power scores (Mohamed 2017)
                progress_bar.set_description('Расчёт мощности полос...')
                band_scores, band_power, psd_data = compute_band_power_scores(raw)
                psds, freqs = psd_data
                psds_mean = psds.mean(axis=0)

                band_plot_path = os.path.join(folder_path, 'band_power.png')
                plot_legacy_band_power(band_power, freqs, psds_mean, band_plot_path)

                # MNE Report
                progress_bar.set_description('Создание MNE Report...')
                rep = Report(title=f'Отчёт качества EEG — {meta_dict.get("id") or "Unknown"}')
                rep.add_raw(raw, title='Raw данные (с помеченными bad каналами)', psd=True)
                if faster_bad_channels:
                    html = "<b>FASTER bad channels:</b> " + ", ".join(faster_bad_channels)
                else:
                    html = "<b>FASTER bad channels:</b> none"

                rep.add_html(html, title="FASTER", section="Качество сигнала")

                fig_band = plot_legacy_band_power(band_power, freqs, psds_mean, save_path=band_plot_path)

                rep.add_figure(
                    fig_band,
                    title="Спектр мощности с полосами",
                    caption="Band power (Welch PSD) + интегралы по диапазонам",
                    section="Качество сигнала",
                )
                import matplotlib.pyplot as plt
                plt.close(fig_band)
                rep.save(os.path.join(folder_path, 'mne_report.html'), overwrite=True)

                # Ваш оригинальный QualityChecker
                q_checker = QualityChecker(**qc_params)
                q_checker.check(FILE_PATH, ELC_PATH, folder_path, scenarious_name=scenario_key, progress_bar=progress_bar)
                qc_report = q_checker.get_report()

                # Объединяем всё
                data_dict = {
                    **{'Start_time': folder_name},
                    **meta_dict,
                    **qc_report,
                    **qc_params,
                    'faster_bad_channels': faster_bad_channels,
                    'faster_n_bad': len(faster_bad_channels),
                    'band_power_image': os.path.basename(band_plot_path),
                    'mne_report_path': 'mne_report.html',
                    **band_scores,
                }

                # Комбинированный подсчёт плохих каналов с источниками
                q_sources = {
                    "quality_high_amp": set(_as_bad_channel_list(qc_report.get("HighAmp", []))),
                    "quality_low_amp": set(_as_bad_channel_list(qc_report.get("LowAmp", []))),
                    "quality_bridged": set(_as_bad_channel_list(qc_report.get("Bridged", []))),
                    "quality_noise_rate": set(_as_bad_channel_list(qc_report.get("Noise_Rate", []))),
                    "faster": set(_as_bad_channel_list(faster_bad_channels)),
                }
                all_bads = sorted(set().union(*q_sources.values()))
                data_dict["N_bad_channels"] = len(all_bads)
                data_dict["all_bad_channels"] = all_bads
                data_dict["bad_channel_sources_json"] = {k: sorted(v) for k, v in q_sources.items()}
                data_dict["bad_channel_counts_by_source"] = {k: len(v) for k, v in q_sources.items()}
                data_dict["bad_channel_thresholds_json"] = {
                    "legacy_qualitychecker_config": qc_params,
                    "legacy_bad_channel_thresholds": qc_report.get("BadChannelThresholds", {}),
                    "legacy_bad_channel_detection_report": qc_report.get("BadChannelDetectionReport", {}),
                }

                # Сохранение HTML и CSV
                page_path = os.path.join(folder_path, 'dossier.html')
                render_page_QC(data_dict, page_path)

                first_columns = ['Start_time', 'id', 'visit_num', 'scenario', 'duration',
                                 'N_bad_channels', 'faster_n_bad', 'mean_amplitude_uv', 'alpha_beta_ratio']
                cols = first_columns + [c for c in data_dict if c not in first_columns]
                df = pd.DataFrame([data_dict])[cols]
                df.to_csv(qc_dataframe_file, mode='a', sep=';', index=False,
                          header=not os.path.isfile(qc_dataframe_file), encoding='utf-8-sig')

            except Exception:
                error_details = traceback.format_exc()
                print(f'Ошибка: {idx} {record}\n{error_details}')
                continue

def compute_preprocessing(quality_records, qc_dataframe_file, clean_dataframe_file, config_dir,
                          hot_clean=True, exist_ok=True):
    qc_df = pd.read_csv(qc_dataframe_file, index_col=False, sep=';', encoding='utf-8-sig')

    with tqdm(quality_records, total=len(quality_records)) as progress_bar:
        for quality_record in progress_bar:
            progress_bar.set_description('Инициализация...')
            _, VISIT_PATH, _, _, _, PREPROCESSED_PATH, _, _, _ = get_local_veriable(quality_record)
            noised_file_name = os.path.splitext(os.path.basename(quality_record))[0]

            folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            clean_path = os.path.join(PREPROCESSED_PATH, folder_name)
            cleaned_file_name = os.path.join(clean_path, noised_file_name)

            if exist_ok and glob.glob(f'{PREPROCESSED_PATH}/**/*_f_r_i.fif', recursive=True):
                continue
            if hot_clean and os.path.exists(PREPROCESSED_PATH):
                shutil.rmtree(PREPROCESSED_PATH)
            os.makedirs(PREPROCESSED_PATH, exist_ok=True)
            os.makedirs(clean_path, exist_ok=True)

            elc_files = glob.glob(os.path.join(VISIT_PATH, "*.elc"))
            elc_file = elc_files[0] if elc_files else None

            bad_channels = get_bad_chs(qc_df, quality_record)

            info = extract_file_info(noised_file_name)
            if info is None:
                raise ValueError(f"Имя файла не соответствует шаблону: {noised_file_name}")
            scenarious_name = info['experiment']

            clean_params_all = get_params_config(config_dir)['Preprocessing']
            if scenarious_name not in clean_params_all:
                raise KeyError(f"Нет параметров препроцессинга для {scenarious_name}")
            clean_params = clean_params_all[scenarious_name]

            autocleaner = AutoCleaner(**clean_params, output_path=clean_path, scenarious_name=scenarious_name)
            raws = autocleaner.clean(quality_record, elc_file, bad_channels=bad_channels, progress_bar=progress_bar)

            mne.set_log_level('ERROR')
            raws[2].save(fname=cleaned_file_name + '_f.fif', overwrite=True)
            raws[4].save(fname=cleaned_file_name + '_f_r.fif', overwrite=True)
            raws[6].save(fname=cleaned_file_name + '_f_r_i.fif', overwrite=True)
            mne.set_log_level('WARNING')

            clean_report = autocleaner.get_report()
            data_dict = {
                **{'Start_time': folder_name},
                **clean_report,
                **clean_params,
                **{'output_file': cleaned_file_name + '_f_r_i.fif'}
            }

            page_path = os.path.join(clean_path, 'prep_info.html')
            render_page_prep(data_dict, page_path)

            df = pd.DataFrame([data_dict])
            first_columns = ['Start_time', 'Record']
            cols = first_columns + [c for c in df.columns if c not in first_columns]
            df = df[cols]
            df.to_csv(clean_dataframe_file, mode='a', sep=';', index=False,
                      header=not os.path.isfile(clean_dataframe_file), encoding='utf-8-sig')
            gc.collect()

def compute_processing(fa_records, epoched_dataframe_file, config_dir, hot_proc=True, exist_ok=True):
    with tqdm(fa_records, total=len(fa_records)) as progress_bar:
        for fa_record in progress_bar:
            progress_bar.set_description('Инициализация...')
            _, _, _, _, _, PREPROCESSED_PATH, PROCESSED_PATH, _, _ = get_local_veriable(fa_record)
            file_name = os.path.splitext(os.path.basename(fa_record))[0]

            info = extract_preprocessed_file_info(file_name)
            if info is None:
                raise ValueError(f"Имя препроцессированного файла не соответствует шаблону: {file_name}")
            scenarious_name = info['experiment']

            analysis_params_all = get_params_config(config_dir)['Processing']
            if scenarious_name not in analysis_params_all:
                raise KeyError(f"Нет параметров обработки для {scenarious_name}")
            analysis_params = analysis_params_all[scenarious_name]

            folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            clean_path = os.path.join(PROCESSED_PATH, folder_name)
            cleaned_file_name = os.path.join(clean_path, file_name)

            if exist_ok and glob.glob(f'{PROCESSED_PATH}/**/*_f_r_i_e_b_d.fif', recursive=True):
                continue
            if hot_proc and os.path.exists(PROCESSED_PATH):
                shutil.rmtree(PROCESSED_PATH)
            os.makedirs(clean_path, exist_ok=True)

            epochs_analysier = EpochsAnalysier(scenarious_name, **analysis_params)
            cleaned_epochses = epochs_analysier.compute(fa_record, clean_path, progress_bar=progress_bar)
            ep_dict = epochs_analysier.get_report()
            ep_dict['Start_time'] = folder_name
            ep_dict['Record'] = fa_record

            mne.set_log_level('ERROR')
            cleaned_epochses[0].save(cleaned_file_name + '_e.fif')
            cleaned_epochses[1].save(cleaned_file_name + '_e_b.fif')
            cleaned_epochses[2].save(cleaned_file_name + '_e_b_d.fif')
            mne.set_log_level('WARNING')

            data_dict = {
                **{'Start_time': folder_name},
                **ep_dict,
                **analysis_params,
                **{'output_file': cleaned_file_name + '_p.fif'}
            }

            df = pd.DataFrame([data_dict])
            first_columns = ['Start_time', 'Record', 'output_file']
            cols = first_columns + [c for c in df.columns if c not in first_columns]
            df = df[cols]
            df.to_csv(epoched_dataframe_file, mode='a', sep=';', index=False,
                      header=not os.path.isfile(epoched_dataframe_file), encoding='utf-8-sig')


# ============================================================
# БЛОК 2. Improved QC
# ============================================================

# improved_qc.py
# -*- coding: utf-8 -*-

import os
import re
import gc
import glob
import json
import shutil
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import mne
from mne.report import Report
from mne.preprocessing import ICA

# Инфраструктурные функции находятся выше в этом же файле.

from eeg_auto_tools.developments import QualityChecker  # type: ignore[reportMissingImports]


# ============================================================
# Optional dependencies
# ============================================================

HAS_FASTER = False
find_bad_channels = None

try:
    from mne_faster import find_bad_channels
    HAS_FASTER = True
except Exception:
    HAS_FASTER = False
    find_bad_channels = None


HAS_ICLABEL = False
label_components = None

try:
    from mne_icalabel import label_components
    HAS_ICLABEL = True
except Exception:
    HAS_ICLABEL = False
    label_components = None


HAS_AUTOREJECT = False
AutoReject = None
get_rejection_threshold = None

try:
    from autoreject import AutoReject, get_rejection_threshold
    HAS_AUTOREJECT = True
except Exception:
    HAS_AUTOREJECT = False
    AutoReject = None
    get_rejection_threshold = None


# ============================================================
# Basic constants
# ============================================================

FREQ_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

ARTIFACT_BANDS = {
    "blink": (0.5, 2.0),
    "muscle": (20.0, 40.0),
    "line_noise": (48.0, 52.0),
}


DEFAULT_IMPROVED_QC_PARAMS = {
    "psd": {
        "fmin": 0.5,
        "fmax": 50.0,
        "n_fft": 2048,
        "n_overlap": 1024,
    },
    "epoch_qc": {
        "enabled": True,
        "tmin": -0.2,
        "tmax": 0.8,
        "baseline": None,
        "min_events": 1,
    },
    "spectral_variants": {
        "enabled": True,
        "run_continuous_raw_pre_ica": True,
        "run_epoch_raw_pre_ica": True,
        "run_continuous_ica_cleaned": True,
        "run_epoch_ica_cleaned": True,
        "bad_muscle_ratio": 0.35,
        "bad_line_noise_ratio": 0.15,
        "bad_blink_ratio": 0.35,
        "save_epoch_detail_csv": True,
    },
    "protocol_marker_qc": {
        "enabled": True,
        "count_tolerance_abs": 2,
        "count_tolerance_ratio": 0.10,
        "scenario_specs": {
            "ANT": {
                "required_codes": [140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155],
                "optional_codes": [1,2,3,4,5,9,10,200,201,202]
            },
            "ASSR": {
                "required_codes": [50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80],
                "optional_codes": [0,1,2,3,4,5,9,10]
            },
            "MMNs": {
                "required_codes": [161,162,163],
                "planned_counts": {"161": 600, "162": 75, "163": 75},
                "optional_codes": [1,2,3,4,5,9,10]
            },
            "RiTi": {
                "required_codes": [115,120,130,160,240],
                "planned_counts": {"115": 60, "120": 60, "130": 60, "160": 60, "240": 60},
                "optional_codes": [1,2,3,4,5,9,10]
            },
            "Rest": {
                "required_codes": [11,12],
                "min_counts": {"11": 3, "12": 3},
                "expected_counts": {"11": 5, "12": 5},
                "optional_codes": [1,2,3,4,5,9,10,500]
            },
            "VFT": {
                "required_codes": [116,117,118,119,121,122,123,124,125,126,127,128],
                "optional_codes": [1,2,3,4,5,9,10,113,115,131,164]
            },
            "VFT6": {
                "required_codes": [116,117,118,119,121,122,123,124,125,126,127,128],
                "optional_codes": [1,2,3,4,5,9,10,113,115,131,164]
            },
            "N400": {
                "required_codes": [203,204],
                "optional_codes": [1,2,3,4,5,9,10,134,135,136,137,138,139,167,168,169,170,171,172]
            },
            "PictureMatch": {
                "required_codes": [203,204],
                "optional_codes": [1,2,3,4,5,9,10,134,135,136,137,138,139,167,168,169,170,171,172]
            }
        }
    },
    "basereport": {
        "enabled": True,
        # BaseReport ищется так: VISIT_PATH / event id / <папка соответствующего сценария> / ** / BaseReport.*
        # Поддерживаются Excel и текстовые CSV/semicolon-separated файлы BaseReport.csv.
        "event_id_folder_names": ["event id", "event_id", "Event ID", "EventID", "events", "Events"],
        "file_patterns": [
            "BaseReport.xlsx",
            "BaseReport.xlsm",
            "BaseReport.xls",
            "BaseReport.csv",
            "BaseReport.txt",
            "*BaseReport*.xlsx",
            "*Base Report*.xlsx",
            "*base*report*.xlsx",
            "*BaseReport*.xlsm",
            "*Base Report*.xlsm",
            "*base*report*.xlsm",
            "*BaseReport*.xls",
            "*Base Report*.xls",
            "*base*report*.xls",
            "*BaseReport*.csv",
            "*Base Report*.csv",
            "*base*report*.csv",
            "*BaseReport*.txt",
            "*Base Report*.txt",
            "*base*report*.txt",
        ],
        "max_files_to_read": 3,
        # Колонки, из которых можно извлекать реальные коды событий BaseReport.
        # ВАЖНО: "trial"/"трайал" здесь намеренно НЕ используются: в MMN BaseReport
        # колонка "Trial Number" содержит номер пробы 1..N, а не LSL/.vmrk-маркер.
        "marker_column_keywords": [
            "lsl", "marker", "trigger", "event", "event_code", "code", "label",
            "stim", "stimulus", "response", "condition",
            "метка", "маркер", "триггер", "событие", "код", "стимул", "ответ", "условие",
        ],
        # Точные имена имеют приоритет над остальными marker-like колонками.
        # Для MMN BaseReport вида "Trial Number; audio; lsl" будет выбрана только колонка "lsl".
        "marker_column_exact_names": [
            "lsl", "lslcode", "lslmarker", "lslevent",
            "marker", "markercode", "trigger", "triggercode",
            "eventcode", "event_code", "code", "stimulus", "stimuluscode",
            "метка", "код", "маркер", "триггер"
        ],
        # Колонки с этими словами никогда не считаются источником маркеров,
        # даже если в названии есть "trial"/"number"/и т.п.
        "marker_column_exclude_keywords": [
            "trial", "trialnumber", "trial_number", "трайал", "проба",
            "row", "index", "idx", "number", "num", "номер",
            "time", "timestamp", "onset", "duration", "latency", "rt",
            "block", "audio", "sound", "correct", "accuracy"
        ],
        "warn_missing_code_ratio": 0.10,
        "fail_missing_code_ratio": 0.40,
        "count_tolerance_abs": 2,
        "count_tolerance_ratio": 0.10,
        "max_rows_in_html": 80,
    },
    "faster": {
        "enabled": True,
        "filter_l_freq": 1.0,
        "filter_h_freq": 40.0,
        "epoch_duration": 2.0,
        "threshold": 5,
        "run_bad_channels": True,
        "run_bad_epochs": True,
        "bad_epoch_z_threshold": 5.0,
        "apply_bad_channels_to_raw": True,
        "save_machine_outputs": True,
    },
    "autoreject": {
        "enabled": True,
        "run_local": True,
        "run_global": True,
        "save_machine_outputs": True,
        "tmin": -0.2,
        "tmax": 0.8,
        "baseline": None,
        "min_epochs": 20,
        "max_epochs": 300,
        "hard_event_limit": 2000,
        "epoch_subset_mode": "stratified",
        "decim": 2,
        "cv": 3,
        "n_interpolate": [1, 4],
        "consensus": [0.3, 0.5],
        "random_state": 97,
        "n_jobs": 1,
    },
    "quality_db": {
        "enabled": True,
        "sqlite_filename": "qc_quality_assessment.sqlite",
        "table_name": "qc_quality_assessments",
    },
    "icalabel": {
        "enabled": True,
        "l_freq": 1.0,
        "h_freq": 45.0,
        "notch_freq": 50.0,
        "resample_sfreq": 250.0,
        "reference": "average",
        "method": "infomax",
        "extended": True,
        "n_components": 0.99,
        "random_state": 97,
        "max_iter": 1000,
        "auto_exclude_threshold": 0.80,
        "manual_review_threshold": 0.50,
        "brain_keep_threshold": 0.70,
        "auto_exclude_classes": [
            "eye blink",
            "muscle artifact",
            "heart beat",
            "line noise",
            "channel noise",
        ],
        "never_auto_exclude_classes": [
            "brain",
            "other",
        ],
    },
}


# ============================================================
# Utility
# ============================================================

def deep_update(base: dict, update: dict) -> dict:
    """Рекурсивно обновляет словарь параметров."""
    result = dict(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def trapz(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def safe_float(x):
    try:
        if x is None or not np.isfinite(x):
            return None
        return float(x)
    except Exception:
        return None



def _norm_channel_name(ch):
    """Нормализует имя канала для сопоставления источников bad-channel QC."""
    if ch is None:
        return ""
    return str(ch).strip()


def _as_bad_channel_list(value):
    """Преобразует разные форматы отчётов QualityChecker/FASTER/MNE в список каналов."""
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
            return _as_bad_channel_list(parsed)
        except Exception:
            return [_norm_channel_name(x) for x in re.split(r"[,;\s]+", s) if _norm_channel_name(x)]
    if isinstance(value, (set, tuple, list, np.ndarray, pd.Series)):
        out = []
        for item in list(value):
            if isinstance(item, (list, tuple, set)):
                out.extend(_as_bad_channel_list(item))
            else:
                name = _norm_channel_name(item)
                if name:
                    out.append(name)
        return sorted(set(out))
    name = _norm_channel_name(value)
    return [name] if name else []


def channel_types_to_misc(raw):
    """Служебные каналы переводятся в misc, чтобы не ломать montage / ICA."""
    type_map = {}

    for ch in raw.ch_names:
        name = ch.upper()

        if name.startswith("BIP"):
            type_map[ch] = "misc"
        elif name.startswith("ECG"):
            type_map[ch] = "ecg"
        elif name.startswith("EOG"):
            type_map[ch] = "eog"
        elif name.startswith("RESP"):
            type_map[ch] = "misc"

    if type_map:
        raw.set_channel_types(type_map, verbose=False)

    return raw



def get_missing_eeg_positions(raw):
    """Возвращает EEG-каналы, у которых после montage нет координат."""
    missing = []
    try:
        eeg_picks = mne.pick_types(
            raw.info,
            eeg=True,
            eog=False,
            ecg=False,
            stim=False,
            misc=False,
            exclude=[],
        )
        for pick in eeg_picks:
            ch = raw.info["chs"][pick]
            loc = ch["loc"][:3]
            if not np.all(np.isfinite(loc)) or np.allclose(loc, 0):
                missing.append(ch["ch_name"])
    except Exception:
        return []
    return missing


def apply_standard_montage(raw, fallback_name="standard_1020"):
    """Применяет стандартный MNE montage, если индивидуальный .elc отсутствует."""
    info = {
        "montage_source": f"fallback_{fallback_name}",
        "montage_status": "OK",
        "montage_warning": None,
    }
    try:
        montage = mne.channels.make_standard_montage(fallback_name)
        try:
            raw.set_montage(montage, match_case=False, on_missing="ignore", verbose=False)
        except TypeError:
            raw.set_montage(montage, on_missing="ignore", verbose=False)
    except Exception as e:
        info["montage_status"] = "FAIL"
        info["montage_warning"] = f"fallback montage failed: {e}"
        print(f"[ERROR] Не удалось применить fallback montage {fallback_name}: {e}")
    return raw, info


def apply_montage_with_fallback(raw, elc_path=None, fallback_name="standard_1020"):
    """
    Применяет индивидуальный .elc montage, если он есть.
    Если .elc отсутствует или не читается — применяет standard_1020.
    """
    montage_info = {
        "elc_file_found": bool(elc_path and os.path.exists(elc_path)),
        "elc_path": elc_path if elc_path else None,
        "montage_source": None,
        "montage_status": "UNKNOWN",
        "montage_warning": None,
        "montage_missing_eeg_channels": [],
    }

    raw = channel_types_to_misc(raw)

    if elc_path is not None and os.path.exists(elc_path):
        try:
            montage = mne.channels.read_custom_montage(elc_path)
            try:
                raw.set_montage(montage, match_case=False, on_missing="ignore", verbose=False)
            except TypeError:
                raw.set_montage(montage, on_missing="ignore", verbose=False)
            montage_info["montage_source"] = "custom_elc"
            montage_info["montage_status"] = "OK"
        except Exception as e:
            print(f"[WARN] .elc файл найден, но не применился: {elc_path}")
            print(f"[WARN] Ошибка .elc: {e}")
            print(f"[WARN] Будет применён fallback montage: {fallback_name}")
            raw, fallback_info = apply_standard_montage(raw, fallback_name=fallback_name)
            montage_info.update(fallback_info)
            montage_info["elc_file_found"] = True
            montage_info["elc_path"] = elc_path
            base_warning = f"custom .elc failed: {e}"
            montage_info["montage_warning"] = (
                base_warning + " | " + montage_info["montage_warning"]
                if montage_info.get("montage_warning") else base_warning
            )
    else:
        print(f"[WARN] .elc файл не найден. Используется стандартный montage {fallback_name}.")
        raw, fallback_info = apply_standard_montage(raw, fallback_name=fallback_name)
        montage_info.update(fallback_info)
        montage_info["elc_file_found"] = False
        montage_info["elc_path"] = None
        base_warning = f"elc file was not found; {fallback_name} montage was used"
        montage_info["montage_warning"] = (
            base_warning + " | " + montage_info["montage_warning"]
            if montage_info.get("montage_warning") else base_warning
        )

    missing = get_missing_eeg_positions(raw)
    montage_info["montage_missing_eeg_channels"] = missing
    if missing:
        montage_info["montage_status"] = "WARN" if montage_info.get("montage_status") != "FAIL" else "FAIL"
        missing_warning = (
            "Missing EEG positions after montage: "
            + ", ".join(missing[:20])
            + (" ..." if len(missing) > 20 else "")
        )
        montage_info["montage_warning"] = (
            montage_info["montage_warning"] + " | " + missing_warning
            if montage_info.get("montage_warning") else missing_warning
        )
        print(f"[WARN] После применения montage нет координат для EEG-каналов: {missing}")

    if montage_info.get("montage_status") == "UNKNOWN":
        montage_info["montage_status"] = "OK"

    return raw, montage_info


def get_eeg_picks(raw):
    return mne.pick_types(
        raw.info,
        eeg=True,
        eog=False,
        ecg=False,
        stim=False,
        misc=False,
        exclude=[],
    )


# ============================================================
# Metadata
# ============================================================

def build_meta_dict(file_path, visit_name, experiment, visit_path):
    file_stem = os.path.splitext(os.path.basename(file_path))[0]
    info = extract_file_info(file_stem) or {}

    scenario_key = canonical_scenario(experiment)

    participant_folder = os.path.basename(os.path.dirname(visit_path))
    prefix_f, id_f = extract_participant_from_folder(participant_folder)
    visit_from_folder = extract_visit_num_from_visit_folder(visit_name)

    id_val = info.get("id") or id_f
    visit_val = info.get("visit_num") or visit_from_folder

    prefix_val = info.get("prefix") or prefix_f
    record_type = detect_record_type((prefix_val or "") + str(id_val or ""))

    return {
        "prefix": prefix_val,
        "record_type": record_type,
        "id": str(id_val) if id_val is not None else None,
        "visit_num": visit_val,
        "scenario": scenario_key,
        "scenario_raw": experiment,
        "operator_code": info.get("operator_code"),
        "date": info.get("date"),
        "Record": file_path,
    }


def compute_metadata_qc(raw, file_path, elc_path, meta_dict):
    duration = raw.times[-1] if len(raw.times) > 0 else 0.0
    eeg_picks = get_eeg_picks(raw)

    warnings = []
    errors = []

    if not os.path.exists(file_path):
        errors.append("raw_file_missing")

    if len(eeg_picks) == 0:
        errors.append("no_eeg_channels")

    if elc_path is None:
        warnings.append("elc_montage_missing")

    if meta_dict.get("id") is None:
        warnings.append("subject_id_not_parsed")

    if meta_dict.get("scenario") is None:
        warnings.append("scenario_not_parsed")

    if errors:
        status = "FAIL"
    elif warnings:
        status = "WARN"
    else:
        status = "PASS"

    return {
        "metadata_status": status,
        "metadata_warnings": warnings,
        "metadata_errors": errors,
        "n_channels_total": len(raw.ch_names),
        "n_channels_eeg": int(len(eeg_picks)),
        "sfreq": safe_float(raw.info["sfreq"]),
        "duration": safe_float(duration),
        "channel_names": raw.ch_names,
        "bads_initial": list(raw.info.get("bads", [])),
    }


# ============================================================
# Continuous signal QC
# ============================================================

def compute_amplitude_metrics(raw):
    eeg_picks = get_eeg_picks(raw)
    data = raw.get_data(picks=eeg_picks) * 1e6  # V -> µV

    if data.size == 0:
        return {
            "mean_amplitude_uv": None,
            "median_amplitude_uv": None,
            "max_ptp_amplitude_uv": None,
            "channel_ptp_uv": {},
        }

    ch_names = [raw.ch_names[p] for p in eeg_picks]
    channel_ptp = np.ptp(data, axis=1)

    return {
        "mean_amplitude_uv": safe_float(np.mean(np.abs(data))),
        "median_amplitude_uv": safe_float(np.median(np.abs(data))),
        "max_ptp_amplitude_uv": safe_float(np.max(channel_ptp)),
        "channel_ptp_uv": {
            ch: safe_float(v) for ch, v in zip(ch_names, channel_ptp)
        },
    }



def _zscore_array(values):
    arr = np.asarray(values, dtype=float)
    med = np.nanmedian(arr)
    mad = np.nanmedian(np.abs(arr - med))
    if not np.isfinite(mad) or mad == 0:
        std = np.nanstd(arr)
        denom = std if np.isfinite(std) and std > 0 else 1.0
        return (arr - np.nanmean(arr)) / denom
    return 0.6745 * (arr - med) / mad


def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _bad_channel_threshold_snapshot(original_qc_report=None, qc_params=None, faster_params=None, autoreject_params=None):
    """Collect thresholds/parameters that can change N_bad_channels."""
    return {
        "legacy_qualitychecker_config": qc_params or {},
        "legacy_bad_channel_thresholds": (original_qc_report or {}).get("BadChannelThresholds", {}),
        "legacy_bad_channel_detection_report": (original_qc_report or {}).get("BadChannelDetectionReport", {}),
        "faster_params": faster_params or {},
        "autoreject_params": autoreject_params or {},
    }

def collect_bad_channel_provenance(original_qc_report, faster_result, raw, metadata_qc,
                                   qc_params=None, faster_params=None, autoreject_params=None,
                                   output_dir=None):
    """
    Transparent source table for N_bad_channels.

    N_bad_channels is the unique union of:
    - initial MNE bads before algorithms,
    - legacy QualityChecker: HighAmp, LowAmp, Bridged, Noise_Rate,
    - FASTER bad channels.

    raw.info['bads'] is saved for audit but is not treated as a separate
    new source to avoid double-counting after FASTER/QualityChecker mark channels.
    """
    source_map = {
        "initial_mne_bads": set(_as_bad_channel_list((metadata_qc or {}).get("bads_initial", []))),
        "quality_high_amp": set(_as_bad_channel_list((original_qc_report or {}).get("HighAmp", []))),
        "quality_low_amp": set(_as_bad_channel_list((original_qc_report or {}).get("LowAmp", []))),
        "quality_bridged": set(_as_bad_channel_list((original_qc_report or {}).get("Bridged", []))),
        "quality_noise_rate": set(_as_bad_channel_list((original_qc_report or {}).get("Noise_Rate", []))),
        "faster": set(_as_bad_channel_list((faster_result or {}).get("faster_bad_channels", []))),
    }
    final_bads = sorted(set().union(*source_map.values())) if source_map else []
    n_eeg = len(get_eeg_picks(raw)) if raw is not None else 0

    rows = []
    for ch in sorted(set(final_bads)):
        sources = sorted([name for name, vals in source_map.items() if ch in vals])
        rows.append({
            "channel": ch,
            "is_final_bad": True,
            "sources": ",".join(sources),
            "n_sources": len(sources),
            "initial_mne_bads": ch in source_map["initial_mne_bads"],
            "quality_high_amp": ch in source_map["quality_high_amp"],
            "quality_low_amp": ch in source_map["quality_low_amp"],
            "quality_bridged": ch in source_map["quality_bridged"],
            "quality_noise_rate": ch in source_map["quality_noise_rate"],
            "faster": ch in source_map["faster"],
        })

    counts_by_source = {k: len(v) for k, v in source_map.items()}
    warnings = []
    if n_eeg and len(final_bads) / n_eeg >= 0.30:
        warnings.append("bad_channel_ratio_ge_30_percent_check_thresholds_and_recording_quality")
    if (faster_result or {}).get("faster_error"):
        warnings.append("faster_failed_or_unavailable_bad_channel_count_depends_more_on_legacy_qualitychecker")
    legacy_union = set().union(source_map["quality_high_amp"], source_map["quality_low_amp"], source_map["quality_bridged"], source_map["quality_noise_rate"])
    if legacy_union and not source_map["faster"]:
        warnings.append("bad_channels_only_from_legacy_qualitychecker_or_faster_empty")

    thresholds = _bad_channel_threshold_snapshot(
        original_qc_report=original_qc_report,
        qc_params=qc_params,
        faster_params=faster_params,
        autoreject_params=autoreject_params,
    )

    csv_name = None
    thresholds_file = None
    if output_dir:
        try:
            csv_path = os.path.join(output_dir, "bad_channel_provenance.csv")
            pd.DataFrame(rows).to_csv(csv_path, sep=";", index=False, encoding="utf-8-sig")
            csv_name = os.path.basename(csv_path)
            thresholds_path = os.path.join(output_dir, "bad_channel_thresholds.json")
            _save_json(thresholds_path, thresholds)
            thresholds_file = os.path.basename(thresholds_path)
        except Exception:
            csv_name = None
            thresholds_file = None

    return {
        "N_bad_channels": int(len(final_bads)),
        "all_bad_channels": final_bads,
        "bad_channel_sources_json": {k: sorted(v) for k, v in source_map.items()},
        "bad_channel_counts_by_source": counts_by_source,
        "bad_channel_thresholds_json": thresholds,
        "bad_channel_provenance_csv": csv_name,
        "bad_channel_thresholds_file": thresholds_file,
        "bad_channel_warning_flags": warnings,
        "bad_channel_bads_after_algorithms": _as_bad_channel_list(raw.info.get("bads", [])) if raw is not None else [],
    }


def compute_faster_bad_channels(raw, params, output_dir=None):
    """
    FASTER-блок, независимый от AutoReject.

    Сохраняет машинночитаемые результаты:
    - faster_channel_metrics.csv: поканальные признаки fixed-length epochs;
    - faster_fixed_epoch_metrics.csv: признаки fixed-length эпох;
    - faster_bad_epochs.csv: эпохи, помеченные как плохие по z-score признакам;
    - faster_summary.json: сводка по запуску.

    Важно: это не меняет BaseReport/MMN-маркеры. FASTER работает на fixed-length epochs.
    """
    result = {
        "faster_enabled": bool(params.get("enabled", True)),
        "faster_available": bool(HAS_FASTER and find_bad_channels is not None),
        "faster_bad_channels": [],
        "faster_n_bad": 0,
        "faster_n_fixed_epochs": 0,
        "faster_n_bad_epochs": 0,
        "faster_bad_epoch_ratio": None,
        "faster_epoch_metrics_csv": None,
        "faster_bad_epochs_csv": None,
        "faster_channel_metrics_csv": None,
        "faster_summary_json": None,
        "faster_error": None,
    }

    if not result["faster_enabled"]:
        result["faster_error"] = None
        return result

    if not result["faster_available"]:
        result["faster_error"] = "mne-faster is not installed"
        return result

    try:
        raw_for_faster = raw.copy().pick_types(
            eeg=True, eog=False, ecg=False, stim=False, misc=False, exclude=[]
        )
        raw_for_faster.load_data()
        raw_for_faster.filter(
            params.get("filter_l_freq", 1.0),
            params.get("filter_h_freq", 40.0),
            verbose=False,
        )

        epochs = mne.make_fixed_length_epochs(
            raw_for_faster,
            duration=params.get("epoch_duration", 2.0),
            overlap=0.0,
            preload=True,
            reject_by_annotation=True,
            verbose=False,
        )
        result["faster_n_fixed_epochs"] = int(len(epochs))

        if len(epochs) == 0:
            result["faster_error"] = "no fixed-length epochs for FASTER"
            return result

        bads = []
        if params.get("run_bad_channels", True):
            bads = find_bad_channels(epochs, thres=params.get("threshold", 5)) or []
        bads = sorted(set([str(x) for x in bads]))
        result["faster_bad_channels"] = bads
        result["faster_n_bad"] = int(len(bads))

        data = epochs.get_data() * 1e6  # epochs x channels x time, µV
        ch_names = epochs.ch_names

        # Fixed-epoch metrics: быстрый независимый epoch-QC, чтобы сравнивать с AutoReject.
        epoch_ptp = np.ptp(data, axis=2).max(axis=1)
        epoch_var = np.var(data, axis=(1, 2))
        epoch_mean_abs = np.mean(np.abs(data), axis=(1, 2))
        z_ptp = _zscore_array(epoch_ptp)
        z_var = _zscore_array(epoch_var)
        z_mean_abs = _zscore_array(epoch_mean_abs)
        z_thr = float(params.get("bad_epoch_z_threshold", params.get("threshold", 5)))
        bad_epoch_mask = (np.abs(z_ptp) >= z_thr) | (np.abs(z_var) >= z_thr) | (np.abs(z_mean_abs) >= z_thr)

        epoch_rows = []
        for i in range(len(epochs)):
            start_s = float(epochs.events[i, 0] / epochs.info["sfreq"])
            epoch_rows.append({
                "epoch_index": int(i),
                "start_sec": start_s,
                "ptp_uv": safe_float(epoch_ptp[i]),
                "variance_uv2": safe_float(epoch_var[i]),
                "mean_abs_uv": safe_float(epoch_mean_abs[i]),
                "z_ptp": safe_float(z_ptp[i]),
                "z_variance": safe_float(z_var[i]),
                "z_mean_abs": safe_float(z_mean_abs[i]),
                "faster_bad_epoch": bool(bad_epoch_mask[i]),
            })
        epoch_df = pd.DataFrame(epoch_rows)
        bad_epoch_df = epoch_df[epoch_df["faster_bad_epoch"]].copy()
        result["faster_n_bad_epochs"] = int(len(bad_epoch_df))
        result["faster_bad_epoch_ratio"] = safe_float(len(bad_epoch_df) / len(epoch_df)) if len(epoch_df) else None

        # Channel metrics: сводные признаки каналов, отдельные от списка find_bad_channels.
        ch_ptp_median = np.median(np.ptp(data, axis=2), axis=0)
        ch_var_median = np.median(np.var(data, axis=2), axis=0)
        ch_mean_abs = np.median(np.mean(np.abs(data), axis=2), axis=0)
        channel_df = pd.DataFrame({
            "channel": ch_names,
            "ptp_uv_median": [safe_float(x) for x in ch_ptp_median],
            "variance_uv2_median": [safe_float(x) for x in ch_var_median],
            "mean_abs_uv_median": [safe_float(x) for x in ch_mean_abs],
            "bad_by_faster": [ch in bads for ch in ch_names],
        })

        if output_dir and params.get("save_machine_outputs", True):
            epoch_csv = os.path.join(output_dir, "faster_fixed_epoch_metrics.csv")
            bad_epoch_csv = os.path.join(output_dir, "faster_bad_epochs.csv")
            channel_csv = os.path.join(output_dir, "faster_channel_metrics.csv")
            summary_json = os.path.join(output_dir, "faster_summary.json")
            epoch_df.to_csv(epoch_csv, sep=";", index=False, encoding="utf-8-sig")
            bad_epoch_df.to_csv(bad_epoch_csv, sep=";", index=False, encoding="utf-8-sig")
            channel_df.to_csv(channel_csv, sep=";", index=False, encoding="utf-8-sig")
            summary = {k: v for k, v in result.items() if not isinstance(v, pd.DataFrame)}
            summary.update({
                "faster_params": params,
                "note": "FASTER uses fixed-length epochs and does not depend on scenario marker semantics.",
            })
            _save_json(summary_json, summary)
            result["faster_epoch_metrics_csv"] = os.path.basename(epoch_csv)
            result["faster_bad_epochs_csv"] = os.path.basename(bad_epoch_csv)
            result["faster_channel_metrics_csv"] = os.path.basename(channel_csv)
            result["faster_summary_json"] = os.path.basename(summary_json)

        return result

    except Exception as e:
        result["faster_error"] = str(e)
        return result


# ============================================================
# AutoReject QC
# ============================================================

def _subset_events_for_autoreject(events, max_epochs, mode="stratified", random_state=97):
    """Ограничивает число событий для AutoReject без изменения маркерной логики конвейера."""
    if events is None or len(events) == 0:
        return events, False, "no events"
    n_events = int(len(events))
    max_epochs = int(max_epochs) if max_epochs else n_events
    if n_events <= max_epochs:
        return events, False, f"all events used: {n_events} <= max_epochs {max_epochs}"

    rng = np.random.default_rng(int(random_state))
    if mode == "stratified":
        selected = []
        codes = np.unique(events[:, 2])
        per_code = max(1, max_epochs // max(1, len(codes)))
        for code in codes:
            idx = np.where(events[:, 2] == code)[0]
            take = min(len(idx), per_code)
            selected.extend(rng.choice(idx, size=take, replace=False).tolist())
        if len(selected) < max_epochs:
            rest = np.setdiff1d(np.arange(n_events), np.asarray(selected, dtype=int), assume_unique=False)
            take = min(len(rest), max_epochs - len(selected))
            if take > 0:
                selected.extend(rng.choice(rest, size=take, replace=False).tolist())
        selected = sorted(selected[:max_epochs])
    else:
        selected = sorted(rng.choice(np.arange(n_events), size=max_epochs, replace=False).tolist())

    return events[selected], True, f"events subsetted for AutoReject: {n_events} -> {len(selected)} ({mode})"


def _make_epochs_for_autoreject(raw, params):
    """
    Создаёт Epochs для AutoReject из текущих raw annotations / .vmrk.

    Не выполняет сценарный remapping и не меняет MMN-маркеры. Единственное ограничение —
    защита от нерационально большого количества событий/эпох.
    """
    events, event_id, event_error = extract_events_from_raw(raw)
    if event_error is not None:
        return None, events, event_id, event_error, False, event_error

    hard_limit = int(params.get("hard_event_limit", 2000) or 0)
    if hard_limit and len(events) > hard_limit:
        return None, events, event_id, (
            f"AutoReject skipped: event count {len(events)} exceeds hard_event_limit {hard_limit}. "
            "This usually indicates an excessive or service-marker event stream; lower the event count or raise the limit explicitly."
        ), False, "hard limit exceeded"

    min_epochs = int(params.get("min_epochs", 20))
    if len(events) < min_epochs:
        return None, events, event_id, f"not enough events for AutoReject: {len(events)} < min_epochs {min_epochs}", False, "too few events"

    used_events, subset_used, subset_note = _subset_events_for_autoreject(
        events,
        max_epochs=params.get("max_epochs", 300),
        mode=params.get("epoch_subset_mode", "stratified"),
        random_state=params.get("random_state", 97),
    )

    picks = get_eeg_picks(raw)
    if len(picks) == 0:
        return None, used_events, event_id, "no EEG channels for AutoReject", subset_used, subset_note

    try:
        epochs = mne.Epochs(
            raw,
            events=used_events,
            event_id=event_id,
            tmin=params.get("tmin", -0.2),
            tmax=params.get("tmax", 0.8),
            baseline=params.get("baseline", None),
            picks=picks,
            preload=True,
            reject_by_annotation=True,
            event_repeated=params.get("event_repeated", "drop"),
            decim=int(params.get("decim", 1) or 1),
            verbose=False,
        )
        if len(epochs) < min_epochs:
            return None, used_events, event_id, f"not enough epochs after Epochs creation: {len(epochs)} < {min_epochs}", subset_used, subset_note
        return epochs, used_events, event_id, None, subset_used, subset_note
    except Exception as e:
        return None, used_events, event_id, str(e), subset_used, subset_note


def compute_autoreject_qc(raw, params, output_dir=None):
    """AutoReject-блок, независимый от FASTER и BaseReport marker comparison."""
    result = {
        "autoreject_enabled": bool(params.get("enabled", True)),
        "autoreject_available": bool(HAS_AUTOREJECT and AutoReject is not None),
        "autoreject_status": "SKIPPED",
        "autoreject_n_events_input": 0,
        "autoreject_n_events_used": 0,
        "autoreject_used_epoch_subset": False,
        "autoreject_subset_note": None,
        "autoreject_n_epochs_total": 0,
        "autoreject_local_n_epochs_rejected": 0,
        "autoreject_local_reject_ratio": None,
        "autoreject_local_n_interpolated_epochs": 0,
        "autoreject_local_repair_ratio": None,
        "autoreject_global_n_epochs_rejected": 0,
        "autoreject_global_reject_ratio": None,
        "autoreject_epoch_log_csv": None,
        "autoreject_channel_log_csv": None,
        "autoreject_condition_summary_csv": None,
        "autoreject_global_thresholds_json": None,
        "autoreject_summary_json": None,
        "autoreject_error": None,
    }
    if not result["autoreject_enabled"]:
        return result
    if not result["autoreject_available"]:
        result["autoreject_error"] = "autoreject is not installed"
        return result

    try:
        epochs, used_events, event_id, err, subset_used, subset_note = _make_epochs_for_autoreject(raw, params)
        all_events, _, _ = extract_events_from_raw(raw)
        result["autoreject_n_events_input"] = int(len(all_events))
        result["autoreject_n_events_used"] = int(len(used_events)) if used_events is not None else 0
        result["autoreject_used_epoch_subset"] = bool(subset_used)
        result["autoreject_subset_note"] = subset_note
        if err is not None or epochs is None:
            result["autoreject_status"] = "SKIPPED"
            result["autoreject_error"] = err
            if output_dir and params.get("save_machine_outputs", True):
                summary_json = os.path.join(output_dir, "autoreject_summary.json")
                _save_json(summary_json, {**result, "autoreject_params": params})
                result["autoreject_summary_json"] = os.path.basename(summary_json)
            return result

        result["autoreject_status"] = "computed"
        result["autoreject_n_epochs_total"] = int(len(epochs))
        inv_event_id = {int(v): k for k, v in event_id.items()}
        event_codes = [int(x) for x in epochs.events[:, 2]]
        event_names = [inv_event_id.get(c, str(c)) for c in event_codes]

        epoch_log = pd.DataFrame({
            "epoch_index": np.arange(len(epochs), dtype=int),
            "event_code": event_codes,
            "event_name": event_names,
            "local_rejected": False,
            "local_n_bad_channels": 0,
            "local_bad_channels": "",
            "global_rejected": False,
        })
        channel_rows = []
        global_thresholds = {}

        if params.get("run_local", True):
            ar = AutoReject(
                n_interpolate=params.get("n_interpolate", [1, 4]),
                consensus=params.get("consensus", [0.3, 0.5]),
                cv=int(params.get("cv", 3)),
                random_state=int(params.get("random_state", 97)),
                n_jobs=int(params.get("n_jobs", 1)),
                verbose=False,
            )
            _, reject_log = ar.fit_transform(epochs, return_log=True)
            bad_epochs = np.asarray(reject_log.bad_epochs, dtype=bool)
            labels = np.asarray(reject_log.labels)
            epoch_log["local_rejected"] = bad_epochs
            epoch_log["local_n_bad_channels"] = labels.astype(bool).sum(axis=1)
            bad_ch_strings = []
            for row in labels.astype(bool):
                bad_ch_strings.append(",".join([ch for ch, bad in zip(epochs.ch_names, row) if bad]))
            epoch_log["local_bad_channels"] = bad_ch_strings
            result["autoreject_local_n_epochs_rejected"] = int(bad_epochs.sum())
            result["autoreject_local_reject_ratio"] = safe_float(bad_epochs.mean())
            interp_epochs = (labels.astype(bool).sum(axis=1) > 0) & (~bad_epochs)
            result["autoreject_local_n_interpolated_epochs"] = int(interp_epochs.sum())
            result["autoreject_local_repair_ratio"] = safe_float(interp_epochs.mean())

            bad_counts = labels.astype(bool).sum(axis=0)
            for ch, n_bad in zip(epochs.ch_names, bad_counts):
                channel_rows.append({
                    "channel": ch,
                    "autoreject_local_n_bad_epochs": int(n_bad),
                    "autoreject_local_bad_epoch_ratio": safe_float(n_bad / len(epochs)),
                })

        if params.get("run_global", True) and get_rejection_threshold is not None:
            reject = get_rejection_threshold(
                epochs,
                cv=int(params.get("cv", 3)),
                random_state=int(params.get("random_state", 97)),
                decim=1,
                verbose=False,
            )
            global_thresholds = {k: safe_float(v) for k, v in reject.items()}
            # Не меняем исходный epochs: создаём копию и смотрим drop_log.
            epochs_global = epochs.copy().drop_bad(reject=reject, verbose=False)
            dropped = np.array([len(x) > 0 for x in epochs_global.drop_log], dtype=bool)
            # drop_log относится к исходному набору; длина должна совпасть с len(epochs)
            if len(dropped) == len(epoch_log):
                epoch_log["global_rejected"] = dropped
                result["autoreject_global_n_epochs_rejected"] = int(dropped.sum())
                result["autoreject_global_reject_ratio"] = safe_float(dropped.mean())

        condition_df = None
        if not epoch_log.empty:
            condition_df = epoch_log.groupby(["event_code", "event_name"], dropna=False).agg(
                n_epochs=("epoch_index", "count"),
                local_rejected=("local_rejected", "sum"),
                global_rejected=("global_rejected", "sum"),
                mean_local_bad_channels=("local_n_bad_channels", "mean"),
            ).reset_index()
            condition_df["local_reject_ratio"] = condition_df["local_rejected"] / condition_df["n_epochs"].replace(0, np.nan)
            condition_df["global_reject_ratio"] = condition_df["global_rejected"] / condition_df["n_epochs"].replace(0, np.nan)

        if output_dir and params.get("save_machine_outputs", True):
            epoch_csv = os.path.join(output_dir, "autoreject_epoch_log.csv")
            channel_csv = os.path.join(output_dir, "autoreject_channel_log.csv")
            condition_csv = os.path.join(output_dir, "autoreject_condition_summary.csv")
            thresholds_json = os.path.join(output_dir, "autoreject_global_thresholds.json")
            summary_json = os.path.join(output_dir, "autoreject_summary.json")
            epoch_log.to_csv(epoch_csv, sep=";", index=False, encoding="utf-8-sig")
            pd.DataFrame(channel_rows).to_csv(channel_csv, sep=";", index=False, encoding="utf-8-sig")
            if condition_df is not None:
                condition_df.to_csv(condition_csv, sep=";", index=False, encoding="utf-8-sig")
                result["autoreject_condition_summary_csv"] = os.path.basename(condition_csv)
            _save_json(thresholds_json, global_thresholds)
            result["autoreject_epoch_log_csv"] = os.path.basename(epoch_csv)
            result["autoreject_channel_log_csv"] = os.path.basename(channel_csv)
            result["autoreject_global_thresholds_json"] = os.path.basename(thresholds_json)
            _save_json(summary_json, {**result, "autoreject_params": params, "global_thresholds": global_thresholds})
            result["autoreject_summary_json"] = os.path.basename(summary_json)

        return result

    except Exception as e:
        result["autoreject_status"] = "ERROR"
        result["autoreject_error"] = str(e)
        if output_dir and params.get("save_machine_outputs", True):
            try:
                summary_json = os.path.join(output_dir, "autoreject_summary.json")
                _save_json(summary_json, {**result, "autoreject_params": params})
                result["autoreject_summary_json"] = os.path.basename(summary_json)
            except Exception:
                pass
        return result


def build_bad_channel_method_comparison(original_qc_report, faster_result, autoreject_result, raw, output_dir=None):
    """Сводит источники bad-channel решений в отдельный машинночитаемый CSV."""
    ch_names = [raw.ch_names[p] for p in get_eeg_picks(raw)]
    rows = []
    q_sources = {
        "HighAmp": set(original_qc_report.get("HighAmp", []) if isinstance(original_qc_report.get("HighAmp", []), list) else []),
        "LowAmp": set(original_qc_report.get("LowAmp", []) if isinstance(original_qc_report.get("LowAmp", []), list) else []),
        "Bridged": set(original_qc_report.get("Bridged", []) if isinstance(original_qc_report.get("Bridged", []), list) else []),
        "Noise_Rate": set(original_qc_report.get("Noise_Rate", []) if isinstance(original_qc_report.get("Noise_Rate", []), list) else []),
    }
    faster_bads = set(faster_result.get("faster_bad_channels", []) or [])
    ar_channel_ratios = {}
    ar_csv = autoreject_result.get("autoreject_channel_log_csv")
    if output_dir and ar_csv:
        try:
            ar_df = pd.read_csv(os.path.join(output_dir, ar_csv), sep=";", encoding="utf-8-sig")
            if "channel" in ar_df.columns and "autoreject_local_bad_epoch_ratio" in ar_df.columns:
                ar_channel_ratios = dict(zip(ar_df["channel"].astype(str), ar_df["autoreject_local_bad_epoch_ratio"]))
        except Exception:
            ar_channel_ratios = {}

    for ch in ch_names:
        row = {
            "channel": ch,
            "bad_by_quality_high_amp": ch in q_sources["HighAmp"],
            "bad_by_quality_low_amp": ch in q_sources["LowAmp"],
            "bad_by_quality_bridged": ch in q_sources["Bridged"],
            "bad_by_quality_noise_rate": ch in q_sources["Noise_Rate"],
            "bad_by_faster": ch in faster_bads,
            "autoreject_local_bad_epoch_ratio": safe_float(ar_channel_ratios.get(ch)) if ch in ar_channel_ratios else None,
        }
        votes = sum(bool(row[k]) for k in [
            "bad_by_quality_high_amp", "bad_by_quality_low_amp", "bad_by_quality_bridged", "bad_by_quality_noise_rate", "bad_by_faster"
        ])
        if row["autoreject_local_bad_epoch_ratio"] is not None and row["autoreject_local_bad_epoch_ratio"] >= 0.10:
            votes += 1
        row["bad_method_votes"] = int(votes)
        row["bad_final_candidate"] = bool(votes > 0)
        rows.append(row)
    df = pd.DataFrame(rows)
    out_name = None
    if output_dir:
        out = os.path.join(output_dir, "bad_channel_method_comparison.csv")
        df.to_csv(out, sep=";", index=False, encoding="utf-8-sig")
        out_name = os.path.basename(out)
    return df, out_name


# ============================================================
# Spectral QC
# ============================================================

def compute_psd(raw, params):
    eeg_raw = raw.copy().pick_types(
        eeg=True,
        eog=False,
        ecg=False,
        stim=False,
        misc=False,
        exclude=[],
    )

    sfreq = eeg_raw.info["sfreq"]
    fmax = min(params.get("fmax", 50.0), sfreq / 2.0 - 1.0)

    spectrum = eeg_raw.compute_psd(
        method="welch",
        fmin=params.get("fmin", 0.5),
        fmax=fmax,
        n_fft=params.get("n_fft", 2048),
        n_overlap=params.get("n_overlap", 1024),
        verbose=False,
    )

    psds, freqs = spectrum.get_data(return_freqs=True)
    return psds, freqs, eeg_raw.ch_names


def compute_band_power_and_artifacts(raw, params):
    psds, freqs, ch_names = compute_psd(raw, params)
    psds_mean = psds.mean(axis=0)

    total_power_mean = trapz(psds_mean, freqs)

    band_power = {}
    for band, (fmin, fmax) in FREQ_BANDS.items():
        idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        band_power[band] = safe_float(trapz(psds_mean[idx], freqs[idx])) if len(idx) else 0.0

    channel_total_power = np.array([
        trapz(psd, freqs) for psd in psds
    ]) + 1e-20

    artifact_ratio_by_channel = {}
    artifact_summary = {}

    for name, (fmin, fmax) in ARTIFACT_BANDS.items():
        idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]

        if len(idx) == 0:
            ratios = np.zeros(len(ch_names))
        else:
            band_vals = np.array([
                trapz(psd[idx], freqs[idx]) for psd in psds
            ])
            ratios = band_vals / channel_total_power

        artifact_ratio_by_channel[name] = {
            ch: safe_float(v) for ch, v in zip(ch_names, ratios)
        }

        artifact_summary[f"{name}_ratio_mean"] = safe_float(np.mean(ratios))
        artifact_summary[f"{name}_ratio_max"] = safe_float(np.max(ratios))
        artifact_summary[f"{name}_ratio_median"] = safe_float(np.median(ratios))

    scores = {
        "dominant_frequency_hz": safe_float(freqs[np.argmax(psds_mean)]),
        "total_power": safe_float(total_power_mean),
        "alpha_beta_ratio": safe_float(
            band_power["alpha"] / (band_power["beta"] + 1e-20)
        ),
        "theta_alpha_ratio": safe_float(
            band_power["theta"] / (band_power["alpha"] + 1e-20)
        ),
    }

    for band, value in band_power.items():
        scores[f"power_{band}"] = value

    scores.update(artifact_summary)

    return {
        "spectral_scores": scores,
        "band_power": band_power,
        "artifact_ratio_by_channel": artifact_ratio_by_channel,
        "psds": psds,
        "freqs": freqs,
        "psds_mean": psds_mean,
        "psd_ch_names": ch_names,
    }


def plot_band_power(freqs, psds_mean, band_power, save_path):
    """Welch PSD с явным цветным разделением delta/theta/alpha/beta/gamma."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.semilogy(freqs, psds_mean, color="black", linewidth=1.6, label="Welch PSD")

    band_colors = {
        "delta": "#9ecae1",
        "theta": "#a1d99b",
        "alpha": "#fdae6b",
        "beta": "#fdd0a2",
        "gamma": "#bcbddc",
    }

    y_values = np.asarray(psds_mean)
    finite_y = y_values[np.isfinite(y_values) & (y_values > 0)]
    y_top = float(np.nanmax(finite_y)) if finite_y.size else 1.0

    for band, (fmin, fmax) in FREQ_BANDS.items():
        color = band_colors.get(str(band).lower(), None)
        ax.axvspan(fmin, fmax, color=color, alpha=0.35, label=band)
        ax.text((fmin + fmax) / 2, y_top, band, ha="center", va="top", fontsize=9)

    ax.set_title("Welch PSD + EEG frequency bands")
    ax.set_xlabel("Frequency, Hz")
    ax.set_ylabel("Power, V²/Hz")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_artifact_ratios(artifact_ratio_by_channel, save_path):
    rows = []
    for artifact_name, values in artifact_ratio_by_channel.items():
        for ch, ratio in values.items():
            rows.append((artifact_name, ch, ratio))

    if not rows:
        return

    df = pd.DataFrame(rows, columns=["artifact", "channel", "ratio"])
    pivot = df.pivot(index="channel", columns="artifact", values="ratio").fillna(0)

    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.18)))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Spectral artifact ratios by channel")

    fig.colorbar(im, ax=ax, label="ratio")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



# ============================================================
# Spectral variants: continuous / epoch, before / after ICA
# ============================================================

def prefix_keys(data: dict, prefix: str) -> dict:
    """Добавляет префикс к ключам словаря для раздельного вывода вариантов анализа."""
    return {f"{prefix}_{key}": value for key, value in (data or {}).items()}


def compute_continuous_spectral_variant(raw, psd_params, output_dir, variant_key, variant_label, stage_label):
    """Отдельный continuous spectral QC-вариант."""
    try:
        spectral = compute_band_power_and_artifacts(raw, psd_params)
        psd_image = os.path.join(output_dir, f"{variant_key}_psd.png")
        plot_band_power(spectral["freqs"], spectral["psds_mean"], spectral["band_power"], psd_image)
        artifact_image = os.path.join(output_dir, f"{variant_key}_artifact_ratios.png")
        plot_artifact_ratios(spectral["artifact_ratio_by_channel"], artifact_image)

        summary = prefix_keys(spectral["spectral_scores"], variant_key)
        summary.update({
            f"{variant_key}_enabled": True,
            f"{variant_key}_status": "computed",
            f"{variant_key}_error": None,
            f"{variant_key}_psd_image": os.path.basename(psd_image),
            f"{variant_key}_artifact_ratios_image": os.path.basename(artifact_image),
        })
        variant_row = {
            "variant": variant_key,
            "label": variant_label,
            "stage": stage_label,
            "epoching": "no",
            "input": "continuous Raw" if "raw" in variant_key else "continuous ICA-cleaned copy",
            "status": "computed",
            "n_epochs": "",
            "bad_epoch_ratio": "",
            "psd_image": os.path.basename(psd_image),
            "artifact_image": os.path.basename(artifact_image),
            "csv": "",
            "notes": "Welch PSD по непрерывному сигналу",
        }
        return {"summary": summary, "variant_row": variant_row, "images": [os.path.basename(psd_image), os.path.basename(artifact_image)], "error": None}
    except Exception as e:
        summary = {f"{variant_key}_enabled": True, f"{variant_key}_status": "error", f"{variant_key}_error": str(e)}
        variant_row = {
            "variant": variant_key,
            "label": variant_label,
            "stage": stage_label,
            "epoching": "no",
            "input": "continuous",
            "status": "error",
            "n_epochs": "",
            "bad_epoch_ratio": "",
            "psd_image": "",
            "artifact_image": "",
            "csv": "",
            "notes": str(e),
        }
        return {"summary": summary, "variant_row": variant_row, "images": [], "error": str(e)}


def _make_epochs_for_spectral_qc(raw, epoch_params):
    """Создаёт epochs из raw annotations / .vmrk для epoch-level spectral QC."""
    events, event_id, event_error = extract_events_from_raw(raw)
    if event_error is not None:
        return None, events, event_id, event_error
    if len(events) < epoch_params.get("min_events", 1):
        return None, events, event_id, "not enough events for epoch-level spectral QC"

    picks = get_eeg_picks(raw)
    if len(picks) == 0:
        return None, events, event_id, "no EEG channels for epoch-level spectral QC"

    try:
        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=epoch_params.get("tmin", -0.2),
            tmax=epoch_params.get("tmax", 0.8),
            baseline=epoch_params.get("baseline", None),
            picks=picks,
            preload=True,
            reject_by_annotation=True,
            event_repeated=epoch_params.get("event_repeated", "drop"),
            verbose=False,
        )
        if len(epochs) == 0:
            return None, events, event_id, "epochs are empty after reject_by_annotation"
        return epochs, events, event_id, None
    except Exception as e:
        return None, events, event_id, str(e)


def _band_values_from_psd(psd_1d, freqs):
    values = {}
    for band, (fmin, fmax) in FREQ_BANDS.items():
        idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        values[band] = safe_float(trapz(psd_1d[idx], freqs[idx])) if len(idx) else 0.0
    return values


def plot_epoch_mean_psd(freqs, epoch_psd_mean, save_path, title):
    """Mean PSD across epochs + variability band."""
    fig, ax = plt.subplots(figsize=(12, 6))
    psd_db = 10.0 * np.log10(epoch_psd_mean + 1e-30)
    mean_db = np.mean(psd_db, axis=0)
    std_db = np.std(psd_db, axis=0)
    ax.plot(freqs, mean_db, label="mean PSD across epochs")
    ax.fill_between(freqs, mean_db - std_db, mean_db + std_db, alpha=0.2, label="±1 SD")
    for band, (fmin, fmax) in FREQ_BANDS.items():
        ax.axvspan(fmin, fmax, alpha=0.08)
        ax.text((fmin + fmax) / 2, np.nanmax(mean_db), band, ha="center", va="top", fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("Frequency, Hz")
    ax.set_ylabel("Power, dB")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bad_epoch_ratio_by_condition(condition_csv, save_path, title):
    try:
        df = pd.read_csv(condition_csv, sep=";", encoding="utf-8-sig")
        if df.empty or "bad_epoch_ratio" not in df.columns:
            return None
        df = df.sort_values("bad_epoch_ratio", ascending=False).head(30)
        labels = df["event_name"].astype(str).tolist() if "event_name" in df.columns else df["event_code"].astype(str).tolist()
        fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.35), 5))
        ax.bar(np.arange(len(df)), df["bad_epoch_ratio"].fillna(0).values)
        ax.set_xticks(np.arange(len(df)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_ylim(0, max(1.0, float(df["bad_epoch_ratio"].fillna(0).max()) * 1.1))
        ax.set_ylabel("Bad epoch ratio")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path
    except Exception:
        return None


def compute_epoch_spectral_variant(raw, epoch_params, psd_params, variant_params, output_dir, variant_key, variant_label, stage_label):
    """
    Отдельный epoch-level spectral QC-вариант.

    Считает PSD по каждой epoch, затем band power и artifact ratios.
    Сейчас используется базовое epoching по raw annotations / .vmrk.
    """
    try:
        epochs, events, event_id, err = _make_epochs_for_spectral_qc(raw, epoch_params)
        if err is not None or epochs is None:
            summary = {
                f"{variant_key}_enabled": True,
                f"{variant_key}_status": "error",
                f"{variant_key}_n_events": int(len(events)),
                f"{variant_key}_n_epochs": 0,
                f"{variant_key}_bad_epochs": 0,
                f"{variant_key}_bad_epoch_ratio": None,
                f"{variant_key}_error": err,
            }
            variant_row = {
                "variant": variant_key,
                "label": variant_label,
                "stage": stage_label,
                "epoching": "yes",
                "input": "epochs from raw annotations / .vmrk",
                "status": "error",
                "n_epochs": 0,
                "bad_epoch_ratio": "",
                "psd_image": "",
                "artifact_image": "",
                "csv": "",
                "notes": err,
            }
            return {"summary": summary, "variant_row": variant_row, "images": [], "error": err}

        data = epochs.get_data()  # epochs x channels x time, V
        sfreq = epochs.info["sfreq"]
        n_times = data.shape[-1]
        fmax = min(psd_params.get("fmax", 50.0), sfreq / 2.0 - 1.0)
        n_fft = int(min(psd_params.get("n_fft", 2048), n_times))
        n_fft = max(n_fft, 2)
        n_overlap = int(min(psd_params.get("n_overlap", n_fft // 2), max(0, n_fft - 1)))

        psds, freqs = mne.time_frequency.psd_array_welch(
            data,
            sfreq=sfreq,
            fmin=psd_params.get("fmin", 0.5),
            fmax=fmax,
            n_fft=n_fft,
            n_per_seg=n_fft,
            n_overlap=n_overlap,
            verbose=False,
        )
        # psds: epochs x channels x freqs
        epoch_psd_mean = psds.mean(axis=1)
        mean_psd = epoch_psd_mean.mean(axis=0)

        inverse_event_id = {code: name for name, code in event_id.items()}
        epoch_event_codes = epochs.events[:, 2]
        epoch_event_names = [inverse_event_id.get(int(code), str(int(code))) for code in epoch_event_codes]

        rows = []
        bad_flags = []
        for i, psd_1d in enumerate(epoch_psd_mean):
            total_power = trapz(psd_1d, freqs) + 1e-20
            band_values = _band_values_from_psd(psd_1d, freqs)
            artifact_values = {}
            for artifact_name, (fmin, fmax_art) in ARTIFACT_BANDS.items():
                idx = np.where((freqs >= fmin) & (freqs <= fmax_art))[0]
                artifact_values[f"{artifact_name}_ratio"] = safe_float(trapz(psd_1d[idx], freqs[idx]) / total_power) if len(idx) else 0.0

            is_bad = (
                (artifact_values.get("muscle_ratio", 0.0) or 0.0) >= variant_params.get("bad_muscle_ratio", 0.35)
                or (artifact_values.get("line_noise_ratio", 0.0) or 0.0) >= variant_params.get("bad_line_noise_ratio", 0.15)
                or (artifact_values.get("blink_ratio", 0.0) or 0.0) >= variant_params.get("bad_blink_ratio", 0.35)
            )
            bad_flags.append(bool(is_bad))
            row = {
                "epoch_index": i,
                "event_code": int(epoch_event_codes[i]),
                "event_name": epoch_event_names[i],
                "spectral_bad_epoch": bool(is_bad),
            }
            row.update({f"power_{k}": v for k, v in band_values.items()})
            row.update(artifact_values)
            rows.append(row)

        detail_df = pd.DataFrame(rows)
        detail_csv = os.path.join(output_dir, f"{variant_key}_epoch_spectral_metrics.csv")
        if variant_params.get("save_epoch_detail_csv", True):
            detail_df.to_csv(detail_csv, sep=";", index=False, encoding="utf-8-sig")
        else:
            detail_csv = None

        condition_csv = None
        condition_image = None
        try:
            condition_df = detail_df.groupby(["event_code", "event_name"], dropna=False).agg(
                n_epochs=("epoch_index", "count"),
                n_bad_epochs=("spectral_bad_epoch", "sum"),
                alpha_power_mean=("power_alpha", "mean"),
                beta_power_mean=("power_beta", "mean"),
                theta_power_mean=("power_theta", "mean"),
                muscle_ratio_mean=("muscle_ratio", "mean"),
                line_noise_ratio_mean=("line_noise_ratio", "mean"),
                blink_ratio_mean=("blink_ratio", "mean"),
            ).reset_index()
            condition_df["bad_epoch_ratio"] = condition_df["n_bad_epochs"] / condition_df["n_epochs"].replace(0, np.nan)
            condition_csv = os.path.join(output_dir, f"{variant_key}_condition_summary.csv")
            condition_df.to_csv(condition_csv, sep=";", index=False, encoding="utf-8-sig")
            condition_image = os.path.join(output_dir, f"{variant_key}_bad_epoch_ratio_by_condition.png")
            plot_bad_epoch_ratio_by_condition(condition_csv, condition_image, f"Bad epoch ratio by condition: {variant_label}")
        except Exception:
            condition_csv = None
            condition_image = None

        mean_psd_image = os.path.join(output_dir, f"{variant_key}_mean_psd_across_epochs.png")
        plot_epoch_mean_psd(freqs, epoch_psd_mean, mean_psd_image, f"Mean PSD across epochs: {variant_label}")

        # Backward-compatible simple PSD image name.
        psd_image = os.path.join(output_dir, f"{variant_key}_mean_psd.png")
        plot_band_power(freqs, mean_psd, _band_values_from_psd(mean_psd, freqs), psd_image)

        heatmap_image = os.path.join(output_dir, f"{variant_key}_epoch_frequency_heatmap.png")
        try:
            fig, ax = plt.subplots(figsize=(12, max(5, min(14, len(epoch_psd_mean) * 0.025))))
            im = ax.imshow(
                10.0 * np.log10(epoch_psd_mean + 1e-30),
                aspect="auto",
                origin="lower",
                extent=[float(freqs[0]), float(freqs[-1]), 0, len(epoch_psd_mean)],
            )
            ax.set_title(f"Epoch × Frequency PSD heatmap: {variant_label}")
            ax.set_xlabel("Frequency, Hz")
            ax.set_ylabel("Epoch index")
            fig.colorbar(im, ax=ax, label="Power, dB")
            fig.tight_layout()
            fig.savefig(heatmap_image, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            heatmap_image = None

        n_epochs = int(len(detail_df))
        n_bad = int(np.sum(bad_flags))
        bad_ratio = safe_float(n_bad / n_epochs) if n_epochs else None
        status = "PASS"
        if bad_ratio is not None and bad_ratio >= 0.30:
            status = "FAIL"
        elif bad_ratio is not None and bad_ratio >= 0.10:
            status = "WARN"

        summary = {
            f"{variant_key}_enabled": True,
            f"{variant_key}_status": status,
            f"{variant_key}_n_events": int(len(events)),
            f"{variant_key}_n_epochs": n_epochs,
            f"{variant_key}_bad_epochs": n_bad,
            f"{variant_key}_bad_epoch_ratio": bad_ratio,
            f"{variant_key}_mean_alpha_power": safe_float(detail_df["power_alpha"].mean()),
            f"{variant_key}_mean_beta_power": safe_float(detail_df["power_beta"].mean()),
            f"{variant_key}_mean_theta_power": safe_float(detail_df["power_theta"].mean()),
            f"{variant_key}_muscle_ratio_max": safe_float(detail_df["muscle_ratio"].max()),
            f"{variant_key}_line_noise_ratio_max": safe_float(detail_df["line_noise_ratio"].max()),
            f"{variant_key}_blink_ratio_max": safe_float(detail_df["blink_ratio"].max()),
            f"{variant_key}_detail_csv": os.path.basename(detail_csv) if detail_csv else None,
            f"{variant_key}_condition_csv": os.path.basename(condition_csv) if condition_csv else None,
            f"{variant_key}_psd_image": os.path.basename(psd_image),
            f"{variant_key}_mean_psd_across_epochs_image": os.path.basename(mean_psd_image),
            f"{variant_key}_heatmap_image": os.path.basename(heatmap_image) if heatmap_image else None,
            f"{variant_key}_bad_epoch_ratio_image": os.path.basename(condition_image) if condition_image else None,
            f"{variant_key}_error": None,
        }
        variant_row = {
            "variant": variant_key,
            "label": variant_label,
            "stage": stage_label,
            "epoching": "yes",
            "input": "event-locked epochs from raw annotations / .vmrk",
            "status": status,
            "n_epochs": n_epochs,
            "bad_epoch_ratio": bad_ratio,
            "psd_image": os.path.basename(mean_psd_image),
            "artifact_image": os.path.basename(heatmap_image) if heatmap_image else "",
            "csv": os.path.basename(detail_csv) if detail_csv else "",
            "notes": "PSD по каждой эпохе; пока без BaseReport/scenario-specific filtering",
        }
        images = [os.path.basename(mean_psd_image), os.path.basename(psd_image)]
        if heatmap_image:
            images.append(os.path.basename(heatmap_image))
        if condition_image:
            images.append(os.path.basename(condition_image))
        return {"summary": summary, "variant_row": variant_row, "images": images, "error": None}
    except Exception as e:
        summary = {f"{variant_key}_enabled": True, f"{variant_key}_status": "error", f"{variant_key}_n_epochs": 0, f"{variant_key}_bad_epoch_ratio": None, f"{variant_key}_error": str(e)}
        variant_row = {
            "variant": variant_key,
            "label": variant_label,
            "stage": stage_label,
            "epoching": "yes",
            "input": "epochs",
            "status": "error",
            "n_epochs": 0,
            "bad_epoch_ratio": "",
            "psd_image": "",
            "artifact_image": "",
            "csv": "",
            "notes": str(e),
        }
        return {"summary": summary, "variant_row": variant_row, "images": [], "error": str(e)}


def plot_epoch_before_after_comparison(result, output_dir):
    """Создаёт сводные картинки before/after ICA для epoch-level spectral QC, если файлы есть."""
    outputs = {}
    pre = result.get("epoch_spectral_raw_pre_ica_mean_psd_across_epochs_image")
    post = result.get("epoch_spectral_ica_cleaned_mean_psd_across_epochs_image")
    if pre and post:
        try:
            pre_img = plt.imread(os.path.join(output_dir, pre))
            post_img = plt.imread(os.path.join(output_dir, post))
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            axes[0].imshow(pre_img)
            axes[0].axis("off")
            axes[0].set_title("Before ICA: mean PSD across epochs")
            axes[1].imshow(post_img)
            axes[1].axis("off")
            axes[1].set_title("After ICA: mean PSD across epochs")
            fig.tight_layout()
            out = os.path.join(output_dir, "epoch_spectral_before_after_ica_mean_psd.png")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            outputs["epoch_spectral_before_after_ica_mean_psd_image"] = os.path.basename(out)
        except Exception:
            pass

    pre_bad = result.get("epoch_spectral_raw_pre_ica_bad_epoch_ratio_image")
    post_bad = result.get("epoch_spectral_ica_cleaned_bad_epoch_ratio_image")
    if pre_bad and post_bad:
        try:
            pre_img = plt.imread(os.path.join(output_dir, pre_bad))
            post_img = plt.imread(os.path.join(output_dir, post_bad))
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            axes[0].imshow(pre_img)
            axes[0].axis("off")
            axes[0].set_title("Before ICA: bad epoch ratio by condition")
            axes[1].imshow(post_img)
            axes[1].axis("off")
            axes[1].set_title("After ICA: bad epoch ratio by condition")
            fig.tight_layout()
            out = os.path.join(output_dir, "epoch_spectral_before_after_ica_bad_epoch_ratio.png")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            outputs["epoch_spectral_before_after_ica_bad_epoch_ratio_image"] = os.path.basename(out)
        except Exception:
            pass

    return outputs


# ============================================================
# Event extraction and basic epoch-level QC
# ============================================================

def extract_events_from_raw(raw):
    try:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        return events, event_id, None
    except Exception as e:
        return np.empty((0, 3), dtype=int), {}, str(e)


def compute_epoch_metrics(raw, params):
    if not params.get("enabled", True):
        return {
            "epoch_qc_enabled": False,
            "n_events": 0,
            "n_epochs": 0,
            "epoch_qc_error": None,
        }

    events, event_id, event_error = extract_events_from_raw(raw)

    if event_error is not None:
        return {
            "epoch_qc_enabled": True,
            "n_events": 0,
            "n_epochs": 0,
            "event_id": {},
            "epoch_qc_error": event_error,
        }

    if len(events) < params.get("min_events", 1):
        return {
            "epoch_qc_enabled": True,
            "n_events": int(len(events)),
            "n_epochs": 0,
            "event_id": event_id,
            "epoch_qc_error": "not enough events for epoching",
        }

    try:
        picks = get_eeg_picks(raw)

        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=params.get("tmin", -0.2),
            tmax=params.get("tmax", 0.8),
            baseline=params.get("baseline", None),
            picks=picks,
            preload=True,
            reject_by_annotation=True,
            event_repeated=params.get("event_repeated", "drop"),
            verbose=False,
        )

        data = epochs.get_data() * 1e6  # epochs x channels x time, µV

        if data.size == 0:
            return {
                "epoch_qc_enabled": True,
                "n_events": int(len(events)),
                "n_epochs": 0,
                "event_id": event_id,
                "epoch_qc_error": "epochs are empty after reject_by_annotation",
            }

        epoch_mean_abs = np.mean(np.abs(data), axis=(1, 2))
        epoch_max_abs = np.max(np.abs(data), axis=(1, 2))
        epoch_ptp = np.ptp(data, axis=2).max(axis=1)

        return {
            "epoch_qc_enabled": True,
            "n_events": int(len(events)),
            "n_epochs": int(len(epochs)),
            "event_id": event_id,
            "epoch_mean_abs_median_uv": safe_float(np.median(epoch_mean_abs)),
            "epoch_mean_abs_iqr_uv": safe_float(
                np.percentile(epoch_mean_abs, 75) - np.percentile(epoch_mean_abs, 25)
            ),
            "epoch_max_abs_median_uv": safe_float(np.median(epoch_max_abs)),
            "epoch_max_abs_iqr_uv": safe_float(
                np.percentile(epoch_max_abs, 75) - np.percentile(epoch_max_abs, 25)
            ),
            "epoch_ptp_median_uv": safe_float(np.median(epoch_ptp)),
            "epoch_ptp_iqr_uv": safe_float(
                np.percentile(epoch_ptp, 75) - np.percentile(epoch_ptp, 25)
            ),
            "epoch_qc_error": None,
        }

    except Exception as e:
        return {
            "epoch_qc_enabled": True,
            "n_events": int(len(events)),
            "n_epochs": 0,
            "event_id": event_id,
            "epoch_qc_error": str(e),
        }


# ============================================================
# ICA + ICLabel
# ============================================================

def prepare_raw_for_ica(raw, params):
    raw_ica = raw.copy().pick_types(
        eeg=True,
        eog=False,
        ecg=False,
        stim=False,
        misc=False,
        exclude=[],
    )

    raw_ica.load_data()

    sfreq = raw_ica.info["sfreq"]
    resample_sfreq = params.get("resample_sfreq", None)
    if resample_sfreq is not None:
        try:
            resample_sfreq = float(resample_sfreq)
            if np.isfinite(resample_sfreq) and resample_sfreq > 0 and sfreq > resample_sfreq:
                raw_ica.resample(resample_sfreq, npad="auto", verbose=False)
        except Exception:
            pass

    sfreq = raw_ica.info["sfreq"]
    l_freq = params.get("l_freq", 1.0)
    h_freq = min(params.get("h_freq", 45.0), sfreq / 2.0 - 1.0)

    notch_freq = params.get("notch_freq", None)
    if notch_freq is not None and notch_freq < sfreq / 2.0:
        raw_ica.notch_filter(freqs=[notch_freq], verbose=False)

    raw_ica.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

    if params.get("reference", "average") == "average":
        raw_ica.set_eeg_reference("average", projection=False, verbose=False)

    return raw_ica


def run_ica_iclabel(raw, params, output_dir):
    """
    ICA decomposition + ICLabel classification.

    Дополнительно формирует объяснение решения по каждой компоненте:
    - что удаляем;
    - почему удаляем;
    - какие компоненты оставить на ручную проверку.

    В служебном ключе `_raw_ica_cleaned` возвращается ICA-cleaned copy,
    которую compute_improved_qc использует только для диагностических spectral variants.
    В итоговый CSV/HTML этот Raw-объект не записывается.
    """
    if not params.get("enabled", True):
        return {
            "icalabel_enabled": False,
            "icalabel_available": HAS_ICLABEL,
            "ica_n_components": 0,
            "iclabel_n_exclude": 0,
            "iclabel_n_review": 0,
            "iclabel_suggested_exclude": [],
            "iclabel_review_components": [],
            "iclabel_keep_components": [],
            "iclabel_component_table": [],
            "iclabel_exclude_table": [],
            "iclabel_exclude_summary": "ICA/ICLabel disabled",
            "iclabel_component_csv": None,
            "iclabel_exclude_csv": None,
            "ica_components_image": None,
            "ica_error": None,
        }

    if not HAS_ICLABEL or label_components is None:
        return {
            "icalabel_enabled": True,
            "icalabel_available": False,
            "ica_n_components": 0,
            "iclabel_n_exclude": 0,
            "iclabel_n_review": 0,
            "iclabel_suggested_exclude": [],
            "iclabel_review_components": [],
            "iclabel_keep_components": [],
            "iclabel_component_table": [],
            "iclabel_exclude_table": [],
            "iclabel_exclude_summary": "ICLabel was not executed: mne-icalabel is not installed",
            "iclabel_component_csv": None,
            "iclabel_exclude_csv": None,
            "ica_components_image": None,
            "ica_error": "mne-icalabel is not installed",
        }

    try:
        eeg_picks = get_eeg_picks(raw)
        if len(eeg_picks) < 8:
            return {
                "icalabel_enabled": True,
                "icalabel_available": True,
                "ica_n_components": 0,
                "iclabel_n_exclude": 0,
                "iclabel_n_review": 0,
                "iclabel_suggested_exclude": [],
                "iclabel_review_components": [],
                "iclabel_keep_components": [],
                "iclabel_component_table": [],
                "iclabel_exclude_table": [],
                "iclabel_exclude_summary": "ICA/ICLabel skipped: too few EEG channels",
                "iclabel_component_csv": None,
                "iclabel_exclude_csv": None,
                "ica_components_image": None,
                "ica_error": "too few EEG channels for reliable ICA/ICLabel",
            }

        raw_ica = prepare_raw_for_ica(raw, params)
        method = params.get("method", "infomax")
        fit_params = {"extended": True} if method == "infomax" and params.get("extended", True) else None

        ica = ICA(
            n_components=params.get("n_components", 0.99),
            method=method,
            fit_params=fit_params,
            random_state=params.get("random_state", 97),
            max_iter=params.get("max_iter", 1000),
        )
        ica.fit(raw_ica, verbose=False)

        label_result = label_components(raw_ica, ica, method="iclabel")
        labels = label_result.get("labels", [])
        y_pred_proba = label_result.get("y_pred_proba", None)

        auto_exclude_threshold = params.get("auto_exclude_threshold", 0.80)
        manual_review_threshold = params.get("manual_review_threshold", 0.50)
        brain_keep_threshold = params.get("brain_keep_threshold", 0.70)
        auto_exclude_classes = set(params.get("auto_exclude_classes", []))
        never_auto_exclude_classes = set(params.get("never_auto_exclude_classes", []))

        component_rows = []
        suggested_exclude = []
        review_components = []
        keep_components = []

        for idx, label in enumerate(labels):
            prob = None
            if y_pred_proba is not None:
                try:
                    prob = float(np.max(y_pred_proba[idx]))
                except Exception:
                    prob = None

            if label in auto_exclude_classes and prob is not None and prob >= auto_exclude_threshold:
                decision = "exclude"
                suggested_exclude.append(idx)
                rule = "auto_exclude_artifact_class"
                reason = (
                    f"remove: ICLabel classified component as '{label}' "
                    f"with probability {prob:.3f} >= auto_exclude_threshold {auto_exclude_threshold:.2f}"
                )
            elif label == "brain" and prob is not None and prob >= brain_keep_threshold:
                decision = "keep"
                keep_components.append(idx)
                rule = "keep_confident_brain"
                reason = (
                    f"keep: ICLabel classified component as brain "
                    f"with probability {prob:.3f} >= brain_keep_threshold {brain_keep_threshold:.2f}"
                )
            elif label in never_auto_exclude_classes:
                decision = "review"
                review_components.append(idx)
                rule = "never_auto_exclude_class"
                reason = f"review: label '{label}' is configured as never_auto_exclude; requires visual/manual confirmation"
            elif prob is not None and prob >= manual_review_threshold:
                decision = "review"
                review_components.append(idx)
                rule = "confident_non_excluded_review"
                reason = (
                    f"review: ICLabel probability {prob:.3f} >= manual_review_threshold "
                    f"{manual_review_threshold:.2f}, but class is not auto-excluded"
                )
            else:
                decision = "review"
                review_components.append(idx)
                rule = "low_confidence_review"
                reason = "review: probability unavailable" if prob is None else (
                    f"review: low confidence probability {prob:.3f} < manual_review_threshold {manual_review_threshold:.2f}"
                )

            component_rows.append({
                "component": idx,
                "label": label,
                "probability": prob,
                "decision": decision,
                "decision_rule": rule,
                "reason": reason,
            })

        exclude_table = [row for row in component_rows if row.get("decision") == "exclude"]
        if exclude_table:
            exclude_summary = "; ".join([
                f"IC{row['component']}={row['label']} p={row['probability']:.3f}: {row['reason']}"
                for row in exclude_table if row.get("probability") is not None
            ])
        else:
            exclude_summary = "No ICA components met automatic exclusion criteria"

        component_csv = os.path.join(output_dir, "ica_iclabel_components.csv")
        pd.DataFrame(component_rows).to_csv(component_csv, index=False, sep=";", encoding="utf-8-sig")
        exclude_csv = os.path.join(output_dir, "ica_components_suggested_for_exclusion.csv")
        pd.DataFrame(exclude_table).to_csv(exclude_csv, index=False, sep=";", encoding="utf-8-sig")

        components_fig_path = os.path.join(output_dir, "ica_components.png")
        try:
            n_plot = min(len(labels), 20)
            fig = ica.plot_components(picks=list(range(n_plot)), show=False, title="ICA components")
            if isinstance(fig, list):
                fig = fig[0]
            fig.savefig(components_fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            components_fig_path = None

        raw_ica_cleaned = None
        try:
            raw_ica_cleaned = raw_ica.copy()
            ica.apply(raw_ica_cleaned, exclude=suggested_exclude, verbose=False)
        except Exception:
            raw_ica_cleaned = None

        return {
            "icalabel_enabled": True,
            "icalabel_available": True,
            "ica_method": method,
            "ica_extended": bool(params.get("extended", True)),
            "ica_n_components": int(len(labels)),
            "iclabel_n_exclude": int(len(suggested_exclude)),
            "iclabel_n_review": int(len(set(review_components))),
            "iclabel_suggested_exclude": suggested_exclude,
            "iclabel_review_components": sorted(set(review_components)),
            "iclabel_keep_components": keep_components,
            "iclabel_component_table": component_rows,
            "iclabel_exclude_table": exclude_table,
            "iclabel_exclude_summary": exclude_summary,
            "iclabel_component_csv": os.path.basename(component_csv),
            "iclabel_exclude_csv": os.path.basename(exclude_csv),
            "ica_components_image": os.path.basename(components_fig_path) if components_fig_path else None,
            "ica_error": None,
            "_raw_ica_cleaned": raw_ica_cleaned,
        }
    except Exception as e:
        return {
            "icalabel_enabled": True,
            "icalabel_available": HAS_ICLABEL,
            "ica_n_components": 0,
            "iclabel_n_exclude": 0,
            "iclabel_n_review": 0,
            "iclabel_suggested_exclude": [],
            "iclabel_review_components": [],
            "iclabel_keep_components": [],
            "iclabel_component_table": [],
            "iclabel_exclude_table": [],
            "iclabel_exclude_summary": "ICA/ICLabel failed before exclusion decision",
            "iclabel_component_csv": None,
            "iclabel_exclude_csv": None,
            "ica_components_image": None,
            "ica_error": str(e),
        }




# ============================================================
# BaseReport / Event-ID validation
# ============================================================

def _norm_text(value):
    if value is None:
        return ""
    try:
        if isinstance(value, float) and np.isnan(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _norm_key(value):
    """Нормализация для сопоставления имён папок сценариев."""
    txt = _norm_text(value).lower()
    # оставляем буквы/цифры, чтобы 'event id' == 'event_id' == 'event-id'
    return re.sub(r"[^0-9a-zа-яё]+", "", txt)


def _scenario_terms(meta_dict):
    """Набор алиасов сценария для поиска папки внутри event id."""
    scenario = _norm_text(meta_dict.get("scenario"))
    raw = _norm_text(meta_dict.get("scenario_raw"))
    canonical = canonical_scenario(scenario or raw or "")
    key = _norm_key(canonical)

    terms = set()
    if scenario:
        terms.add(scenario)
    if raw:
        terms.add(raw)
    if canonical:
        terms.add(canonical)

    alias_map = {
        "ant": ["ANT", "ANTs", "ANTs_button", "attention network test"],
        "ants": ["ANT", "ANTs", "ANTs_button", "attention network test"],
        "riti": ["RiTi", "RiseTime", "Rise Time", "rise_time", "ART", "FRT"],
        "risetime": ["RiTi", "RiseTime", "Rise Time", "rise_time", "ART", "FRT"],
        "mmns": ["MMN", "MMNs", "mismatch negativity"],
        "mmn": ["MMN", "MMNs", "mismatch negativity"],
        "rest": ["Rest", "resting", "resting_state", "rs11", "rs12", "rs13", "rs14"],
        "assr": ["ASSR"],
        "vft": ["VFT", "Visual Frequency Tagging"],
        "picturematch": ["PictureMatch", "Picture-match", "picture_match", "N400", "N400_picture"],
        "n400": ["PictureMatch", "Picture-match", "picture_match", "N400"],
        "ssd": ["SSD", "SSD_final_LSI", "LSI"],
        "ssdfinallsi": ["SSD", "SSD_final_LSI", "LSI"],
        "speech": ["Speech"],
    }

    if key in alias_map:
        terms.update(alias_map[key])

    # Если scenario_raw содержит, например, ANTs или rs12, добавляем куски имени.
    for token in re.split(r"[_\-\s.]+", raw):
        if len(token) >= 2:
            terms.add(token)

    # Убираем пустые и нормализуем дальше при сравнении.
    return sorted({t for t in terms if _norm_text(t)})


def _folder_matches_current_scenario(folder_path, file_path, meta_dict):
    """
    True, если папка внутри event id относится к текущему сценарию.

    Пример ожидаемой структуры:
    DATA/INP0005/посещение 4/event id/INP0005_v1.4_ANTs_.../BaseReport.xlsx
    """
    folder_name = os.path.basename(os.path.normpath(folder_path))
    folder_norm = _norm_key(folder_name)
    record_stem = os.path.splitext(os.path.basename(file_path))[0]
    record_norm = _norm_key(record_stem)

    # 1) Полное или почти полное совпадение с именем записи.
    if record_norm and (record_norm in folder_norm or folder_norm in record_norm):
        return True

    # 2) Алиасы сценария.
    for term in _scenario_terms(meta_dict):
        term_norm = _norm_key(term)
        if term_norm and term_norm in folder_norm:
            return True

    return False


def _safe_int_marker(value):
    """Возвращает int-код, если значение похоже на marker/trigger/event code.

    Поддерживает реальные подписи из BrainVision/MNE:
    - Stimulus/s140
    - Stimulus/S 140
    - Response/R 201
    - Marker 140
    """
    if value is None:
        return None

    if isinstance(value, (int, np.integer)):
        code = int(value)
        return code if 0 <= code <= 9999 else None

    if isinstance(value, float):
        if np.isfinite(value) and float(value).is_integer():
            code = int(value)
            return code if 0 <= code <= 9999 else None
        return None

    text_value = _norm_text(value)
    if not text_value:
        return None

    if re.fullmatch(r"\d{1,4}", text_value):
        code = int(text_value)
        return code if 0 <= code <= 9999 else None

    m = re.search(
        r"(?i)(?:Stimulus|Response|Marker|Trigger|Code|Метка|Маркер|Триггер|Код)?\s*/?\s*[sSrR]?\s*[:=]?\s*(\d{1,4})(?:$|[^0-9])",
        text_value,
    )
    if m and re.search(r"(?i)(Stimulus|Response|Marker|Trigger|Code|Метка|Маркер|Триггер|Код|/s|/r|^s\s*\d|^r\s*\d)", text_value):
        code = int(m.group(1))
        return code if 0 <= code <= 9999 else None

    m = re.search(
        r"(?i)(?:^|[^A-Za-zА-Яа-я0-9])(?:S|R|Stimulus|Response|Marker|Trigger|Code|Метка|Маркер|Триггер|Код)\s*/?\s*[:=]?\s*(\d{1,4})(?:$|[^0-9])",
        text_value,
    )
    if m:
        code = int(m.group(1))
        return code if 0 <= code <= 9999 else None

    return None


def _extract_marker_codes_from_text(value):
    """Осторожно извлекает marker-коды только из marker-like текстов."""
    text_value = _norm_text(value)
    if not text_value:
        return []

    direct = _safe_int_marker(value)
    if direct is not None:
        return [direct]

    codes = []
    patterns = [
        r"(?i)(?:S|R|Stimulus|Response|Marker|Trigger|Code|Метка|Маркер|Триггер|Код)\s*/?\s*[:=]?\s*(\d{1,4})",
        r"(?i)(\d{1,4})\s*(?:marker|trigger|event|code|метка|маркер|триггер|код)",
    ]

    for pat in patterns:
        for m in re.finditer(pat, text_value):
            try:
                code = int(m.group(1))
                if 0 <= code <= 9999:
                    codes.append(code)
            except Exception:
                continue

    return sorted(set(codes))


def _column_norm_key(col_name):
    """Нормализованное имя колонки BaseReport: только буквы/цифры."""
    return _norm_key(col_name)


def _column_is_excluded_for_marker(col_name, params):
    """True для служебных числовых колонок, которые нельзя трактовать как marker code."""
    name = _norm_text(col_name).lower()
    key = _column_norm_key(col_name)
    if not name and not key:
        return True

    exclude_words = params.get("marker_column_exclude_keywords", [])
    for word in exclude_words:
        w_text = _norm_text(word).lower()
        w_key = _norm_key(word)
        if not w_text and not w_key:
            continue
        if (w_text and w_text in name) or (w_key and w_key in key):
            return True
    return False


def _column_is_exact_marker_name(col_name, params):
    """True, если имя колонки точно соответствует известному имени маркерной колонки."""
    if _column_is_excluded_for_marker(col_name, params):
        return False
    key = _column_norm_key(col_name)
    exact = {_norm_key(x) for x in params.get("marker_column_exact_names", [])}
    return bool(key and key in exact)


def _column_is_marker_like(col_name, params):
    name = _norm_text(col_name).lower()
    key = _column_norm_key(col_name)
    if not name and not key:
        return False
    if _column_is_excluded_for_marker(col_name, params):
        return False
    if _column_is_exact_marker_name(col_name, params):
        return True

    for kw in params.get("marker_column_keywords", []):
        kw_text = _norm_text(kw).lower()
        kw_key = _norm_key(kw)
        if not kw_text and not kw_key:
            continue
        # Для коротких ключей вроде "lsl" и "code" безопаснее сравнивать нормализованно.
        if kw_key and kw_key in key:
            return True
        if kw_text and kw_text in name:
            return True
    return False


def _select_basereport_marker_columns(df, params):
    """
    Выбирает колонки BaseReport, содержащие реальные event/trigger/LSL коды.

    Приоритет:
    1) точные имена вроде lsl, trigger, marker, event_code;
    2) прочие marker-like колонки;
    3) если таких колонок нет, возвращает [] — тогда код осторожно сканирует текстовые значения.

    Это исправляет MMN BaseReport формата:
        Trial Number; audio; lsl
    где Trial Number — номер пробы, audio — условие/звук, а lsl — реальный маркер.
    """
    if df is None or df.empty:
        return []

    exact_cols = [c for c in df.columns if _column_is_exact_marker_name(c, params)]
    if exact_cols:
        return exact_cols

    marker_cols = [c for c in df.columns if _column_is_marker_like(c, params)]
    return marker_cols


def _read_excel_sheets_safely(path):
    try:
        sheets = pd.read_excel(path, sheet_name=None)
        return sheets, None
    except Exception as e:
        return {}, str(e)

def _read_text_file_with_fallback(path):
    """Читает BaseReport.csv/txt с разными кодировками."""
    errors = []
    for enc in ["utf-8-sig", "utf-8", "cp1251", "latin1"]:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read(), enc, None
        except Exception as e:
            errors.append(f"{enc}: {e}")
    return "", None, " | ".join(errors)


def _detect_delimiter(lines):
    """Определяет разделитель для BaseReport.csv. В реальных файлах чаще всего ';'."""
    candidates = [";", "\t", ","]
    scores = {}
    for delim in candidates:
        counts = []
        for line in lines[:80]:
            if not line.strip() or set(line.strip()) <= {"*"}:
                continue
            counts.append(line.count(delim))
        scores[delim] = max(counts) if counts else 0
    return max(scores, key=scores.get) if max(scores.values()) > 0 else ";"


def _find_basereport_header_line(lines, delimiter, params):
    """
    Находит строку заголовка таблицы в BaseReport.csv.
    """
    for idx, line in enumerate(lines):
        raw = line.strip()
        if not raw or set(raw) <= {"*"}:
            continue
        cells = [c.strip() for c in raw.split(delimiter)]
        if len(cells) < 2:
            continue
        marker_like = [c for c in cells if _column_is_marker_like(c, params)]
        if marker_like:
            return idx

    for idx, line in enumerate(lines):
        raw = line.strip()
        if not raw or set(raw) <= {"*"}:
            continue
        if raw.count(delimiter) >= 3:
            return idx
    return 0


def _read_delimited_basereport_safely(path, params):
    """
    Читает BaseReport.csv / BaseReport.txt.
    """
    import io

    text, enc, err = _read_text_file_with_fallback(path)
    if err:
        return {}, err

    lines = text.splitlines()
    delimiter = _detect_delimiter(lines)
    header_idx = _find_basereport_header_line(lines, delimiter, params)
    table_text = "\n".join(lines[header_idx:])

    try:
        df = pd.read_csv(
            io.StringIO(table_text),
            sep=delimiter,
            engine="python",
            dtype=str,
            on_bad_lines="skip",
        )

        df = df.dropna(how="all")
        df = df.loc[:, [c for c in df.columns if _norm_text(c) and not str(c).lower().startswith("unnamed")]]
        df.columns = [_norm_text(c) for c in df.columns]
        df = df.dropna(how="all")

        sheet_name = f"csv:{os.path.basename(path)}"
        return {sheet_name: df}, None
    except Exception as e:
        return {}, f"failed to parse delimited BaseReport ({delimiter=}, {enc=}): {e}"


def _read_basereport_sheets_safely(path, params):
    """Читает BaseReport в Excel или CSV/semicolon-separated формате."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext in {".xlsx", ".xlsm", ".xls"}:
        return _read_excel_sheets_safely(path)
    if ext in {".csv", ".txt"}:
        return _read_delimited_basereport_safely(path, params)
    return _read_delimited_basereport_safely(path, params)



def _extract_basereport_markers_from_df(df, sheet_name, params):
    """
    Извлекает expected markers из листа BaseReport.

    Если в BaseReport есть колонки marker/trigger/event/code/метка/триггер,
    числа в этих колонках считаются ожидаемыми событиями. Если явных
    колонок нет, код ищет только текстовые формы S 11 / Marker 140 и т.п.,
    чтобы не спутать даты/время с маркерами.
    """
    rows = []
    if df is None or df.empty:
        return rows

    marker_cols = _select_basereport_marker_columns(df, params)
    scan_all = len(marker_cols) == 0
    cols_to_scan = list(df.columns) if scan_all else marker_cols

    for col in cols_to_scan:
        for row_idx, value in df[col].items():
            if pd.isna(value):
                continue

            text_value = _norm_text(value)
            if not text_value:
                continue

            if scan_all:
                codes = _extract_marker_codes_from_text(text_value)
            else:
                code = _safe_int_marker(value)
                codes = [code] if code is not None else _extract_marker_codes_from_text(text_value)

            for code in codes:
                if code is None:
                    continue
                rows.append({
                    "source_sheet": sheet_name,
                    "source_column": _norm_text(col),
                    "source_row": int(row_idx) + 2,
                    "marker_code": int(code),
                    "raw_value": text_value[:250],
                })

    if rows:
        rows = (
            pd.DataFrame(rows)
            .drop_duplicates(subset=["source_sheet", "source_column", "source_row", "marker_code", "raw_value"])
            .to_dict("records")
        )

    return rows


def _score_basereport_candidate(path, record_stem, meta_dict):
    """Оценка соответствия BaseReport текущей EEG-записи."""
    path_low = str(path).lower()
    stem_low = (record_stem or "").lower()
    score = 0

    if stem_low and stem_low in path_low:
        score += 20

    for tok in re.split(r"[_\-\s.]+", stem_low):
        tok = tok.strip().lower()
        if len(tok) >= 3 and tok in path_low:
            score += 1

    for key, weight in [("id", 5), ("scenario", 4), ("scenario_raw", 4), ("operator_code", 2)]:
        value = meta_dict.get(key)
        if value and str(value).lower() in path_low:
            score += weight

    for term in _scenario_terms(meta_dict):
        if _norm_key(term) and _norm_key(term) in _norm_key(path):
            score += 3

    return score


def find_event_id_roots(visit_path, params):
    """Находит папки event id внутри папки визита."""
    roots = []
    preferred_names = {_norm_key(x) for x in params.get("event_id_folder_names", [])}

    try:
        for name in os.listdir(visit_path):
            full = os.path.join(visit_path, name)
            if not os.path.isdir(full):
                continue
            key = _norm_key(name)
            if key in preferred_names or key in {"eventid", "events"}:
                roots.append(os.path.normpath(full))
    except Exception:
        pass

    return sorted(set(roots))


def find_scenario_event_folders(event_roots, file_path, meta_dict):
    """Ищет внутри event id только папки соответствующего текущей записи сценария."""
    scenario_dirs = []
    for root in event_roots:
        try:
            for name in os.listdir(root):
                full = os.path.join(root, name)
                if not os.path.isdir(full):
                    continue
                if _folder_matches_current_scenario(full, file_path, meta_dict):
                    scenario_dirs.append(os.path.normpath(full))
        except Exception:
            continue
    return sorted(set(scenario_dirs))


def find_basereport_files(visit_path, file_path, meta_dict, params):
    """
    Ищет BaseReport через папку сценария:
        visit_path / event id / <scenario folder> / ** / BaseReport.xlsx / BaseReport.csv
    """
    record_stem = os.path.splitext(os.path.basename(file_path))[0]
    event_roots = find_event_id_roots(visit_path, params)
    scenario_dirs = find_scenario_event_folders(event_roots, file_path, meta_dict)

    def _is_basereport_file(filename):
        base = os.path.basename(filename)
        low = base.lower()
        norm = _norm_key(base)
        if not low.endswith((".xlsx", ".xlsm", ".xls", ".csv", ".txt")):
            return False
        if low in {"basereport.xlsx", "basereport.xlsm", "basereport.xls", "basereport.csv", "basereport.txt"}:
            return True
        if "basereport" in norm:
            return True
        if "base" in low and "report" in low:
            return True
        return False

    def _walk_for_basereports(root_dir):
        found = []
        try:
            for cur_root, _, files in os.walk(root_dir):
                for fn in files:
                    full = os.path.join(cur_root, fn)
                    if _is_basereport_file(fn):
                        found.append(os.path.normpath(full))
        except Exception:
            pass
        return found

    def _sample_files(roots, only_data_files=False, limit=80):
        sample = []
        for root in roots:
            try:
                for cur_root, _, files in os.walk(root):
                    for fn in files:
                        full = os.path.normpath(os.path.join(cur_root, fn))
                        if only_data_files and not fn.lower().endswith((".xlsx", ".xlsm", ".xls", ".csv", ".txt")):
                            continue
                        sample.append(full)
                        if len(sample) >= limit:
                            return sample
            except Exception:
                continue
        return sample

    scan_files_sample = _sample_files(scenario_dirs, only_data_files=False, limit=80)
    scan_xls_files_sample = _sample_files(scenario_dirs, only_data_files=True, limit=80)

    candidates = []
    for scen_dir in scenario_dirs:
        for pattern in params.get("file_patterns", []):
            for fp in glob.glob(os.path.join(scen_dir, "**", pattern), recursive=True):
                if os.path.isfile(fp):
                    candidates.append(os.path.normpath(fp))
        candidates.extend(_walk_for_basereports(scen_dir))

    fallback_candidates = []
    if not candidates:
        for root in event_roots:
            fallback_candidates.extend(_walk_for_basereports(root))
        candidates.extend(fallback_candidates)

    candidates = sorted(set(candidates))

    scored = []
    for fp in candidates:
        scored.append({
            "path": fp,
            "score": _score_basereport_candidate(fp, record_stem, meta_dict),
            "inside_matching_scenario_folder": any(os.path.normpath(fp).startswith(os.path.normpath(d)) for d in scenario_dirs),
        })

    scored = sorted(
        scored,
        key=lambda x: (bool(x.get("inside_matching_scenario_folder")), x["score"]),
        reverse=True,
    )
    selected = [x["path"] for x in scored[:int(params.get("max_files_to_read", 3))]]

    return {
        "event_roots": event_roots,
        "scenario_dirs": scenario_dirs,
        "all_candidates": scored,
        "selected_files": selected,
        "scan_files_sample": scan_files_sample,
        "scan_xls_files_sample": scan_xls_files_sample,
        "fallback_candidates": sorted(set(fallback_candidates)),
    }

def _event_label_to_marker_code(label):
    """
    Нормализует подпись MNE/BrainVision события в реальный marker_code.

    Примеры:
        Stimulus/s141 -> 141
        Stimulus/S 141 -> 141
        Response/R 201 -> 201
        141 -> 141
    """
    label_text = _norm_text(label)
    if not label_text:
        return None

    code = _safe_int_marker(label_text)
    if code is not None:
        return code

    patterns = [
        r"(?i)(?:Stimulus|Response)\s*/\s*[sSrR]?\s*(\d{1,4})",
        r"(?i)\b[sSrR]\s*(\d{1,4})\b",
    ]
    for pat in patterns:
        m = re.search(pat, label_text)
        if m:
            try:
                code = int(m.group(1))
                return code if 0 <= code <= 9999 else None
            except Exception:
                pass
    return None

def _raw_event_counts(raw):
    events, event_id, event_error = extract_events_from_raw(raw)
    if event_error is not None:
        return {
            "events": events,
            "event_id": event_id,
            "event_error": event_error,
            "count_by_code": {},
            "label_by_code": {},
            "count_by_label": {},
            "count_by_mne_code": {},
            "mne_label_by_code": {},
        }

    mne_codes = [int(x) for x in events[:, 2]] if len(events) else []
    count_by_mne_code = pd.Series(mne_codes, dtype="int64").value_counts().sort_index().to_dict() if mne_codes else {}
    count_by_mne_code = {int(k): int(v) for k, v in count_by_mne_code.items()}

    mne_label_by_code = {}
    for label, code in (event_id or {}).items():
        try:
            mne_label_by_code[int(code)] = label
        except Exception:
            pass

    count_by_code = {}
    label_by_code = {}
    count_by_label = {}

    for mne_code, count in count_by_mne_code.items():
        label = mne_label_by_code.get(int(mne_code), str(mne_code))
        marker_code = _event_label_to_marker_code(label)

        if marker_code is None:
            marker_code = int(mne_code)

        count_by_code[int(marker_code)] = int(count_by_code.get(int(marker_code), 0) + int(count))
        if int(marker_code) not in label_by_code:
            label_by_code[int(marker_code)] = label
        count_by_label[label] = int(count)

    return {
        "events": events,
        "event_id": event_id,
        "event_error": None,
        "count_by_code": count_by_code,
        "label_by_code": label_by_code,
        "count_by_label": count_by_label,
        "count_by_mne_code": count_by_mne_code,
        "mne_label_by_code": mne_label_by_code,
    }



def detect_record_type(prefix_or_name):
    """Определяет тип записи INP/RNS/UNKNOWN по префиксу имени записи или участника."""
    txt = _norm_text(prefix_or_name).upper() if '_norm_text' in globals() else str(prefix_or_name or '').upper()
    m = re.search(r"\b(INP|RNS)\d{3,4}", txt)
    if m:
        return m.group(1)
    if txt.startswith("INP"):
        return "INP"
    if txt.startswith("RNS"):
        return "RNS"
    return "UNKNOWN"


def _as_int_dict(value):
    """Converts JSON-like dict keys to int where possible."""
    out = {}
    for k, v in (value or {}).items():
        try:
            out[int(k)] = int(v)
        except Exception:
            continue
    return out


def _scenario_spec_keys(scenario):
    """Possible keys for scenario-specific protocol specs."""
    s = canonical_scenario(scenario)
    keys = [s]
    if s == "MMN":
        keys.append("MMNs")
    if s == "MMNs":
        keys.append("MMN")
    if s == "PictureMatch":
        keys.append("N400")
    if s == "N400":
        keys.append("PictureMatch")
    if s == "VFT6":
        keys.append("VFT")
    if s == "VFT":
        keys.append("VFT6")
    # remove duplicates preserving order
    return list(dict.fromkeys(keys))


def _get_protocol_spec(scenario, params):
    specs = (params or {}).get("scenario_specs", {})
    for key in _scenario_spec_keys(scenario):
        if key in specs:
            return key, specs[key]
    return canonical_scenario(scenario), None


def compute_protocol_marker_qc(raw, meta_dict, output_dir, params):
    """
    Protocol Marker QC:
    сверяет реальные .vmrk / MNE annotations с ожидаемой картой маркеров сценария.
    Не зависит от наличия BaseReport и не меняет raw-аннотации.
    """
    result = {
        "protocol_marker_enabled": bool((params or {}).get("enabled", True)),
        "protocol_marker_status": "SKIPPED",
        "protocol_marker_spec_key": None,
        "protocol_marker_required_codes": [],
        "protocol_marker_optional_codes": [],
        "protocol_marker_observed_codes": [],
        "protocol_marker_missing_required_codes": [],
        "protocol_marker_count_mismatch_codes": [],
        "protocol_marker_expected_counts_json": {},
        "protocol_marker_observed_counts_json": {},
        "protocol_marker_comparison_csv": None,
        "protocol_marker_summary_json": None,
        "protocol_marker_notes": None,
        "protocol_marker_error": None,
    }
    if not result["protocol_marker_enabled"]:
        result["protocol_marker_notes"] = "Protocol Marker QC disabled"
        return result

    try:
        scenario = (meta_dict or {}).get("scenario")
        spec_key, spec = _get_protocol_spec(scenario, params or {})
        result["protocol_marker_spec_key"] = spec_key
        if not spec:
            result["protocol_marker_status"] = "SKIPPED"
            result["protocol_marker_notes"] = f"No protocol marker spec for scenario={scenario}"
            return result

        raw_counts_info = _raw_event_counts(raw)
        if raw_counts_info.get("event_error"):
            result["protocol_marker_status"] = "ERROR"
            result["protocol_marker_error"] = raw_counts_info.get("event_error")
            return result

        observed_counts = {int(k): int(v) for k, v in (raw_counts_info.get("count_by_code") or {}).items()}
        observed_codes = sorted(observed_counts.keys())

        required = [int(x) for x in spec.get("required_codes", [])]
        optional = [int(x) for x in spec.get("optional_codes", [])]
        planned_counts = _as_int_dict(spec.get("planned_counts", {}))
        expected_counts = _as_int_dict(spec.get("expected_counts", {}))
        min_counts = _as_int_dict(spec.get("min_counts", {}))

        # planned_counts has priority for strict comparison; expected_counts/min_counts are scenario guards.
        comparison_expected = {}
        comparison_expected.update(planned_counts)
        comparison_expected.update(expected_counts)
        comparison_expected.update(min_counts)

        missing_required = [code for code in required if observed_counts.get(code, 0) <= 0]
        mismatches = []
        rows = []
        tol_abs = int((params or {}).get("count_tolerance_abs", 2))
        tol_ratio = float((params or {}).get("count_tolerance_ratio", 0.10))

        check_codes = sorted(set(required) | set(comparison_expected.keys()))
        for code in check_codes:
            obs = int(observed_counts.get(code, 0))
            exp = comparison_expected.get(code)
            min_req = min_counts.get(code)
            status = "OK"
            note = ""
            if code in required and obs <= 0:
                status = "MISSING"
                note = "required code is absent"
            elif min_req is not None and obs < int(min_req):
                status = "TOO_FEW"
                note = f"observed {obs} < min_count {min_req}"
            elif exp is not None and code not in min_counts:
                diff = abs(obs - int(exp))
                allowed = max(tol_abs, int(round(abs(int(exp)) * tol_ratio)))
                if diff > allowed:
                    status = "COUNT_MISMATCH"
                    note = f"abs_diff={diff} > tolerance={allowed}"
            if status in ("TOO_FEW", "COUNT_MISMATCH"):
                mismatches.append(code)
            rows.append({
                "code": int(code),
                "required": code in required,
                "expected_or_min_count": exp,
                "observed_count": obs,
                "status": status,
                "note": note,
            })

        if missing_required:
            status = "FAIL"
            notes = f"Missing required protocol marker codes: {missing_required}"
        elif mismatches:
            status = "WARN"
            notes = f"Protocol marker count warnings: {mismatches}"
        else:
            status = "PASS"
            notes = "Required protocol marker codes are present"

        result.update({
            "protocol_marker_status": status,
            "protocol_marker_required_codes": required,
            "protocol_marker_optional_codes": optional,
            "protocol_marker_observed_codes": observed_codes,
            "protocol_marker_missing_required_codes": missing_required,
            "protocol_marker_count_mismatch_codes": mismatches,
            "protocol_marker_expected_counts_json": comparison_expected,
            "protocol_marker_observed_counts_json": observed_counts,
            "protocol_marker_notes": notes,
        })

        if output_dir:
            comp_csv = os.path.join(output_dir, "protocol_marker_comparison.csv")
            summary_json = os.path.join(output_dir, "protocol_marker_summary.json")
            pd.DataFrame(rows).to_csv(comp_csv, sep=";", index=False, encoding="utf-8-sig")
            _save_json(summary_json, {**result, "protocol_marker_params": params})
            result["protocol_marker_comparison_csv"] = os.path.basename(comp_csv)
            result["protocol_marker_summary_json"] = os.path.basename(summary_json)

        return result
    except Exception as e:
        result["protocol_marker_status"] = "ERROR"
        result["protocol_marker_error"] = str(e)
        return result


def make_protocol_marker_summary_html(result):
    rows = [
        {"parameter": "protocol_marker_enabled", "value": result.get("protocol_marker_enabled")},
        {"parameter": "protocol_marker_status", "value": result.get("protocol_marker_status")},
        {"parameter": "protocol_marker_spec_key", "value": result.get("protocol_marker_spec_key")},
        {"parameter": "required_codes", "value": result.get("protocol_marker_required_codes")},
        {"parameter": "observed_codes", "value": result.get("protocol_marker_observed_codes")},
        {"parameter": "missing_required_codes", "value": result.get("protocol_marker_missing_required_codes")},
        {"parameter": "count_mismatch_codes", "value": result.get("protocol_marker_count_mismatch_codes")},
        {"parameter": "comparison_csv", "value": result.get("protocol_marker_comparison_csv")},
        {"parameter": "summary_json", "value": result.get("protocol_marker_summary_json")},
        {"parameter": "notes", "value": result.get("protocol_marker_notes")},
        {"parameter": "error", "value": result.get("protocol_marker_error")},
    ]
    html = "<h2>Protocol Marker QC</h2>"
    html += make_html_table(rows, columns=["parameter", "value"], table_class="qc-table qc-kv")
    return html


def plot_basereport_event_comparison(comparison_df, save_path, max_markers=40):
    if comparison_df is None or comparison_df.empty:
        return None

    df = comparison_df.copy()
    df["max_count"] = df[["expected_count", "observed_count"]].max(axis=1)
    df = df[df["max_count"] > 0].sort_values(
        ["missing_flag", "count_mismatch_flag", "max_count"],
        ascending=[False, False, False],
    ).head(max_markers)

    if df.empty:
        return None

    labels = df["marker_code"].astype(str).tolist()
    x = np.arange(len(df))
    width = 0.42

    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.35), 6))
    ax.bar(x - width / 2, df["expected_count"].values, width, label="BaseReport expected")
    ax.bar(x + width / 2, df["observed_count"].values, width, label=".vmrk / annotations observed")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_xlabel("Marker code")
    ax.set_ylabel("Count")
    ax.set_title("BaseReport vs raw event stream: marker counts")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path



def qc_report_table_css():
    """CSS для устойчивой вёрстки wide-таблиц в HTML/MNE Report."""
    return """
    <style>
        .qc-table-wrap {
            width: 100%;
            overflow-x: auto;
            margin: 10px 0 24px 0;
            border: 1px solid #ddd;
            background: #fff;
        }
        table.qc-table {
            border-collapse: collapse;
            width: max-content;
            min-width: 100%;
            table-layout: auto;
            font-size: 12px;
        }
        table.qc-table th,
        table.qc-table td {
            border: 1px solid #ddd;
            padding: 6px 8px;
            vertical-align: top;
            text-align: left;
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: break-word;
            max-width: 360px;
        }
        table.qc-table th {
            background: #eef3fb;
            font-weight: 700;
            position: sticky;
            top: 0;
            z-index: 1;
        }
        table.qc-kv th {
            min-width: 230px;
            max-width: 300px;
        }
        table.qc-kv td {
            min-width: 260px;
        }
        .qc-note {
            background: #fff7d6;
            padding: 10px 12px;
            border-left: 5px solid #e1b12c;
            margin: 10px 0 18px 0;
        }
    </style>
    """


def make_html_table(items, columns=None, table_class="qc-table", empty_text="Нет данных."):
    """Безопасная HTML-таблица с прокруткой, чтобы header не съезжал относительно колонок."""
    if items is None:
        return f"<p>{html_escape(empty_text)}</p>"

    if isinstance(items, pd.DataFrame):
        rows = items.to_dict("records")
        if columns is None:
            columns = list(items.columns)
    elif isinstance(items, dict):
        rows = [{"parameter": k, "value": v} for k, v in items.items()]
        if columns is None:
            columns = ["parameter", "value"]
    else:
        rows = list(items) if isinstance(items, (list, tuple)) else []
        if columns is None:
            colset = []
            for r in rows:
                if isinstance(r, dict):
                    for k in r.keys():
                        if k not in colset:
                            colset.append(k)
            columns = colset

    if not rows:
        return f"<p>{html_escape(empty_text)}</p>"

    html = qc_report_table_css()
    html += f'<div class="qc-table-wrap"><table class="{html_escape(table_class)}">'
    html += "<thead><tr>" + "".join(f"<th>{html_escape(c)}</th>" for c in columns) + "</tr></thead><tbody>"
    for r in rows:
        html += "<tr>"
        for c in columns:
            value = r.get(c, "") if isinstance(r, dict) else ""
            html += f"<td>{html_escape(value)}</td>"
        html += "</tr>"
    html += "</tbody></table></div>"
    return html

def make_basereport_summary_html(result):
    status = result.get("basereport_status", "not_computed")

    kv = [
        {"parameter": "basereport_enabled", "value": result.get("basereport_enabled")},
        {"parameter": "basereport_found", "value": result.get("basereport_found")},
        {"parameter": "basereport_status", "value": status},
        {"parameter": "event_id_roots", "value": result.get("basereport_event_roots", [])},
        {"parameter": "scenario_event_folders", "value": result.get("basereport_scenario_dirs", [])},
        {"parameter": "selected BaseReport files", "value": result.get("basereport_files", [])},
        {"parameter": "expected_marker_count_total", "value": result.get("basereport_expected_marker_count_total")},
        {"parameter": "expected_unique_marker_codes", "value": result.get("basereport_expected_unique_marker_codes")},
        {"parameter": "observed_unique_event_codes", "value": result.get("basereport_observed_unique_event_codes")},
        {"parameter": "missing_code_ratio", "value": result.get("basereport_missing_code_ratio")},
        {"parameter": "n_missing_codes", "value": result.get("basereport_n_missing_codes")},
        {"parameter": "n_count_mismatch_codes", "value": result.get("basereport_n_count_mismatch_codes")},
        {"parameter": "notes", "value": result.get("basereport_notes")},
        {"parameter": "error", "value": result.get("basereport_error")},
    ]

    # Диагностика поиска: нужна, чтобы понять, почему BaseReport.xlsx не найден.
    diag = [
        {"parameter": "all_candidates", "value": result.get("basereport_all_candidates", [])},
        {"parameter": "xls_files_seen_sample", "value": result.get("basereport_scan_xls_files_sample", [])},
        {"parameter": "files_seen_in_scenario_sample", "value": result.get("basereport_scan_files_sample", [])},
    ]

    html = "<h2>BaseReport / Event-ID validation</h2>"
    html += make_html_table(kv, columns=["parameter", "value"], table_class="qc-table qc-kv")
    html += "<h3>BaseReport search diagnostics</h3>"
    html += make_html_table(diag, columns=["parameter", "value"], table_class="qc-table qc-kv")

    img = result.get("basereport_comparison_image")
    if img:
        html += f"""
        <h3>BaseReport vs .vmrk / annotations</h3>
        <img src="{html_escape(img)}" style="max-width:100%; border:1px solid #ddd;">
        """

    table_html = result.get("basereport_comparison_table_html")
    if table_html:
        html += "<h3>Marker comparison table</h3>" + table_html

    return html

def compute_basereport_qc(raw, visit_path, file_path, meta_dict, output_dir, params):
    """
    BaseReport QC:
    1) ищет папку visit/event id;
    2) внутри неё выбирает только папки соответствующего сценария текущей записи;
    3) в этих папках ищет BaseReport;
    4) извлекает expected marker/trigger/event коды;
    5) сравнивает их с raw events из .vmrk / MNE annotations.
    """
    if not params.get("enabled", True):
        return {
            "basereport_enabled": False,
            "basereport_found": False,
            "basereport_status": "SKIPPED",
            "basereport_error": None,
            "basereport_notes": "BaseReport validation is disabled in Improved_QC.basereport.enabled",
        }

    result = {
        "basereport_enabled": True,
        "basereport_found": False,
        "basereport_status": "WARN",
        "basereport_error": None,
        "basereport_notes": "",
        "basereport_event_roots": [],
        "basereport_scenario_dirs": [],
        "basereport_files": [],
        "basereport_sheets": [],
        "basereport_expected_marker_count_total": 0,
        "basereport_expected_unique_marker_codes": 0,
        "basereport_observed_unique_event_codes": 0,
        "basereport_missing_codes": [],
        "basereport_extra_observed_codes": [],
        "basereport_count_mismatch_codes": [],
        "basereport_n_missing_codes": 0,
        "basereport_n_extra_observed_codes": 0,
        "basereport_n_count_mismatch_codes": 0,
        "basereport_missing_code_ratio": None,
        "basereport_expected_markers_csv": None,
        "basereport_event_comparison_csv": None,
        "basereport_summary_json": None,
        "basereport_comparison_image": None,
        "basereport_comparison_table_html": None,
        "basereport_all_candidates": [],
        "basereport_scan_files_sample": [],
        "basereport_scan_xls_files_sample": [],
    }

    try:
        # События из .vmrk / annotations считаем сразу, даже если BaseReport не найден:
        # так в отчёте видно, что проблема именно в поиске BaseReport, а не в отсутствии событий.
        raw_counts_for_diag = _raw_event_counts(raw)
        if raw_counts_for_diag.get("event_error") is None:
            result["basereport_observed_unique_event_codes"] = int(len(raw_counts_for_diag.get("count_by_code", {})))
        else:
            result["basereport_error"] = raw_counts_for_diag.get("event_error")

        found = find_basereport_files(visit_path, file_path, meta_dict, params)
        result["basereport_event_roots"] = found.get("event_roots", [])
        result["basereport_scenario_dirs"] = found.get("scenario_dirs", [])
        result["basereport_all_candidates"] = found.get("all_candidates", [])
        result["basereport_scan_files_sample"] = found.get("scan_files_sample", [])
        result["basereport_scan_xls_files_sample"] = found.get("scan_xls_files_sample", [])

        if not result["basereport_event_roots"]:
            result["basereport_notes"] = "event id folder was not found inside visit folder"
            return result

        if not result["basereport_scenario_dirs"]:
            result["basereport_notes"] = (
                "scenario folder was not found inside event id folder; "
                f"searched scenario aliases: {_scenario_terms(meta_dict)}"
            )
            return result

        selected_files = found.get("selected_files", [])
        result["basereport_files"] = selected_files

        if not selected_files:
            result["basereport_notes"] = "BaseReport file was not found inside the matching scenario event-id folder"
            return result

        result["basereport_found"] = True

        marker_rows = []
        read_errors = []
        sheets_seen = []

        for br_path in selected_files:
            sheets, err = _read_basereport_sheets_safely(br_path, params)
            if err:
                read_errors.append(f"{br_path}: {err}")
                continue

            for sheet_name, df in sheets.items():
                sheets_seen.append(f"{os.path.basename(br_path)}::{sheet_name}")
                rows = _extract_basereport_markers_from_df(df, sheet_name, params)
                for r in rows:
                    r["basereport_file"] = br_path
                marker_rows.extend(rows)

        result["basereport_sheets"] = sheets_seen

        if read_errors and not marker_rows:
            result["basereport_error"] = " | ".join(read_errors)
            result["basereport_notes"] = "BaseReport files were found but could not be read"
            return result

        if not marker_rows:
            result["basereport_error"] = " | ".join(read_errors) if read_errors else None
            result["basereport_notes"] = (
                "BaseReport was found, but marker-like columns/codes were not extracted. "
                "Check marker column names or extend Improved_QC.basereport.marker_column_keywords."
            )
            return result

        markers_df = pd.DataFrame(marker_rows)
        markers_csv = os.path.join(output_dir, "basereport_expected_markers.csv")
        markers_df.to_csv(markers_csv, sep=";", index=False, encoding="utf-8-sig")
        result["basereport_expected_markers_csv"] = os.path.basename(markers_csv)

        expected_counts = markers_df["marker_code"].astype(int).value_counts().sort_index().to_dict()
        expected_counts = {int(k): int(v) for k, v in expected_counts.items()}

        raw_counts = raw_counts_for_diag if 'raw_counts_for_diag' in locals() else _raw_event_counts(raw)
        if raw_counts.get("event_error"):
            result["basereport_error"] = raw_counts.get("event_error")
            result["basereport_notes"] = "BaseReport was read, but raw events could not be extracted"
            return result

        observed_counts = raw_counts.get("count_by_code", {})
        label_by_code = raw_counts.get("label_by_code", {})

        expected_codes = set(expected_counts.keys())
        observed_codes = set(observed_counts.keys())

        missing_codes = sorted(expected_codes - observed_codes)
        extra_codes = sorted(observed_codes - expected_codes)

        comparison_rows = []
        mismatch_codes = []
        tol_abs = float(params.get("count_tolerance_abs", 2))
        tol_ratio = float(params.get("count_tolerance_ratio", 0.10))

        for code in sorted(expected_codes | observed_codes):
            exp = int(expected_counts.get(code, 0))
            obs = int(observed_counts.get(code, 0))
            diff = obs - exp
            missing_flag = bool(exp > 0 and obs == 0)
            extra_flag = bool(obs > 0 and exp == 0)

            if exp > 0:
                allowed = max(tol_abs, abs(exp) * tol_ratio)
                mismatch = bool(abs(diff) > allowed)
            else:
                mismatch = False

            if mismatch and not extra_flag and not missing_flag:
                mismatch_codes.append(int(code))

            comparison_rows.append({
                "marker_code": int(code),
                "raw_event_label": label_by_code.get(int(code), ""),
                "expected_count": exp,
                "observed_count": obs,
                "difference_observed_minus_expected": int(diff),
                "missing_flag": missing_flag,
                "extra_observed_flag": extra_flag,
                "count_mismatch_flag": mismatch,
            })

        comparison_df = pd.DataFrame(comparison_rows)
        comparison_csv = os.path.join(output_dir, "basereport_event_comparison.csv")
        comparison_df.to_csv(comparison_csv, sep=";", index=False, encoding="utf-8-sig")
        result["basereport_event_comparison_csv"] = os.path.basename(comparison_csv)

        comparison_image = os.path.join(output_dir, "basereport_event_comparison.png")
        if plot_basereport_event_comparison(comparison_df, comparison_image):
            result["basereport_comparison_image"] = os.path.basename(comparison_image)

        max_rows = int(params.get("max_rows_in_html", 80))
        table_df = comparison_df.copy()
        table_df["priority"] = (
            table_df["missing_flag"].astype(int) * 3
            + table_df["count_mismatch_flag"].astype(int) * 2
            + table_df["extra_observed_flag"].astype(int)
            + table_df[["expected_count", "observed_count"]].max(axis=1) / 100000.0
        )
        table_df = table_df.sort_values("priority", ascending=False).drop(columns=["priority"]).head(max_rows)
        result["basereport_comparison_table_html"] = make_html_table(table_df)

        expected_unique = len(expected_codes)
        observed_unique = len(observed_codes)
        missing_ratio = (len(missing_codes) / expected_unique) if expected_unique else None

        result.update({
            "basereport_expected_marker_count_total": int(sum(expected_counts.values())),
            "basereport_expected_unique_marker_codes": int(expected_unique),
            "basereport_observed_unique_event_codes": int(observed_unique),
            "basereport_missing_codes": [int(x) for x in missing_codes],
            "basereport_extra_observed_codes": [int(x) for x in extra_codes],
            "basereport_count_mismatch_codes": [int(x) for x in mismatch_codes],
            "basereport_n_missing_codes": int(len(missing_codes)),
            "basereport_n_extra_observed_codes": int(len(extra_codes)),
            "basereport_n_count_mismatch_codes": int(len(mismatch_codes)),
            "basereport_missing_code_ratio": safe_float(missing_ratio),
        })

        if expected_unique == 0:
            result["basereport_status"] = "WARN"
            result["basereport_notes"] = "BaseReport found, but no expected marker codes were extracted"
        elif missing_ratio is not None and missing_ratio >= params.get("fail_missing_code_ratio", 0.40):
            result["basereport_status"] = "FAIL"
            result["basereport_notes"] = "large fraction of BaseReport marker codes is missing from raw event stream"
        elif missing_ratio is not None and missing_ratio >= params.get("warn_missing_code_ratio", 0.10):
            result["basereport_status"] = "WARN"
            result["basereport_notes"] = "some BaseReport marker codes are missing from raw event stream"
        elif mismatch_codes:
            result["basereport_status"] = "WARN"
            result["basereport_notes"] = "all expected marker codes are present, but some marker counts differ from BaseReport"
        else:
            result["basereport_status"] = "PASS"
            result["basereport_notes"] = "BaseReport marker codes are present in raw event stream"

        summary_json_path = os.path.join(output_dir, "basereport_summary.json")
        json_ready = dict(result)
        json_ready.pop("basereport_comparison_table_html", None)
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(json_ready, f, ensure_ascii=False, indent=2)
        result["basereport_summary_json"] = os.path.basename(summary_json_path)

        return result

    except Exception as e:
        result["basereport_status"] = "WARN"
        result["basereport_error"] = str(e)
        result["basereport_notes"] = "BaseReport validation failed with exception"
        return result

# ============================================================
# Aggregation
# ============================================================

def compute_overall_status(result):
    reasons = []
    status = "PASS"

    if result.get("metadata_status") == "FAIL":
        return "FAIL", ["metadata_qc_failed"]

    n_bad = result.get("N_bad_channels", 0) or 0
    n_eeg = result.get("n_channels_eeg", 0) or 0

    if n_eeg > 0:
        bad_ratio = n_bad / n_eeg
    else:
        bad_ratio = 1.0

    if bad_ratio >= 0.30:
        status = "FAIL"
        reasons.append("too_many_bad_channels")
    elif bad_ratio >= 0.10:
        status = "WARN"
        reasons.append("elevated_bad_channel_ratio")

    if result.get("line_noise_ratio_max") is not None and result["line_noise_ratio_max"] > 0.15:
        status = max_status(status, "WARN")
        reasons.append("elevated_line_noise_ratio")

    if result.get("muscle_ratio_max") is not None and result["muscle_ratio_max"] > 0.35:
        status = max_status(status, "WARN")
        reasons.append("elevated_muscle_ratio")

    if result.get("epoch_qc_error"):
        status = max_status(status, "WARN")
        reasons.append("epoch_qc_error")

    if result.get("ica_error"):
        status = max_status(status, "WARN")
        reasons.append("ica_or_iclabel_error")

    if result.get("iclabel_n_exclude", 0) and result.get("iclabel_n_exclude", 0) >= 5:
        status = max_status(status, "WARN")
        reasons.append("many_artifact_ica_components")

    if result.get("basereport_status") == "FAIL":
        status = max_status(status, "FAIL")
        reasons.append("basereport_event_validation_failed")
    elif result.get("basereport_status") == "WARN":
        status = max_status(status, "WARN")
        reasons.append("basereport_event_validation_warning")

    if result.get("protocol_marker_status") == "FAIL":
        status = max_status(status, "FAIL")
        reasons.append("protocol_marker_validation_failed")
    elif result.get("protocol_marker_status") == "WARN":
        status = max_status(status, "WARN")
        reasons.append("protocol_marker_validation_warning")
    elif result.get("protocol_marker_status") == "ERROR":
        status = max_status(status, "WARN")
        reasons.append("protocol_marker_validation_error")

    if not reasons:
        reasons.append("no_critical_issues_detected")

    return status, reasons


def max_status(a, b):
    order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    return a if order[a] >= order[b] else b



# ============================================================
# Quality assessment registry database
# ============================================================

def _quality_db_json(value):
    """Сериализует сложные значения для записи в SQLite."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _ratio_status(value, warn=0.10, fail=0.30, none_status="NA"):
    """PASS/WARN/FAIL для долевых метрик, где больше = хуже."""
    if value is None or value == "":
        return none_status
    try:
        v = float(value)
    except Exception:
        return none_status
    if not np.isfinite(v):
        return none_status
    if v >= fail:
        return "FAIL"
    if v >= warn:
        return "WARN"
    return "PASS"


def _derive_component_statuses(result):
    """
    Формирует отдельные статусы по составляющим QC-процесса.

    Эти поля не меняют исходные метрики и итоговый overall_qc_status.
    Они нужны для БД/HTML, чтобы было видно, какой блок дал PASS/WARN/FAIL/SKIPPED/ERROR.
    """
    statuses = {}

    # 1) Metadata QC уже имеет собственный статус.
    statuses["metadata_qc_status"] = result.get("metadata_status") or "UNKNOWN"

    # 2) Montage QC уже имеет собственный статус, но в коде успешный вариант может называться OK.
    montage_status = result.get("montage_status")
    if montage_status == "OK":
        statuses["montage_qc_status"] = "PASS"
    elif montage_status in {"WARN", "FAIL"}:
        statuses["montage_qc_status"] = montage_status
    else:
        statuses["montage_qc_status"] = "UNKNOWN"

    # 3) Bad-channel QC: общий технический статус по доле плохих каналов.
    n_bad = result.get("N_bad_channels", 0) or 0
    n_eeg = result.get("n_channels_eeg", 0) or 0
    bad_ratio = None
    try:
        bad_ratio = float(n_bad) / float(n_eeg) if float(n_eeg) > 0 else None
    except Exception:
        bad_ratio = None
    statuses["bad_channel_qc_status"] = _ratio_status(bad_ratio, warn=0.10, fail=0.30, none_status="UNKNOWN")
    statuses["bad_channel_ratio"] = safe_float(bad_ratio) if bad_ratio is not None else None

    # 4) Continuous spectral QC: сейчас нет отдельного поля status, поэтому выводится из line/muscle ratios.
    cont_status = "PASS"
    if result.get("line_noise_ratio_max") is not None:
        try:
            if float(result.get("line_noise_ratio_max")) > 0.15:
                cont_status = max_status(cont_status, "WARN")
        except Exception:
            pass
    if result.get("muscle_ratio_max") is not None:
        try:
            if float(result.get("muscle_ratio_max")) > 0.35:
                cont_status = max_status(cont_status, "WARN")
        except Exception:
            pass
    statuses["continuous_spectral_qc_status"] = cont_status

    # 5) Epoch QC: basic epoch creation/metrics.
    if result.get("epoch_qc_error"):
        statuses["epoch_qc_status"] = "WARN"
    elif result.get("epoch_qc_enabled") is False:
        statuses["epoch_qc_status"] = "SKIPPED"
    elif (result.get("n_epochs") or 0) == 0:
        statuses["epoch_qc_status"] = "WARN"
    else:
        statuses["epoch_qc_status"] = "PASS"

    # 6) FASTER: если библиотека/блок недоступен — SKIPPED/ERROR; иначе статус по доле плохих fixed epochs.
    if result.get("faster_enabled") is False:
        statuses["faster_qc_status"] = "SKIPPED"
    elif result.get("faster_available") is False:
        statuses["faster_qc_status"] = "SKIPPED"
    elif result.get("faster_error"):
        statuses["faster_qc_status"] = "ERROR"
    else:
        statuses["faster_qc_status"] = _ratio_status(result.get("faster_bad_epoch_ratio"), warn=0.10, fail=0.30, none_status="PASS")

    # 7) AutoReject: отдельный статус запуска + статус по доле rejected epochs, если посчитано.
    ar_status = result.get("autoreject_status") or "SKIPPED"
    if str(ar_status).upper() in {"SKIPPED", "ERROR"}:
        statuses["autoreject_qc_status"] = str(ar_status).upper()
    elif result.get("autoreject_error"):
        statuses["autoreject_qc_status"] = "ERROR"
    else:
        local_status = _ratio_status(result.get("autoreject_local_reject_ratio"), warn=0.10, fail=0.30, none_status="PASS")
        global_status = _ratio_status(result.get("autoreject_global_reject_ratio"), warn=0.10, fail=0.30, none_status="PASS")
        statuses["autoreject_qc_status"] = max_status(local_status, global_status)

    # 8) BaseReport/.vmrk validation уже имеет собственный статус.
    statuses["basereport_qc_status"] = result.get("basereport_status") or "SKIPPED"
    statuses["protocol_marker_qc_status"] = result.get("protocol_marker_status") or "SKIPPED"

    # 9) ICA/ICLabel: ошибка = WARN; много auto-exclude = WARN; иначе PASS/SKIPPED.
    if result.get("icalabel_enabled") is False:
        statuses["ica_iclabel_qc_status"] = "SKIPPED"
    elif result.get("ica_error"):
        statuses["ica_iclabel_qc_status"] = "WARN"
    elif result.get("ica_n_components", 0) in (None, 0):
        statuses["ica_iclabel_qc_status"] = "SKIPPED"
    elif (result.get("iclabel_n_exclude") or 0) >= 5:
        statuses["ica_iclabel_qc_status"] = "WARN"
    else:
        statuses["ica_iclabel_qc_status"] = "PASS"

    # 10) Epoch-level spectral variants: эти статусы уже считаются в compute_epoch_spectral_variant.
    statuses["epoch_spectral_raw_pre_ica_qc_status"] = result.get("epoch_spectral_raw_pre_ica_status") or "SKIPPED"
    statuses["epoch_spectral_ica_cleaned_qc_status"] = result.get("epoch_spectral_ica_cleaned_status") or "SKIPPED"
    statuses["continuous_ica_cleaned_qc_status"] = result.get("continuous_ica_cleaned_status") or "SKIPPED"

    # 11) Target-channel QC, если такая версия конвейера уже содержит этот блок.
    if result.get("target_channel_status") is not None:
        statuses["target_channel_qc_status"] = result.get("target_channel_status")
    else:
        statuses["target_channel_qc_status"] = "NOT_IMPLEMENTED_IN_THIS_FILE"

    # JSON-сводка для удобного просмотра одной колонкой в SQLite.
    statuses["component_qc_statuses_json"] = {
        k: v for k, v in statuses.items()
        if k.endswith("_qc_status")
    }
    return statuses


def _ensure_sqlite_columns(con, table_name, row):
    """Мягкая миграция: добавляет новые колонки в уже существующую SQLite-таблицу."""
    existing = set()
    try:
        info = con.execute(f"PRAGMA table_info({table_name})").fetchall()
        existing = {str(x[1]) for x in info}
    except Exception:
        existing = set()

    for col in row.keys():
        if col in existing:
            continue
        con.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} TEXT")


def _make_quality_conclusion(result):
    """
    Формирует краткие машинночитаемые и человекочитаемые выводы о качестве
    на основе уже рассчитанных QC-блоков. Не меняет сами метрики.
    """
    status = result.get("overall_qc_status") or "UNKNOWN"
    main_issues = []
    positive_findings = []
    recommendations = []

    n_bad = result.get("N_bad_channels") or 0
    n_eeg = result.get("n_channels_eeg") or 0
    if n_eeg:
        bad_ratio = n_bad / n_eeg
        if bad_ratio >= 0.30:
            main_issues.append(f"критически высокая доля плохих каналов: {n_bad}/{n_eeg}")
            recommendations.append("проверить контакт электродов, montage и рассмотреть исключение/повтор записи")
        elif bad_ratio >= 0.10:
            main_issues.append(f"повышенная доля плохих каналов: {n_bad}/{n_eeg}")
            recommendations.append("проверить список плохих каналов и возможность интерполяции")
        else:
            positive_findings.append(f"доля плохих каналов не превышает 10%: {n_bad}/{n_eeg}")

    montage_status = result.get("montage_status")
    if montage_status in {"WARN", "FAIL"}:
        main_issues.append(f"проблема montage/координат электродов: {result.get('montage_warning')}")
        recommendations.append("проверить .elc-файл, имена каналов и каналы без координат")
    elif montage_status == "OK":
        positive_findings.append("montage применён без критических ошибок")

    br_status = result.get("basereport_status")
    if br_status == "FAIL":
        main_issues.append("BaseReport/.vmrk validation: FAIL")
        recommendations.append("проверить соответствие BaseReport, .vmrk и сценарных marker codes")
    elif br_status == "WARN":
        main_issues.append("BaseReport/.vmrk validation: WARN")
        recommendations.append("проверить пропущенные или расходящиеся marker counts")
    elif br_status == "PASS":
        positive_findings.append("BaseReport marker codes сопоставлены с raw event stream")

    for key, label in [
        ("faster_bad_epoch_ratio", "FASTER"),
        ("autoreject_local_reject_ratio", "AutoReject local"),
        ("epoch_spectral_raw_pre_ica_bad_epoch_ratio", "epoch spectral before ICA"),
        ("epoch_spectral_ica_cleaned_bad_epoch_ratio", "epoch spectral after ICA-copy"),
    ]:
        val = result.get(key)
        if val is None or val == "":
            continue
        try:
            val_f = float(val)
        except Exception:
            continue
        if val_f >= 0.30:
            main_issues.append(f"{label}: высокая доля плохих эпох ({val_f:.3f})")
        elif val_f >= 0.10:
            main_issues.append(f"{label}: умеренная доля плохих эпох ({val_f:.3f})")
        else:
            positive_findings.append(f"{label}: низкая доля плохих эпох ({val_f:.3f})")

    if result.get("ica_error"):
        main_issues.append(f"ICA/ICLabel error: {result.get('ica_error')}")
        recommendations.append("проверить montage, число EEG-каналов и параметры ICA/ICLabel")
    else:
        if result.get("ica_n_components"):
            positive_findings.append(f"ICA/ICLabel выполнен, компонент: {result.get('ica_n_components')}")
        if result.get("iclabel_n_exclude"):
            main_issues.append(f"ICLabel предложил исключить компонент: {result.get('iclabel_n_exclude')}")
            recommendations.append("проверить автоматически предложенные ICA-компоненты перед окончательной очисткой")

    if not main_issues:
        main_issues.append("критические проблемы качества не выявлены")
    if not recommendations:
        if status == "PASS":
            recommendations.append("запись можно использовать для дальнейшего анализа с учётом стандартного визуального контроля")
        else:
            recommendations.append("провести ручную проверку отмеченных QC-блоков")

    if status == "PASS":
        conclusion = "Запись прошла автоматическую оценку качества."
    elif status == "WARN":
        conclusion = "Запись требует проверки: обнаружены предупреждения качества."
    elif status == "FAIL":
        conclusion = "Запись не прошла автоматическую оценку качества."
    else:
        conclusion = "Статус качества записи не определён."

    conclusion += " Основные замечания: " + "; ".join(main_issues[:6]) + "."
    if positive_findings:
        conclusion += " Положительные признаки: " + "; ".join(positive_findings[:5]) + "."
    conclusion += " Рекомендации: " + "; ".join(recommendations[:5]) + "."

    return {
        "quality_conclusion_text": conclusion,
        "quality_main_issues": main_issues,
        "quality_positive_findings": positive_findings,
        "quality_recommendations": recommendations,
    }


def save_quality_assessment_to_db(result, params):
    """
    Сохраняет итоговые выводы QC в отдельную SQLite-БД рядом с этим скриптом.
    Не меняет вычисления QC, HTML, FASTER, AutoReject, ICA или BaseReport.
    """
    db_params = params or {}
    if not db_params.get("enabled", True):
        return {"quality_db_path": None, "quality_db_status": "disabled", "quality_db_error": None}

    script_dir = Path(__file__).resolve().parent
    db_path = script_dir / db_params.get("sqlite_filename", "qc_quality_assessment.sqlite")
    table_name = db_params.get("table_name", "qc_quality_assessments")

    component_statuses = _derive_component_statuses(result)
    result.update(component_statuses)

    conclusions = _make_quality_conclusion(result)
    result.update(conclusions)

    record_name = os.path.splitext(os.path.basename(str(result.get("Record") or "")))[0]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "assessment_datetime": now,
        "record_name": record_name,
        "record_path": result.get("Record"),
        "record_type": result.get("record_type"),
        "prefix": result.get("prefix"),
        "participant_id": result.get("id"),
        "visit_num": result.get("visit_num"),
        "scenario": result.get("scenario"),
        "overall_qc_status": result.get("overall_qc_status"),
        "overall_qc_reasons": _quality_db_json(result.get("overall_qc_reasons")),
        "component_qc_statuses_json": _quality_db_json(result.get("component_qc_statuses_json")),
        "metadata_qc_status": result.get("metadata_qc_status"),
        "montage_qc_status": result.get("montage_qc_status"),
        "bad_channel_qc_status": result.get("bad_channel_qc_status"),
        "bad_channel_ratio": result.get("bad_channel_ratio"),
        "continuous_spectral_qc_status": result.get("continuous_spectral_qc_status"),
        "epoch_qc_status": result.get("epoch_qc_status"),
        "faster_qc_status": result.get("faster_qc_status"),
        "autoreject_qc_status": result.get("autoreject_qc_status"),
        "basereport_qc_status": result.get("basereport_qc_status"),
        "protocol_marker_qc_status": result.get("protocol_marker_qc_status"),
        "ica_iclabel_qc_status": result.get("ica_iclabel_qc_status"),
        "epoch_spectral_raw_pre_ica_qc_status": result.get("epoch_spectral_raw_pre_ica_qc_status"),
        "epoch_spectral_ica_cleaned_qc_status": result.get("epoch_spectral_ica_cleaned_qc_status"),
        "continuous_ica_cleaned_qc_status": result.get("continuous_ica_cleaned_qc_status"),
        "target_channel_qc_status": result.get("target_channel_qc_status"),
        "quality_conclusion_text": result.get("quality_conclusion_text"),
        "quality_main_issues": _quality_db_json(result.get("quality_main_issues")),
        "quality_positive_findings": _quality_db_json(result.get("quality_positive_findings")),
        "quality_recommendations": _quality_db_json(result.get("quality_recommendations")),
        "duration": result.get("duration"),
        "n_channels_eeg": result.get("n_channels_eeg"),
        "n_bad_channels": result.get("N_bad_channels"),
        "all_bad_channels": _quality_db_json(result.get("all_bad_channels")),
        "montage_status": result.get("montage_status"),
        "montage_warning": result.get("montage_warning"),
        "basereport_status": result.get("basereport_status"),
        "basereport_notes": result.get("basereport_notes"),
        "protocol_marker_status": result.get("protocol_marker_status"),
        "protocol_marker_notes": result.get("protocol_marker_notes"),
        "protocol_marker_missing_required_codes": _quality_db_json(result.get("protocol_marker_missing_required_codes")),
        "protocol_marker_required_codes": _quality_db_json(result.get("protocol_marker_required_codes")),
        "protocol_marker_observed_codes": _quality_db_json(result.get("protocol_marker_observed_codes")),
        "faster_bad_epoch_ratio": result.get("faster_bad_epoch_ratio"),
        "autoreject_status": result.get("autoreject_status"),
        "autoreject_local_reject_ratio": result.get("autoreject_local_reject_ratio"),
        "ica_n_components": result.get("ica_n_components"),
        "iclabel_n_exclude": result.get("iclabel_n_exclude"),
        "iclabel_n_review": result.get("iclabel_n_review"),
        "epoch_spectral_raw_pre_ica_bad_epoch_ratio": result.get("epoch_spectral_raw_pre_ica_bad_epoch_ratio"),
        "epoch_spectral_ica_cleaned_bad_epoch_ratio": result.get("epoch_spectral_ica_cleaned_bad_epoch_ratio"),
        "report_folder": result.get("report_folder"),
        "improved_report_path": result.get("improved_report_path"),
        "mne_report_path": result.get("mne_report_path"),
    }

    columns_sql = ",\n                ".join([f"{col} TEXT" for col in row.keys()])
    placeholders = ", ".join(["?"] * len(row))
    quoted_cols = ", ".join(row.keys())

    try:
        with sqlite3.connect(str(db_path)) as con:
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    {columns_sql}
                )
            """)
            _ensure_sqlite_columns(con, table_name, row)
            con.execute(
                f"INSERT INTO {table_name} ({quoted_cols}) VALUES ({placeholders})",
                [None if v is None else str(v) for v in row.values()],
            )
            con.commit()
        return {"quality_db_path": str(db_path), "quality_db_status": "saved", "quality_db_error": None}
    except Exception as e:
        return {"quality_db_path": str(db_path), "quality_db_status": "error", "quality_db_error": str(e)}


# ============================================================
# HTML report
# ============================================================

def html_escape(x):
    if x is None:
        return ""
    return (
        str(x)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )



def make_bad_channels_epochs_html(result):
    rows = [
        {
            "method": "FASTER",
            "available": result.get("faster_available"),
            "status": "computed" if result.get("faster_error") in (None, "") else "warning/error",
            "bad_channels": result.get("faster_n_bad"),
            "bad_epochs": result.get("faster_n_bad_epochs"),
            "bad_epoch_ratio": result.get("faster_bad_epoch_ratio"),
            "outputs": ", ".join([str(x) for x in [
                result.get("faster_channel_metrics_csv"),
                result.get("faster_epoch_metrics_csv"),
                result.get("faster_bad_epochs_csv"),
                result.get("faster_summary_json"),
            ] if x]),
            "notes": result.get("faster_error") or "fixed-length epochs; independent from event marker semantics",
        },
        {
            "method": "AutoReject local",
            "available": result.get("autoreject_available"),
            "status": result.get("autoreject_status"),
            "bad_channels": "epoch-wise",
            "bad_epochs": result.get("autoreject_local_n_epochs_rejected"),
            "bad_epoch_ratio": result.get("autoreject_local_reject_ratio"),
            "outputs": ", ".join([str(x) for x in [
                result.get("autoreject_epoch_log_csv"),
                result.get("autoreject_channel_log_csv"),
                result.get("autoreject_condition_summary_csv"),
            ] if x]),
            "notes": result.get("autoreject_error") or result.get("autoreject_subset_note") or "local repair/drop on event-locked epochs",
        },
        {
            "method": "AutoReject global",
            "available": result.get("autoreject_available"),
            "status": result.get("autoreject_status"),
            "bad_channels": "global threshold",
            "bad_epochs": result.get("autoreject_global_n_epochs_rejected"),
            "bad_epoch_ratio": result.get("autoreject_global_reject_ratio"),
            "outputs": ", ".join([str(x) for x in [
                result.get("autoreject_global_thresholds_json"),
                result.get("autoreject_summary_json"),
            ] if x]),
            "notes": result.get("autoreject_error") or result.get("autoreject_subset_note") or "global peak-to-peak threshold",
        },
    ]
    html = "<h2>Bad channels / epochs QC</h2>"
    html += "<h3>FASTER vs AutoReject machine-readable QC</h3>"
    html += make_html_table(rows, columns=["method", "available", "status", "bad_channels", "bad_epochs", "bad_epoch_ratio", "outputs", "notes"])
    comp_csv = result.get("bad_channel_method_comparison_csv")
    if comp_csv:
        html += f"<p><b>Bad channel method comparison CSV:</b> <code>{html_escape(comp_csv)}</code></p>"
    html += "<p class='note'><b>AutoReject guard:</b> AutoReject is skipped if the number of events exceeds <code>hard_event_limit</code>; if events exceed <code>max_epochs</code>, a stratified subset is used. This protects the VM from non-rationally large epoch counts without changing MMN/BaseReport marker parsing.</p>"
    return html

def write_improved_html(result, output_path):
    def row(k, v):
        return f"<tr><th>{html_escape(k)}</th><td>{html_escape(v)}</td></tr>"

    def simple_table(items, columns):
        return make_html_table(items, columns=columns)

    comp_table = result.get("iclabel_component_table", [])
    if comp_table:
        ic_table_html = simple_table(comp_table, ["component", "label", "probability", "decision", "decision_rule", "reason"])
    else:
        ic_table_html = "<p>ICLabel table is unavailable.</p>"

    exclude_table_html = simple_table(
        result.get("iclabel_exclude_table", []),
        ["component", "label", "probability", "decision", "decision_rule", "reason"],
    )

    spectral_variant_table = result.get("spectral_variant_table", [])
    spectral_variant_html = simple_table(
        spectral_variant_table,
        ["variant", "label", "stage", "epoching", "input", "status", "n_epochs", "bad_epoch_ratio", "psd_image", "artifact_image", "csv", "notes"],
    )

    images_html = ""
    for title, key in [
        ("Main continuous PSD before ICA", "band_power_image"),
        ("Main continuous artifact ratios before ICA", "artifact_ratios_image"),
        ("ICA components", "ica_components_image"),
    ]:
        img = result.get(key)
        if img:
            images_html += f"""
            <h3>{html_escape(title)}</h3>
            <img src="{html_escape(img)}" style="max-width:100%; border:1px solid #ddd;">
            """

    for item in spectral_variant_table:
        for img_key, img_title in [("psd_image", "PSD / mean spectrum"), ("artifact_image", "Artifact ratios / epoch-frequency heatmap")]:
            img = item.get(img_key)
            if img:
                images_html += f"""
                <h3>{html_escape(item.get('label'))}: {html_escape(img_title)}</h3>
                <p><b>Variant:</b> {html_escape(item.get('variant'))}; <b>Stage:</b> {html_escape(item.get('stage'))}; <b>Epoching:</b> {html_escape(item.get('epoching'))}</p>
                <img src="{html_escape(img)}" style="max-width:100%; border:1px solid #ddd;">
                """

    for title, key in [
        ("Before/after ICA: mean PSD across epochs", "epoch_spectral_before_after_ica_mean_psd_image"),
        ("Before/after ICA: bad epoch ratio", "epoch_spectral_before_after_ica_bad_epoch_ratio_image"),
    ]:
        img = result.get(key)
        if img:
            images_html += f"""
            <h3>{html_escape(title)}</h3>
            <img src="{html_escape(img)}" style="max-width:100%; border:1px solid #ddd;">
            """

    basereport_html = ""
    if result.get("basereport_enabled") is not None:
        basereport_html = make_basereport_summary_html(result)

    protocol_marker_html = ""
    if result.get("protocol_marker_enabled") is not None:
        protocol_marker_html = make_protocol_marker_summary_html(result)

    bad_channels_epochs_html = make_bad_channels_epochs_html(result)

    summary_keys = [
        "overall_qc_status", "overall_qc_reasons",
        "component_qc_statuses_json",
        "metadata_qc_status", "montage_qc_status", "bad_channel_qc_status", "bad_channel_ratio",
        "continuous_spectral_qc_status", "epoch_qc_status", "faster_qc_status", "autoreject_qc_status",
        "basereport_qc_status", "protocol_marker_qc_status", "ica_iclabel_qc_status", "epoch_spectral_raw_pre_ica_qc_status",
        "epoch_spectral_ica_cleaned_qc_status", "continuous_ica_cleaned_qc_status", "target_channel_qc_status",
        "quality_conclusion_text", "quality_main_issues", "quality_recommendations", "quality_db_path", "quality_db_status", "quality_db_error",
        "elc_file_found", "elc_path", "montage_source", "montage_status", "montage_warning", "montage_missing_eeg_channels",
        "id", "visit_num", "scenario", "duration", "n_channels_eeg", "N_bad_channels", "faster_n_bad",
        "mean_amplitude_uv", "max_ptp_amplitude_uv", "dominant_frequency_hz",
        "alpha_beta_ratio", "theta_alpha_ratio", "blink_ratio_max", "muscle_ratio_max", "line_noise_ratio_max",
        "n_events", "n_epochs",
        "basereport_found", "basereport_status", "basereport_expected_unique_marker_codes", "basereport_observed_unique_event_codes",
        "protocol_marker_status", "protocol_marker_spec_key", "protocol_marker_missing_required_codes", "protocol_marker_count_mismatch_codes",
        "basereport_missing_code_ratio", "basereport_n_missing_codes", "basereport_n_count_mismatch_codes",
        "ica_n_components", "iclabel_n_exclude", "iclabel_n_review", "iclabel_suggested_exclude", "iclabel_exclude_summary",
        "epoch_spectral_raw_pre_ica_status", "epoch_spectral_raw_pre_ica_n_epochs", "epoch_spectral_raw_pre_ica_bad_epoch_ratio",
        "continuous_ica_cleaned_status",
        "epoch_spectral_ica_cleaned_status", "epoch_spectral_ica_cleaned_n_epochs", "epoch_spectral_ica_cleaned_bad_epoch_ratio",
    ]

    summary_table = make_html_table(
        [{"parameter": key, "value": result.get(key)} for key in summary_keys],
        columns=["parameter", "value"],
        table_class="qc-table qc-kv",
    )

    html = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>Improved EEG QC Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; background: #f5f5f5; }}
            .container {{ background: white; padding: 24px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.08); }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; font-size: 13px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
            th {{ background: #eee; text-align: left; }}
            .status {{ font-size: 24px; font-weight: bold; padding: 12px; border-radius: 8px; background: #eee; display: inline-block; }}
            .note {{ background: #fff7d6; padding: 12px; border-left: 5px solid #e1b12c; margin-bottom: 18px; }}
            code {{ background: #eee; padding: 2px 4px; border-radius: 4px; }}
        </style>
    </head>
    <body>
    <div class="container">
        <h1>Improved EEG QC Report</h1>
        <p class="status">Overall QC status: {html_escape(result.get("overall_qc_status"))}</p>
        <div class="note">
            <b>Важно:</b> в одном отчёте объединены BaseReport/Event-ID validation, continuous spectral QC,
            epoch-level spectral QC, ICA/ICLabel и сравнения до/после ICA. Варианты после ICA считаются
            на ICA-prepared copy и используются как диагностическое сравнение.
        </div>
        <h2>Summary</h2>
        {summary_table}
        {bad_channels_epochs_html}
        <h2>Spectral analysis variants</h2>
        {spectral_variant_html}
        <h2>Figures</h2>
        {images_html}
        {basereport_html}
        {protocol_marker_html}
        <h2>ICA components suggested for exclusion</h2>
        <p>{html_escape(result.get("iclabel_exclude_summary"))}</p>
        {exclude_table_html}
        <h2>ICLabel component classification</h2>
        {ic_table_html}
        <h2>Notes</h2>
        <p>
            BaseReport validation is used when a matching BaseReport is found under visit/event id/&lt;scenario folder&gt;.
            If BaseReport is missing or cannot be parsed, event metrics still fall back to raw annotations / vmrk events.
            Epoch-level spectral QC currently uses basic event-locked epochs from annotations.
        </p>
    </div>
    </body>
    </html>
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)




# ============================================================
# Main improved QC
# ============================================================

def compute_improved_qc(
    records,
    qc_dataframe_file,
    config_dir,
    hot_qc=True,
    exist_ok=True,
):
    config = get_params_config(config_dir)
    qc_params = config.get("Quality_Check", {})
    improved_params = deep_update(
        DEFAULT_IMPROVED_QC_PARAMS,
        config.get("Improved_QC", {}),
    )

    with tqdm(records, total=len(records)) as progress_bar:
        for idx, record in enumerate(progress_bar):
            try:
                (
                    PARTICIPANT_PATH,
                    VISIT_PATH,
                    EXPERIMENT_PATH,
                    FILE_PATH,
                    RAW_PATH,
                    PREPROCESSED_PATH,
                    PROCESSED_PATH,
                    visit_name,
                    experiment,
                ) = get_local_veriable(record)

                elc_files = glob.glob(os.path.join(VISIT_PATH, "*.elc"))
                ELC_PATH = elc_files[0] if elc_files else None

                qc_path = os.path.join(EXPERIMENT_PATH, "QC_improved")
                if exist_ok and glob.glob(f"{qc_path}/**/improved_dossier.html", recursive=True):
                    continue
                if hot_qc and os.path.exists(qc_path):
                    shutil.rmtree(qc_path)
                os.makedirs(qc_path, exist_ok=True)

                folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                folder_path = os.path.join(qc_path, folder_name)
                os.makedirs(folder_path, exist_ok=True)

                progress_bar.set_description("Loading raw")
                raw = mne.io.read_raw_brainvision(FILE_PATH, preload=True)

                raw, montage_info = apply_montage_with_fallback(raw, elc_path=ELC_PATH, fallback_name="standard_1020")

                meta_dict = build_meta_dict(
                    file_path=FILE_PATH,
                    visit_name=visit_name,
                    experiment=experiment,
                    visit_path=VISIT_PATH,
                )

                # ---------- Metadata QC ----------
                progress_bar.set_description("Metadata QC")
                metadata_qc = compute_metadata_qc(raw=raw, file_path=FILE_PATH, elc_path=ELC_PATH, meta_dict=meta_dict)

                # ---------- Original QualityChecker ----------
                progress_bar.set_description("Original QualityChecker")
                q_checker = QualityChecker(**qc_params)
                q_checker.check(FILE_PATH, ELC_PATH, folder_path, scenarious_name=meta_dict["scenario"], progress_bar=progress_bar)
                original_qc_report = q_checker.get_report()

                # ---------- Continuous QC ----------
                progress_bar.set_description("Continuous QC")
                amplitude_metrics = compute_amplitude_metrics(raw)

                # ---------- FASTER ----------
                progress_bar.set_description("FASTER")
                faster_result = compute_faster_bad_channels(raw, improved_params["faster"], output_dir=folder_path)
                if improved_params.get("faster", {}).get("apply_bad_channels_to_raw", True) and faster_result.get("faster_bad_channels"):
                    raw.info["bads"] = sorted(set(raw.info.get("bads", [])) | set(faster_result["faster_bad_channels"]))

                # ---------- Spectral QC: main continuous raw before ICA ----------
                progress_bar.set_description("Spectral QC")
                spectral = compute_band_power_and_artifacts(raw, improved_params["psd"])
                band_power_image = os.path.join(folder_path, "band_power.png")
                plot_band_power(spectral["freqs"], spectral["psds_mean"], spectral["band_power"], band_power_image)
                artifact_ratios_image = os.path.join(folder_path, "artifact_ratios.png")
                plot_artifact_ratios(spectral["artifact_ratio_by_channel"], artifact_ratios_image)

                spectral_variant_table = []
                spectral_variant_results = {}
                if improved_params.get("spectral_variants", {}).get("run_continuous_raw_pre_ica", True):
                    spectral_variant_table.append({
                        "variant": "continuous_raw_pre_ica",
                        "label": "Continuous spectral QC before ICA",
                        "stage": "before ICA",
                        "epoching": "no",
                        "input": "raw continuous EEG",
                        "status": "computed",
                        "n_epochs": "",
                        "bad_epoch_ratio": "",
                        "psd_image": os.path.basename(band_power_image),
                        "artifact_image": os.path.basename(artifact_ratios_image),
                        "csv": "",
                        "notes": "Main Welch PSD / band power / artifact ratios on continuous raw",
                    })

                # ---------- Epoch-level QC ----------
                progress_bar.set_description("Epoch QC")
                epoch_metrics = compute_epoch_metrics(raw, improved_params["epoch_qc"])

                # ---------- AutoReject QC ----------
                progress_bar.set_description("AutoReject")
                autoreject_result = compute_autoreject_qc(
                    raw=raw,
                    params=improved_params.get("autoreject", {}),
                    output_dir=folder_path,
                )

                # ---------- Epoch-level Spectral QC before ICA ----------
                if improved_params.get("spectral_variants", {}).get("enabled", True) and improved_params.get("spectral_variants", {}).get("run_epoch_raw_pre_ica", True):
                    progress_bar.set_description("Epoch Spectral QC before ICA")
                    epoch_spec_pre = compute_epoch_spectral_variant(
                        raw=raw,
                        epoch_params=improved_params["epoch_qc"],
                        psd_params=improved_params["psd"],
                        variant_params=improved_params.get("spectral_variants", {}),
                        output_dir=folder_path,
                        variant_key="epoch_spectral_raw_pre_ica",
                        variant_label="Epoch-level spectral QC before ICA",
                        stage_label="before ICA",
                    )
                    spectral_variant_results.update(epoch_spec_pre.get("summary", {}))
                    spectral_variant_table.append(epoch_spec_pre.get("variant_row", {}))

                # ---------- BaseReport / Event-ID QC ----------
                progress_bar.set_description("BaseReport QC")
                basereport_result = compute_basereport_qc(
                    raw=raw,
                    visit_path=VISIT_PATH,
                    file_path=FILE_PATH,
                    meta_dict=meta_dict,
                    output_dir=folder_path,
                    params=improved_params.get("basereport", {}),
                )

                # ---------- Protocol Marker QC ----------
                progress_bar.set_description("Protocol Marker QC")
                protocol_marker_result = compute_protocol_marker_qc(
                    raw=raw,
                    meta_dict=meta_dict,
                    output_dir=folder_path,
                    params=improved_params.get("protocol_marker_qc", {}),
                )

                # ---------- ICA + ICLabel ----------
                progress_bar.set_description("ICA + ICLabel")
                ica_result = run_ica_iclabel(raw, improved_params["icalabel"], folder_path)
                raw_ica_cleaned = ica_result.pop("_raw_ica_cleaned", None)

                # ---------- Spectral QC after ICA-cleaned copy ----------
                if raw_ica_cleaned is not None and improved_params.get("spectral_variants", {}).get("enabled", True) and improved_params.get("spectral_variants", {}).get("run_continuous_ica_cleaned", True):
                    progress_bar.set_description("Spectral QC after ICA")
                    cont_after = compute_continuous_spectral_variant(
                        raw=raw_ica_cleaned,
                        psd_params=improved_params["psd"],
                        output_dir=folder_path,
                        variant_key="continuous_ica_cleaned",
                        variant_label="Continuous spectral QC after ICA-cleaned copy",
                        stage_label="after ICA filtering",
                    )
                    spectral_variant_results.update(cont_after.get("summary", {}))
                    spectral_variant_table.append(cont_after.get("variant_row", {}))

                # ---------- Epoch-level Spectral QC after ICA-cleaned copy ----------
                if raw_ica_cleaned is not None and improved_params.get("spectral_variants", {}).get("enabled", True) and improved_params.get("spectral_variants", {}).get("run_epoch_ica_cleaned", True):
                    progress_bar.set_description("Epoch Spectral QC after ICA")
                    epoch_spec_after = compute_epoch_spectral_variant(
                        raw=raw_ica_cleaned,
                        epoch_params=improved_params["epoch_qc"],
                        psd_params=improved_params["psd"],
                        variant_params=improved_params.get("spectral_variants", {}),
                        output_dir=folder_path,
                        variant_key="epoch_spectral_ica_cleaned",
                        variant_label="Epoch-level spectral QC after ICA-cleaned copy",
                        stage_label="after ICA filtering",
                    )
                    spectral_variant_results.update(epoch_spec_after.get("summary", {}))
                    spectral_variant_table.append(epoch_spec_after.get("variant_row", {}))

                ica_defaults = {
                    "icalabel_enabled": True,
                    "icalabel_available": False,
                    "ica_method": None,
                    "ica_extended": None,
                    "ica_n_components": 0,
                    "iclabel_n_exclude": 0,
                    "iclabel_n_review": 0,
                    "iclabel_suggested_exclude": [],
                    "iclabel_review_components": [],
                    "iclabel_keep_components": [],
                    "iclabel_component_table": [],
                    "iclabel_exclude_table": [],
                    "iclabel_exclude_summary": "No ICA components met automatic exclusion criteria",
                    "iclabel_component_csv": None,
                    "iclabel_exclude_csv": None,
                    "ica_components_image": None,
                    "ica_error": None,
                }
                ica_result = {**ica_defaults, **ica_result}

                # ---------- Combine bad channels with machine-readable provenance ----------
                bad_channel_comparison_df, bad_channel_comparison_csv = build_bad_channel_method_comparison(
                    original_qc_report=original_qc_report,
                    faster_result=faster_result,
                    autoreject_result=autoreject_result,
                    raw=raw,
                    output_dir=folder_path,
                )
                bad_channel_provenance = collect_bad_channel_provenance(
                    original_qc_report=original_qc_report,
                    faster_result=faster_result,
                    raw=raw,
                    metadata_qc=metadata_qc,
                    qc_params=qc_params,
                    faster_params=improved_params.get("faster", {}),
                    autoreject_params=improved_params.get("autoreject", {}),
                    output_dir=folder_path,
                )

                # ---------- Final dict ----------
                result = {
                    "Start_time": folder_name,
                    **meta_dict,
                    **metadata_qc,
                    **montage_info,
                    **original_qc_report,
                    **qc_params,
                    **amplitude_metrics,
                    **faster_result,
                    **autoreject_result,
                    **spectral["spectral_scores"],
                    **epoch_metrics,
                    **spectral_variant_results,
                    **basereport_result,
                    **protocol_marker_result,
                    **ica_result,
                    **bad_channel_provenance,
                    "spectral_variant_table": spectral_variant_table,
                    "bad_channel_method_comparison_csv": bad_channel_comparison_csv,
                    "band_power_image": os.path.basename(band_power_image),
                    "artifact_ratios_image": os.path.basename(artifact_ratios_image),
                    "mne_report_path": "mne_report.html",
                    "improved_report_path": "improved_dossier.html",
                }

                # Сводные before/after картинки создаются после формирования result.
                comparison_images = plot_epoch_before_after_comparison(result, folder_path)
                if comparison_images:
                    result.update(comparison_images)

                overall_status, overall_reasons = compute_overall_status(result)
                result["overall_qc_status"] = overall_status
                result["overall_qc_reasons"] = overall_reasons

                result["report_folder"] = folder_path
                quality_db_result = save_quality_assessment_to_db(
                    result,
                    improved_params.get("quality_db", {}),
                )
                result.update(quality_db_result)

                # ---------- MNE Report ----------
                progress_bar.set_description("MNE Report")
                rep = Report(title=f"Improved EEG QC — {meta_dict.get('id') or 'Unknown'}")
                try:
                    rep.add_raw(raw, title="Raw EEG with marked bad channels", psd=True)
                except Exception:
                    pass

                try:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.semilogy(spectral["freqs"], spectral["psds_mean"], color="black", linewidth=1.6, label="Welch PSD")
                    band_colors = {
                        "delta": "#9ecae1",
                        "theta": "#a1d99b",
                        "alpha": "#fdae6b",
                        "beta": "#fdd0a2",
                        "gamma": "#bcbddc",
                    }
                    finite_y = np.asarray(spectral["psds_mean"])
                    finite_y = finite_y[np.isfinite(finite_y) & (finite_y > 0)]
                    y_top = float(np.nanmax(finite_y)) if finite_y.size else 1.0
                    for band, (fmin, fmax) in FREQ_BANDS.items():
                        ax.axvspan(fmin, fmax, color=band_colors.get(str(band).lower()), alpha=0.35, label=band)
                        ax.text((fmin + fmax) / 2, y_top, band, ha="center", va="top", fontsize=9)
                    ax.set_title("Welch PSD + EEG frequency bands")
                    ax.set_xlabel("Frequency, Hz")
                    ax.set_ylabel("Power, V²/Hz")
                    ax.legend(loc="best")
                    ax.grid(alpha=0.25)
                    rep.add_figure(fig, title="Welch PSD / Band Power", caption="Power spectral density and frequency-band quality metrics", section="Spectral QC")
                    plt.close(fig)
                except Exception:
                    pass

                try:
                    rep.add_html(make_bad_channels_epochs_html(result), title="FASTER / AutoReject QC", section="Bad channels / epochs QC")
                except Exception:
                    pass

                try:
                    if result.get("basereport_enabled") is not None:
                        rep.add_html(make_basereport_summary_html(result), title="BaseReport / Event-ID validation", section="Event / Trigger QC")
                    if result.get("protocol_marker_enabled") is not None:
                        rep.add_html(make_protocol_marker_summary_html(result), title="Protocol Marker QC", section="Event / Trigger QC")
                        if result.get("basereport_comparison_image"):
                            img_path = os.path.join(folder_path, result.get("basereport_comparison_image"))
                            try:
                                img = plt.imread(img_path)
                                fig, ax = plt.subplots(figsize=(12, 6))
                                ax.imshow(img)
                                ax.axis("off")
                                rep.add_figure(fig, title="BaseReport vs raw event stream", caption="Expected marker counts from BaseReport compared with observed .vmrk / annotations counts", section="Event / Trigger QC")
                                plt.close(fig)
                            except Exception:
                                pass
                except Exception:
                    pass

                try:
                    comp_table = result.get("iclabel_component_table", [])
                    if comp_table:
                        ic_html = make_html_table(pd.DataFrame(comp_table))
                    else:
                        ic_html = f"""
                        <p><b>ICLabel table is empty.</b></p>
                        <table>
                            <tr><th>icalabel_enabled</th><td>{result.get("icalabel_enabled")}</td></tr>
                            <tr><th>icalabel_available</th><td>{result.get("icalabel_available")}</td></tr>
                            <tr><th>ica_n_components</th><td>{result.get("ica_n_components")}</td></tr>
                            <tr><th>ica_error</th><td>{result.get("ica_error")}</td></tr>
                        </table>
                        """
                    rep.add_html(ic_html, title="ICLabel component classification", section="ICA QC")
                except Exception as e:
                    rep.add_html(f"<p><b>Failed to add ICLabel table:</b> {str(e)}</p>", title="ICLabel error", section="ICA QC")

                try:
                    if spectral_variant_table:
                        rep.add_html(make_html_table(pd.DataFrame(spectral_variant_table)), title="Spectral analysis variants", section="Spectral QC")
                except Exception:
                    pass

                try:
                    exclude_table = result.get("iclabel_exclude_table", [])
                    if exclude_table:
                        exclude_html = make_html_table(pd.DataFrame(exclude_table))
                    else:
                        exclude_html = f"<p>{html_escape(result.get('iclabel_exclude_summary'))}</p>"
                    rep.add_html(exclude_html, title="ICA components suggested for exclusion and reasons", section="ICA QC")
                except Exception:
                    pass

                try:
                    for title, key in [
                        ("Before/after ICA: mean PSD across epochs", "epoch_spectral_before_after_ica_mean_psd_image"),
                        ("Before/after ICA: bad epoch ratio", "epoch_spectral_before_after_ica_bad_epoch_ratio_image"),
                    ]:
                        if result.get(key):
                            img_path = os.path.join(folder_path, result.get(key))
                            img = plt.imread(img_path)
                            fig, ax = plt.subplots(figsize=(12, 8))
                            ax.imshow(img)
                            ax.axis("off")
                            rep.add_figure(fig, title=title, caption="Epoch-level spectral QC comparison", section="Spectral QC")
                            plt.close(fig)
                except Exception:
                    pass

                rep.save(os.path.join(folder_path, "mne_report.html"), overwrite=True)

                # ---------- HTML ----------
                page_path = os.path.join(folder_path, "improved_dossier.html")
                write_improved_html(result, page_path)

                # ---------- CSV ----------
                result_for_csv = dict(result)
                for heavy_key in [
                    "channel_names", "channel_ptp_uv", "artifact_ratio_by_channel",
                    "iclabel_component_table", "iclabel_exclude_table", "spectral_variant_table",
                    "event_id", "all_bad_channels", "metadata_warnings", "metadata_errors", "overall_qc_reasons", "component_qc_statuses_json",
                    "montage_missing_eeg_channels",
                    "basereport_event_roots", "basereport_scenario_dirs", "basereport_files", "basereport_sheets",
                    "basereport_missing_codes", "basereport_extra_observed_codes", "basereport_count_mismatch_codes",
                    "basereport_all_candidates", "basereport_scan_files_sample", "basereport_scan_xls_files_sample",
                    "basereport_comparison_table_html",
                    "quality_main_issues", "quality_positive_findings", "quality_recommendations",
                ]:
                    if heavy_key in result_for_csv:
                        result_for_csv[heavy_key] = json.dumps(result_for_csv[heavy_key], ensure_ascii=False)

                first_columns = [
                    "Start_time", "id", "visit_num", "scenario", "overall_qc_status",
                    "metadata_qc_status", "montage_qc_status", "bad_channel_qc_status", "continuous_spectral_qc_status",
                    "epoch_qc_status", "faster_qc_status", "autoreject_qc_status", "basereport_qc_status", "ica_iclabel_qc_status",
                    "epoch_spectral_raw_pre_ica_qc_status", "epoch_spectral_ica_cleaned_qc_status", "continuous_ica_cleaned_qc_status", "target_channel_qc_status",
                    "quality_conclusion_text", "quality_db_path", "quality_db_status", "montage_source", "montage_status",
                    "duration", "n_channels_eeg", "N_bad_channels", "faster_n_bad",
                    "mean_amplitude_uv", "max_ptp_amplitude_uv", "dominant_frequency_hz", "alpha_beta_ratio", "theta_alpha_ratio",
                    "blink_ratio_max", "muscle_ratio_max", "line_noise_ratio_max", "n_events", "n_epochs",
                    "basereport_found", "basereport_status", "basereport_expected_unique_marker_codes",
                    "basereport_observed_unique_event_codes", "basereport_missing_code_ratio", "basereport_n_missing_codes",
                    "basereport_n_count_mismatch_codes", "basereport_expected_markers_csv", "basereport_event_comparison_csv",
                    "faster_n_fixed_epochs", "faster_n_bad_epochs", "faster_bad_epoch_ratio", "faster_epoch_metrics_csv",
                    "autoreject_status", "autoreject_n_epochs_total", "autoreject_used_epoch_subset", "autoreject_subset_note",
                    "autoreject_local_n_epochs_rejected", "autoreject_local_reject_ratio", "autoreject_global_n_epochs_rejected", "autoreject_global_reject_ratio",
                    "autoreject_epoch_log_csv", "autoreject_channel_log_csv", "autoreject_condition_summary_csv", "bad_channel_method_comparison_csv",
                    "ica_n_components", "iclabel_n_exclude", "iclabel_n_review", "iclabel_exclude_summary",
                    "epoch_spectral_raw_pre_ica_status", "epoch_spectral_raw_pre_ica_n_epochs", "epoch_spectral_raw_pre_ica_bad_epoch_ratio",
                    "continuous_ica_cleaned_status", "continuous_ica_cleaned_alpha_beta_ratio", "continuous_ica_cleaned_muscle_ratio_max", "continuous_ica_cleaned_line_noise_ratio_max",
                    "epoch_spectral_ica_cleaned_status", "epoch_spectral_ica_cleaned_n_epochs", "epoch_spectral_ica_cleaned_bad_epoch_ratio",
                ]
                df = pd.DataFrame([result_for_csv])
                for col in first_columns:
                    if col not in df.columns:
                        df[col] = None
                cols = first_columns + [c for c in df.columns if c not in first_columns]
                df = df[cols]
                df.to_csv(qc_dataframe_file, mode="a", sep=";", index=False, header=not os.path.isfile(qc_dataframe_file), encoding="utf-8-sig")
                gc.collect()

            except Exception:
                error_details = traceback.format_exc()
                print(f"Ошибка improved QC: {idx} {record}\n{error_details}")
                continue
