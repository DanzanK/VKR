import os
import re
import glob
import shutil
import json
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

import eeg_auto_tools
from eeg_auto_tools.developments import QualityChecker, AutoCleaner, EpochsAnalysier
from eeg_auto_tools.scenarious import canonical_scenario, extract_visit_num_from_visit_folder

SCENARIO_ALIASES = {
    "ant": "ANT",
    "ants": "ANT",
    "riti": "RiTi",
    "risetime": "RiTi",
    "rise_time": "RiTi",
    "rise time": "RiTi",
    "mmn": "MMNs",
    "mmns": "MMNs",
    "rest": "Rest",
    "resting": "Rest",
    "restingstate": "Rest",
    "restingstateeeg": "Rest",
}

def canonical_scenario(name: str) -> str:
    if not name:
        return name
    s = name.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if re.fullmatch(r"rs[12]\d", s):
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
    records = []
    if len(analysis_ids) == 1 and analysis_ids[0] == '*':
        participants = [p for p in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, p))]
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


def plot_band_power(band_power, freqs, psds_mean, save_path=None):
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
                plot_band_power(band_power, freqs, psds_mean, band_plot_path)

                # MNE Report
                progress_bar.set_description('Создание MNE Report...')
                rep = Report(title=f'Отчёт качества EEG — {meta_dict.get("id") or "Unknown"}')
                rep.add_raw(raw, title='Raw данные (с помеченными bad каналами)', psd=True)
                if faster_bad_channels:
                    html = "<b>FASTER bad channels:</b> " + ", ".join(faster_bad_channels)
                else:
                    html = "<b>FASTER bad channels:</b> none"

                rep.add_html(html, title="FASTER", section="Качество сигнала")

                fig_band = plot_band_power(band_power, freqs, psds_mean, save_path=band_plot_path)

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

                # Комбинированный подсчёт плохих каналов
                original_bads = set(qc_report.get('HighAmp', []) + qc_report.get('LowAmp', []) +
                                    qc_report.get('Bridged', []) + qc_report.get('Noise_Rate', []))
                all_bads = original_bads.union(faster_bad_channels)
                data_dict['N_bad_channels'] = len(all_bads)

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
