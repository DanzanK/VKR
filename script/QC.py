# QC.py
import mne
mne.utils.set_config('MNE_USE_CUDA', 'false')

from architecture import compute_qc, get_records, get_params_config

# Пути к конфигам
#data_config_path = r"C:\\Users\\DKyunkrikov\\Desktop\\QC_python\\lab machine\\script\\data_config.json"
data_config_path = r"C:\\Users\\Danzan\\Desktop\\quality check\\31.03.26\\script\\data_config.json"
config_path = r"config.json"  # должен быть в той же папке, что и QC.py
#
print("Чтение data_config...")
data_params = get_params_config(data_config_path)

DATA_PATH = data_params["Data_path"]
qc_dataframe_file = data_params["QC_DataFrame_path"]
analysis_experiments = data_params["Scenarious"]
analysis_ids = data_params["Participant_IDs"]
analysis_visits = data_params["Visits"]
hot_qc = data_params.get("hot", True)
exist_ok = data_params.get("exist_ok", True)

print("Параметры:")
print(data_params)
print("Поиск EEG файлов...")

records = get_records(DATA_PATH, analysis_visits, analysis_experiments, analysis_ids)
print(f"Найдено записей: {len(records)}")

if len(records) == 0:
    print("Нет файлов для обработки. Завершение.")
else:
    print("Запуск Quality Check с расширенными метриками (FASTER + Band Power + MNE Report)...")
    compute_qc(
        records=records,
        qc_dataframe_file=qc_dataframe_file,
        config_dir=config_path,
        hot_qc=hot_qc,
        exist_ok=exist_ok
    )

    print(f"Quality Check завершён. Результаты сохранены в:\n{qc_dataframe_file}")
    print("Отчёты (dossier.html + mne_report.html) созданы в папках QC/")