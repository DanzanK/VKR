# QC_improved.py
import mne
from pathlib import Path

mne.utils.set_config("MNE_USE_CUDA", "false")

#from architecture import get_records, get_params_config
#from improved_qc import compute_improved_qc
from qc_pipeline import get_records, get_params_config, compute_improved_qc

SCRIPT_DIR = Path(__file__).resolve().parent

data_config_path = SCRIPT_DIR / "data_config.json"
config_path = SCRIPT_DIR / "config.json"

print("Чтение data_config...")
data_params = get_params_config(data_config_path)

DATA_PATH_RAW = Path(data_params["Data_path"])
DATA_PATH = DATA_PATH_RAW if DATA_PATH_RAW.is_absolute() else (SCRIPT_DIR / DATA_PATH_RAW).resolve()

analysis_experiments = data_params["Scenarious"]
analysis_ids = data_params["Participant_IDs"]
analysis_visits = data_params["Visits"]
hot_qc = data_params.get("hot", True)
exist_ok = data_params.get("exist_ok", True)

qc_dataframe_file = str(SCRIPT_DIR / "qc_report_improved.csv")

print(f"Папка данных: {DATA_PATH}")
print("Поиск EEG файлов...")
records = get_records(str(DATA_PATH), analysis_visits, analysis_experiments, analysis_ids)
print(f"Найдено записей: {len(records)}")

if len(records) == 0:
    print("Нет файлов для обработки.")
else:
    print("Запуск improved QC без BaseReport, но с ICLabel...")
    compute_improved_qc(
        records=records,
        qc_dataframe_file=qc_dataframe_file,
        config_dir=config_path,
        hot_qc=hot_qc,
        exist_ok=exist_ok,
    )
    print("Improved QC завершён.")