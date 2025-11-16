from pathlib import Path

# =============================
# BASE PROJECT DIRECTORIES
# =============================
# src/ ichida emas – bir daraja yuqoriga chiqamiz
BASE_DIR = Path(__file__).resolve().parent.parent

# =============================
# DATA DIRECTORIES
# =============================
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DATA_DIR = DATA_DIR / "clean"
MERGED_DATA_DIR = DATA_DIR / "merged"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# =============================
# MODEL DIRECTORY
# =============================
MODEL_DIR = BASE_DIR / "models"

# =============================
# Create required directories
# =============================
for folder in [
    DATA_DIR,
    RAW_DATA_DIR,
    CLEAN_DATA_DIR,
    MERGED_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR
]:
    folder.mkdir(exist_ok=True, parents=True)

# =============================
# OUTPUT PATHS
# =============================
MERGED_OUTPUT = MERGED_DATA_DIR / "merged_clean_data.csv"
FINAL_DATASET = PROCESSED_DATA_DIR / "final.csv"

# =============================
# MODEL FILES
# =============================
MODEL_PATH = MODEL_DIR / "model_rf.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# =============================
# TRAINING CONFIG
# =============================
TEST_SIZE = 0.20
RANDOM_STATE = 42