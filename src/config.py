import os

# Base paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Specific data directories
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
OPENFOAM_DATA_DIR = os.path.join(DATA_DIR, "openfoam")
DNS_DATA_DIR = os.path.join(DATA_DIR, "dns")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Data files
BF_DATASET_PATH = os.path.join(PROCESSED_DATA_DIR, "BF_full_set.csv")