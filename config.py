

import os

# Dataset paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "preprocessed_eeg_data.csv")
RAW_DATA_PATH = os.path.join(BASE_DIR, "archive (1)", "Epileptic Seizure Recognition.csv")

# Data settings
SEQUENCE_LENGTH = 178  # Time steps (X1 to X178)
FEATURES_PER_STEP = 1  # Single feature per time step
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42

# Model settings
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

# Model architecture
RNN_UNITS = 64
LSTM_UNITS = 64
GRU_UNITS = 64
DROPOUT_RATE = 0.3
DENSE_UNITS = 32

# Output settings
NUM_CLASSES = 5  # 5 classes in the dataset
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

