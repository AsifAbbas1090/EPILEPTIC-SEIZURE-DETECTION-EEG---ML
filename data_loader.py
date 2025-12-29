
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
import config


def load_data():
    print("Loading preprocessed EEG data...")
    
    if not os.path.exists(config.DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {config.DATA_PATH}")
    
    df = pd.read_csv(config.DATA_PATH, index_col=0)
    print(f"✓ Loaded {len(df):,} samples")
    
    return df


def prepare_sequential_data(df):
   
    print("\nPreparing sequential data...")
    
    # Extract feature columns (X1 to X178)
    feature_cols = [col for col in df.columns if col.startswith('X')]
    feature_cols.sort(key=lambda x: int(x[1:]))  # Sort X1, X2, ..., X178
    
    # Extract features and labels
    X = df[feature_cols].values
    y = df['class_label'].values
    
    print(f"  - Original shape: {X.shape}")
    print(f"  - Features: {len(feature_cols)} time steps")
    
    # Reshape to 3D: (samples, time_steps, features_per_step)
    # Each sample has 178 time steps, each with 1 feature
    X_reshaped = X.reshape(X.shape[0], config.SEQUENCE_LENGTH, config.FEATURES_PER_STEP)
    
    print(f"  - Reshaped to 3D: {X_reshaped.shape} (samples × time_steps × features)")
    
    return X_reshaped, y


def normalize_data(X_train, X_test):
    print("\nNormalizing data...")
    
    # Reshape for scaling (flatten time dimension)
    n_samples_train, n_timesteps, n_features = X_train.shape
    n_samples_test = X_test.shape[0]
    
    X_train_flat = X_train.reshape(n_samples_train, -1)
    X_test_flat = X_test.reshape(n_samples_test, -1)
    
    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    # Reshape back to 3D
    X_train_scaled = X_train_scaled.reshape(n_samples_train, n_timesteps, n_features)
    X_test_scaled = X_test_scaled.reshape(n_samples_test, n_timesteps, n_features)
    
    print("  ✓ Data normalized")
    
    return X_train_scaled, X_test_scaled, scaler


def encode_labels(y):
    print("\nEncoding labels...")
    
    # Label encoding (0 to 4 for 5 classes)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    y_categorical = to_categorical(y_encoded, num_classes=config.NUM_CLASSES)
    
    print(f"  - Original labels: {np.unique(y)}")
    print(f"  - Encoded labels: {np.unique(y_encoded)}")
    print(f"  - One-hot shape: {y_categorical.shape}")
    print("  ✓ Labels encoded")
    
    return y_encoded, y_categorical, label_encoder


def split_data(X, y):
    print("\nSplitting data...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=1 - config.TRAIN_TEST_SPLIT,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    
    print(f"  - Training set: {X_train.shape[0]:,} samples ({config.TRAIN_TEST_SPLIT*100:.0f}%)")
    print(f"  - Testing set: {X_test.shape[0]:,} samples ({(1-config.TRAIN_TEST_SPLIT)*100:.0f}%)")
    
    return X_train, X_test, y_train, y_test


def get_data_summary():
    df = load_data()
    
    summary = {
        'dataset_name': 'Epileptic Seizure Recognition Data Set',
        'source': 'UCI Machine Learning Repository / Kaggle',
        'data_type': 'Sequential / Time-series EEG data',
        'num_samples': len(df),
        'sequence_length': config.SEQUENCE_LENGTH,
        'features_per_step': config.FEATURES_PER_STEP,
        'num_classes': config.NUM_CLASSES,
        'class_labels': sorted(df['class_label'].unique().tolist())
    }
    
    return summary


def load_and_prepare_data():
    print("=" * 70)
    print("DATA PREPARATION PIPELINE")
    print("=" * 70)
    
    # Load data
    df = load_data()
    
    # Prepare sequential data
    X, y = prepare_sequential_data(df)
    
    # Encode labels
    y_encoded, y_categorical, label_encoder = encode_labels(y)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y_categorical)
    
    # Normalize data
    X_train, X_test, scaler = normalize_data(X_train, X_test)
    
    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)
    print(f"Training shape: {X_train.shape}")
    print(f"Testing shape: {X_test.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Testing labels shape: {y_test.shape}")
    print("=" * 70 + "\n")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'label_encoder': label_encoder,
        'scaler': scaler
    }


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    data = load_and_prepare_data()
    print("Data preparation successful!")

