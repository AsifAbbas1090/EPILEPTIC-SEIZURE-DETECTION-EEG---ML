# Sequence-Based Deep Learning Classification for EEG Data

## Project Overview

This is a **Machine Learning semester project** that implements sequence-based deep learning models for **Epileptic Seizure Recognition** using EEG (Electroencephalogram) data. The project compares three recurrent neural network architectures: **RNN**, **LSTM**, and **GRU** to classify EEG signals into 5 different classes.

**Project Type:** Deliverable 5.4 - Sequence-Based Deep Learning Classification

---

## Dataset

- **Source:** Epileptic Seizure Recognition Dataset
- **Type:** Time-series EEG data
- **Structure:**
  - **Features:** 178 time steps (X1 to X178) representing sequential EEG measurements
  - **Samples:** Multiple EEG recordings
  - **Classes:** 5 different seizure/non-seizure classifications
- **Preprocessing:** Data is already preprocessed and stored in `preprocessed_eeg_data.csv`

---

## Project Structure

```
├── main.py                      # Main entry point for running the project
├── models.py                    # RNN, LSTM, and GRU model definitions
├── config.py                    # Configuration and hyperparameters
├── data_loader.py              # Data loading and preprocessing
├── trainer.py                  # Model training with callbacks
├── evaluator.py                # Model evaluation and metrics
├── requirements.txt            # Python dependencies
├── saved_models/               # Trained model weights (.h5 files)
│   ├── RNN_best.h5
│   ├── LSTM_best.h5
│   └── GRU_best.h5
├── results/                    # Evaluation results
│   └── results.txt
└── README.md                   # This file
```

---

## Models Implemented

### 1. **RNN (Recurrent Neural Network)**
- Simple RNN layer with 64 units
- Architecture: RNN(64) → Dropout(0.3) → Dense(32) → Dense(5-softmax)
- Baseline model for sequence processing

### 2. **LSTM (Long Short-Term Memory)**
- LSTM layer with 64 units
- Architecture: LSTM(64) → Dropout(0.3) → Dense(32) → Dense(5-softmax)
- Captures long-term dependencies in sequences

### 3. **GRU (Gated Recurrent Unit)**
- GRU layer with 64 units
- Architecture: GRU(64) → Dropout(0.3) → Dense(32) → Dense(5-softmax)
- Balanced between RNN simplicity and LSTM complexity

---

## Configuration & Hyperparameters

Key settings defined in `config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Sequence Length** | 178 | Time steps per sample |
| **Features per Step** | 1 | Single EEG channel |
| **Train/Test Split** | 80/20 | Data split ratio |
| **Validation Split** | 20% | Validation from training set |
| **Batch Size** | 32 | Training batch size |
| **Epochs** | 30 | Maximum training epochs |
| **Learning Rate** | 0.001 | Adam optimizer LR |
| **Dropout Rate** | 0.3 | Regularization dropout |
| **Dense Units** | 32 | Hidden layer units |
| **Number of Classes** | 5 | Output classes |

---

## Key Features

### Data Processing Pipeline
1. **Loading:** Reads preprocessed EEG data from CSV
2. **Reshaping:** Converts flat data to 3D sequences (samples × time_steps × features)
3. **Normalization:** StandardScaler applied to all features
4. **Encoding:** One-hot encoding for 5 output classes
5. **Splitting:** Stratified train/test split to maintain class distribution

### Training Features
- **Early Stopping:** Prevents overfitting by monitoring validation loss
- **Model Checkpointing:** Saves best model weights during training
- **Learning Rate Reduction:** Adaptive learning rate on validation loss plateau
- **Validation Split:** 20% of training data used for validation

### Evaluation Metrics
- **Accuracy:** Overall correctness
- **Precision:** Positive prediction accuracy
- **Recall:** True positive detection rate
- **F1-Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** Per-class performance breakdown
- **Classification Report:** Detailed per-class metrics

---

## Installation & Setup

### 1. Prerequisites
- Python 3.8+
- pip package manager

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
```
tensorflow>=2.10.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
```

---

## Usage

### Running the Complete Pipeline

Execute the main script to train and evaluate all three models:

```bash
python main.py
```

**This will:**
1. Load and prepare the EEG dataset
2. Build RNN, LSTM, and GRU models
3. Train each model with validation
4. Evaluate each model on the test set
5. Save trained models to `saved_models/`
6. Save results to `results/results.txt`

### Output During Execution
- Dataset summary (number of samples, classes, features)
- Model architecture for each model type
- Training progress (loss, accuracy, validation metrics)
- Training callbacks information (early stopping, checkpoints)
- Evaluation metrics and confusion matrices
- Final results file with performance summary

---

## File Descriptions

### `main.py`
- **Purpose:** Orchestrates the entire workflow
- **Functions:**
  - `run_model()`: Trains and evaluates a single model type
  - `main()`: Loads data, trains all models, saves results

### `models.py`
- **Purpose:** Defines neural network architectures
- **Functions:**
  - `build_rnn_model()`: Creates SimpleRNN architecture
  - `build_lstm_model()`: Creates LSTM architecture
  - `build_gru_model()`: Creates GRU architecture
  - `get_model()`: Factory function for model selection
  - `print_model_summary()`: Displays model architecture

### `data_loader.py`
- **Purpose:** Handles data loading and preprocessing
- **Functions:**
  - `load_data()`: Reads CSV data file
  - `prepare_sequential_data()`: Reshapes to 3D sequences
  - `normalize_data()`: Standardizes feature values
  - `encode_labels()`: One-hot encodes class labels
  - `split_data()`: Performs stratified train/test split
  - `load_and_prepare_data()`: Full pipeline wrapper
  - `get_data_summary()`: Returns dataset statistics

### `trainer.py`
- **Purpose:** Model training with callbacks
- **Functions:**
  - `get_callbacks()`: Creates training callbacks (early stopping, checkpointing, LR reduction)
  - `train_model()`: Trains model with validation data

### `evaluator.py`
- **Purpose:** Model evaluation and metrics calculation
- **Functions:**
  - `evaluate_model()`: Calculates all performance metrics
  - `save_results()`: Writes results to output file

### `config.py`
- **Purpose:** Centralized configuration management
- **Contains:** Paths, hyperparameters, and model architecture settings

---

## Results & Outputs

### Model Weights
Trained models are saved as HDF5 files:
- `saved_models/RNN_best.h5`
- `saved_models/LSTM_best.h5`
- `saved_models/GRU_best.h5`

### Evaluation Results
All results are saved to `results/results.txt` containing:
- Accuracy, Precision, Recall, F1-Score for each model
- Comparison across all three architectures

---

## Model Comparison & Analysis

The project trains three sequence models to compare their effectiveness:

| Model | Best for | Characteristics |
|-------|----------|-----------------|
| **RNN** | Baseline sequences | Simple, fast, basic memory |
| **LSTM** | Long-term dependencies | Better memory, handles vanishing gradient |
| **GRU** | Balanced performance | Simpler than LSTM, good accuracy |

All three models use the same:
- Input shape: (178 time steps, 1 feature)
- Output: 5-class softmax classification
- Training settings: 32 batch size, 30 epochs max
- Regularization: 30% dropout

---

## Dependencies & Versions

- **TensorFlow/Keras:** ≥2.10.0 (Deep learning framework)
- **scikit-learn:** ≥1.0.0 (Metrics and data splitting)
- **pandas:** ≥1.3.0 (Data manipulation)
- **NumPy:** ≥1.21.0 (Numerical computing)

---

## Training & Optimization

### Regularization Techniques
- **Dropout:** 30% dropout after RNN/LSTM/GRU layers
- **Early Stopping:** Stops if validation loss doesn't improve for 5 epochs
- **Learning Rate Reduction:** Reduces LR by 50% if val loss plateaus for 3 epochs

### Data Handling
- **Normalization:** StandardScaler fit on training data
- **Stratified Split:** Maintains class distribution in train/test splits
- **Validation Data:** Separate 20% from training for monitoring

---

## Expected Performance

The models should achieve good classification on the EEG dataset:
- **Typical Accuracy Range:** 85-98% (depending on model and data quality)
- **Best Performing:** Usually LSTM or GRU due to sequential nature
- **Training Time:** 2-10 minutes per model (varies by hardware)

---

## Troubleshooting

### Missing Data File
If you get "Data file not found" error:
- Ensure `preprocessed_eeg_data.csv` exists in the project root
- Check file path in `config.py`

### Memory Issues
- Reduce `BATCH_SIZE` in `config.py`
- Reduce `SEQUENCE_LENGTH` if possible
- Run models individually instead of all together

### Training Takes Too Long
- Reduce `EPOCHS` in `config.py`
- Reduce model `UNITS` (RNN_UNITS, LSTM_UNITS, GRU_UNITS)
- Increase `BATCH_SIZE` for faster epochs

---

## Author & Course Information

- **Course:** 7th Semester Machine Learning
- **Project Type:** Semester Project - Deliverable 5.4
- **Objective:** Implement and compare sequence-based deep learning models for EEG classification

---

## Notes

- All models use categorical cross-entropy loss for multi-class classification
- Adam optimizer with default decay for efficient training
- GPU acceleration (if available) will significantly speed up training
- Models are saved only if they improve validation loss (best model saved)

---

## Future Improvements

Potential enhancements to the project:
1. Implement bidirectional LSTM/GRU for context from both directions
2. Add attention mechanisms for interpretability
3. Implement ensemble methods combining all three models
4. Add hyperparameter tuning (random search or grid search)
5. Implement cross-validation for more robust evaluation
6. Add visualization of training history and predictions
7. Implement real-time seizure detection pipeline
8. Add class weight balancing for imbalanced datasets

---

**Created:** December 2025  
**Status:** Complete for semester submission
