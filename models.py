
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    SimpleRNN, LSTM, GRU,
    Dense, Dropout, Input
)
from tensorflow.keras.optimizers import Adam
import config


def build_rnn_model(input_shape, num_classes):
    
    model = Sequential([
        Input(shape=input_shape),
        SimpleRNN(config.RNN_UNITS, return_sequences=False),
        Dropout(config.DROPOUT_RATE),
        Dense(config.DENSE_UNITS, activation='relu'),
        Dropout(config.DROPOUT_RATE),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_lstm_model(input_shape, num_classes):
   
    model = Sequential([
        Input(shape=input_shape),
        LSTM(config.LSTM_UNITS, return_sequences=False),
        Dropout(config.DROPOUT_RATE),
        Dense(config.DENSE_UNITS, activation='relu'),
        Dropout(config.DROPOUT_RATE),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_gru_model(input_shape, num_classes):
   
    model = Sequential([
        Input(shape=input_shape),
        GRU(config.GRU_UNITS, return_sequences=False),
        Dropout(config.DROPOUT_RATE),
        Dense(config.DENSE_UNITS, activation='relu'),
        Dropout(config.DROPOUT_RATE),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_model(model_type, input_shape, num_classes):
   
    model_type = model_type.lower()
    
    if model_type == 'rnn':
        return build_rnn_model(input_shape, num_classes)
    elif model_type == 'lstm':
        return build_lstm_model(input_shape, num_classes)
    elif model_type == 'gru':
        return build_gru_model(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'rnn', 'lstm', or 'gru'")


def print_model_summary(model, model_name):
    """Print model architecture summary"""
    print(f"\n{'='*70}")
    print(f"{model_name.upper()} MODEL ARCHITECTURE")
    print(f"{'='*70}")
    model.summary()
    print(f"{'='*70}\n")

