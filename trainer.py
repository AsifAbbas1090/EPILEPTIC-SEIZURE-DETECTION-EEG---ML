

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import config


def get_callbacks(model_name):
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(config.MODELS_DIR, f'{model_name}_best.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks


def train_model(model, X_train, y_train, X_val, y_val, model_name):
    
    print(f"\n{'='*70}")
    print(f"TRAINING {model_name.upper()} MODEL")
    print(f"{'='*70}")
    print(f"Training samples: {X_train.shape[0]:,}")
    print(f"Validation samples: {X_val.shape[0]:,}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"{'='*70}\n")
    
    callbacks = get_callbacks(model_name)
    
    history = model.fit(
        X_train, y_train,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(X_val, y_val),
        validation_split=0.0,  
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n✓ Training completed for {model_name}")
    
    return history

