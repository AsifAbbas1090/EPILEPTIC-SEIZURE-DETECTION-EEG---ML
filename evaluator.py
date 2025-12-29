

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import config


def evaluate_model(model, X_test, y_test, label_encoder, model_name):
    
    print(f"\n{'='*70}")
    print(f"EVALUATING {model_name.upper()} MODEL")
    print(f"{'='*70}")
    
    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Print results
    print(f"\nResults:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    print(f"{'='*70}\n")
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_true': y_true,
        'y_pred': y_pred,
        'confusion_matrix': cm
    }


def save_results(results, filename='results.txt'):
    import os
    filepath = os.path.join(config.OUTPUT_DIR, filename)
    
    with open(filepath, 'w') as f:
        f.write("="*70 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("="*70 + "\n\n")
        
        for result in results:
            f.write(f"Model: {result['model_name'].upper()}\n")
            f.write(f"  Accuracy:  {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)\n")
            f.write(f"  Precision: {result['precision']:.4f} ({result['precision']*100:.2f}%)\n")
            f.write(f"  Recall:    {result['recall']:.4f} ({result['recall']*100:.2f}%)\n")
            f.write(f"  F1-Score:  {result['f1_score']:.4f} ({result['f1_score']*100:.2f}%)\n")
            f.write("\n" + "-"*70 + "\n\n")
    
    print(f"✓ Results saved to: {filepath}")

