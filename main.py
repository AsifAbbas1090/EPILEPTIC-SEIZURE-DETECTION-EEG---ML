

import os
import sys
import numpy as np
from data_loader import load_and_prepare_data, get_data_summary
from models import get_model, print_model_summary
from trainer import train_model
from evaluator import evaluate_model, save_results
import config


def run_model(model_type, data, model_name=None):
    """
    Run a single sequence model
    
    Args:
        model_type: 'rnn', 'lstm', or 'gru'
        data: Dictionary with training/test data
        model_name: Optional custom name for the model
    """
    if model_name is None:
        model_name = model_type.upper()
    
    print(f"\n{'#'*70}")
    print(f"# {model_name} MODEL")
    print(f"{'#'*70}")
    
    # Get input shape
    input_shape = (data['X_train'].shape[1], data['X_train'].shape[2])
    num_classes = data['y_train'].shape[1]
    
    # Build model
    model = get_model(model_type, input_shape, num_classes)
    print_model_summary(model, model_name)
    
    # Split validation from training
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        data['X_train'], data['y_train'],
        test_size=config.VALIDATION_SPLIT,
        random_state=config.RANDOM_STATE,
        stratify=np.argmax(data['y_train'], axis=1)
    )
    
    # Train model
    history = train_model(model, X_train, y_train, X_val, y_val, model_name)
    
    # Evaluate model
    results = evaluate_model(
        model, data['X_test'], data['y_test'],
        data['label_encoder'], model_name
    )
    
    return results, history


def main():
    """Main execution function"""
    print("="*70)
    print("DELIVERABLE 5.4: SEQUENCE-BASED DEEP LEARNING CLASSIFICATION")
    print("="*70)
    
    # Change to project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # Get dataset summary
    summary = get_data_summary()
    print("\nDataset Summary:")
    print(f"  Name: {summary['dataset_name']}")
    print(f"  Source: {summary['source']}")
    print(f"  Type: {summary['data_type']}")
    print(f"  Samples: {summary['num_samples']:,}")
    print(f"  Sequence Length: {summary['sequence_length']}")
    print(f"  Features per Step: {summary['features_per_step']}")
    print(f"  Classes: {summary['num_classes']}")
    print(f"  Class Labels: {summary['class_labels']}")
    
    # Load and prepare data
    data = load_and_prepare_data()
    
    # Run all models
    all_results = []
    
    # Choose which model to run
    models_to_run = ['rnn', 'lstm', 'gru']  # Run all sequence models
    
    for model_type in models_to_run:
        try:
            results, history = run_model(model_type, data)
            all_results.append(results)
        except Exception as e:
            print(f"\n✗ Error running {model_type}: {str(e)}")
            import traceback
            traceback.print_exception(*sys.exc_info())
    
    # Save results
    if all_results:
        save_results(all_results)
        
        # Print summary table
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        print(f"{'Model':<15} {'Accuracy':<12} {'F1-Score':<12}")
        print("-"*70)
        for result in all_results:
            print(f"{result['model_name']:<15} {result['accuracy']*100:>10.2f}%  {result['f1_score']*100:>10.2f}%")
        print("="*70)
    
    print("\n✓ Deliverable 5.4 execution complete!")


if __name__ == "__main__":
    main()

