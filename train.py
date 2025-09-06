#!/usr/bin/env python3
"""
Main Training Script for Medical Text Classification

This script trains a trainable sentence transformer on medical abstracts
with backpropagation to improve medical term extraction.
"""

import argparse
import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from training import MedicalTextTrainer, create_training_config, create_model_config


def main():
    parser = argparse.ArgumentParser(description='Train Medical Text Classifier')
    parser.add_argument('--train-data', type=str, default='data/medical_tc_train.csv',
                       help='Path to training data')
    parser.add_argument('--test-data', type=str, default='data/medical_tc_test.csv',
                       help='Path to test data')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--gcp', action='store_true',
                       help='Run on Google Cloud Platform')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f'medical_classifier_{timestamp}'

    print("Medical Text Classification Training")
    print("=" * 50)
    print(f"Experiment: {args.experiment_name}")
    print(f"Train data: {args.train_data}")
    print(f"Test data: {args.test_data}")
    print(f"Output dir: {args.output_dir}")

    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        model_config = config['model']
        training_config = config['training']
    else:
        # Create default configuration
        model_config = create_model_config()
        training_config = create_training_config()

    print("\nModel Configuration:")
    for k, v in model_config.items():
        print(f"  {k}: {v}")

    print("\nTraining Configuration:")
    for k, v in training_config.items():
        print(f"  {k}: {v}")

    # Initialize trainer
    trainer = MedicalTextTrainer(model_config, training_config)

    # Load and preprocess data
    print("\nLoading data...")
    train_df, test_df = trainer.load_data(args.train_data, args.test_data)

    # Split test data for validation (since we don't have a separate validation set)
    val_size = int(len(test_df) * 0.5)
    val_df = test_df[:val_size]
    test_df = test_df[val_size:]

    print(f"Using {len(val_df)} samples for validation")

    # Preprocess data
    train_texts, train_labels, val_texts, val_labels = trainer.preprocess_data(train_df, val_df)

    # Train the model
    print("\nStarting training...")
    history = trainer.train(train_texts, train_labels, val_texts, val_labels)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_texts, test_labels, _, _ = trainer.preprocess_data(test_df, test_df)
    test_metrics = trainer.evaluate(test_texts, test_labels)

    print("\nTest Results:")
    print(f"  F1 (Macro): {test_metrics['f1_macro']:.4f}")
    print(f"  F1 (Weighted): {test_metrics['f1_weighted']:.4f}")
    print(f"  F1 (Micro): {test_metrics['f1_micro']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")

    # Save results
    results = {
        'experiment_name': args.experiment_name,
        'model_config': model_config,
        'training_config': training_config,
        'training_history': history,
        'test_metrics': test_metrics,
        'timestamp': datetime.now().isoformat()
    }

    results_file = os.path.join(args.output_dir, f'{args.experiment_name}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Save final model
    model_file = os.path.join(args.output_dir, f'{args.experiment_name}_final_model.npz')
    trainer.save_checkpoint(f'../{model_file}')

    print(f"\nResults saved to: {results_file}")
    print(f"Model saved to: {model_file}")

    # GCP deployment info
    if args.gcp:
        print("\nGCP Deployment Information:")
        print("-" * 30)
        print("To deploy on GCP, use the following commands:")
        print("1. Build Docker image:")
        print("   docker build -t medical-text-classifier .")
        print("2. Push to Google Container Registry:")
        print("   docker push gcr.io/YOUR_PROJECT/medical-text-classifier")
        print("3. Deploy to Vertex AI:")
        print("   gcloud ai custom-jobs create --region=us-central1 --display-name=medical-training --worker-pool-spec=machine-type=n1-standard-8,replica-count=1,container-image-uri=gcr.io/YOUR_PROJECT/medical-text-classifier")


if __name__ == '__main__':
    main()
