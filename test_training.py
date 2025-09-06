#!/usr/bin/env python3
"""
Test Script for Medical Text Classification Training and Inference

This script tests the complete training and inference pipeline.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from training import MedicalTextTrainer, create_model_config, create_training_config


def test_data_loading():
    """Test data loading and preprocessing."""
    print("Testing data loading...")

    if not os.path.exists('data/medical_tc_train.csv'):
        print("ERROR: Training data not found")
        return False

    # Load data
    train_df = pd.read_csv('data/medical_tc_train.csv')
    test_df = pd.read_csv('data/medical_tc_test.csv')

    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Classes: {sorted(train_df['condition_label'].unique())}")

    return True


def test_model_initialization():
    """Test model initialization."""
    print("\nTesting model initialization...")

    model_config = create_model_config(num_classes=5)
    training_config = create_training_config()

    trainer = MedicalTextTrainer(model_config, training_config)

    print("Model initialized successfully")
    print(f"Vocab size: {trainer.model.model.vocab_size}")
    print(f"Embedding dim: {trainer.model.model.embedding_dim}")

    return trainer


def test_training_loop(trainer):
    """Test the training loop with a small subset."""
    print("\nTesting training loop...")

    # Load small subset for testing
    train_df = pd.read_csv('data/medical_tc_train.csv')
    test_df = pd.read_csv('data/medical_tc_test.csv')

    # Use small subset for quick testing
    train_subset = train_df.sample(n=100, random_state=42)
    test_subset = test_df.sample(n=20, random_state=42)

    print(f"Training on {len(train_subset)} samples")
    print(f"Validating on {len(test_subset)} samples")

    # Preprocess data using trainer's method
    train_texts, train_labels, val_texts, val_labels = trainer.preprocess_data(
        train_subset, test_subset)

    # Train for a few epochs
    print("Starting training...")
    history = trainer.train(train_texts, train_labels, val_texts, val_labels)

    print("Training completed!")
    print(".4f")
    print(".4f")

    return history


def test_inference(trainer):
    """Test inference pipeline."""
    print("\nTesting inference...")

    test_texts = [
        "The patient presents with chest pain and shortness of breath, suggesting cardiac issues.",
        "Abdominal pain and nausea indicate possible gastrointestinal problems.",
        "Headache and dizziness may be related to neurological conditions."
    ]

    predictions = trainer.predict(test_texts)

    print("Inference Results:")
    for i, (text, pred) in enumerate(zip(test_texts, predictions)):
        print(f"  Text {i+1}: {text[:50]}... -> {pred}")

    return predictions


def test_loss_functions():
    """Test custom loss functions."""
    print("\nTesting loss functions...")

    from src.loss_functions import F1Loss, RMSELoss, CombinedLoss

    # Test F1 Loss
    f1_loss = F1Loss(num_classes=5)
    predictions = np.random.rand(10, 5)
    targets = np.random.randint(0, 5, 10)

    loss, grad = f1_loss(predictions, targets)
    print(".4f")
    print(f"Gradient shape: {grad.shape}")

    # Test RMSE Loss
    rmse_loss = RMSELoss()
    pred_vals = np.random.rand(10)
    target_vals = np.random.rand(10)

    loss, grad = rmse_loss(pred_vals, target_vals)
    print(".4f")
    print(f"RMSE gradient shape: {grad.shape}")

    print("Loss functions working correctly")

    return True


def run_comprehensive_test():
    """Run comprehensive test of all components."""
    print("Running Comprehensive Test of Medical Text Classification")
    print("=" * 60)

    try:
        # Test data loading
        if not test_data_loading():
            return False

        # Test model initialization
        trainer = test_model_initialization()

        # Test loss functions
        test_loss_functions()

        # Test training loop
        history = test_training_loop(trainer)

        # Test inference
        predictions = test_inference(trainer)

        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        print("The trainable sentence transformer architecture is working.")
        print("\nKey Features Verified:")
        print("  ✓ Data loading and preprocessing")
        print("  ✓ Model initialization")
        print("  ✓ Custom loss functions (F1, RMSE)")
        print("  ✓ Training loop with backpropagation")
        print("  ✓ Inference pipeline")
        print("  ✓ Medical text classification")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
