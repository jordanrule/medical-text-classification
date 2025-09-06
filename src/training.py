"""
Training Loop for Medical Text Classification with Backpropagation

Implements the complete training pipeline with backpropagation through
the trainable sentence transformer.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import os
import json
from datetime import datetime

try:
    from .trainable_sentence_transformer import MedicalTextClassifier
    from .loss_functions import CombinedLoss, compute_metrics
except ImportError:
    # For testing without proper package structure
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from trainable_sentence_transformer import MedicalTextClassifier
    from loss_functions import CombinedLoss, compute_metrics


class MedicalTextTrainer:
    """
    Trainer for medical text classification with trainable sentence transformer.
    """

    def __init__(self,
                 model_config: Dict,
                 training_config: Dict):
        """
        Initialize the trainer.

        Args:
            model_config: Configuration for the model
            training_config: Configuration for training
        """
        self.model_config = model_config
        self.training_config = training_config

        # Initialize model
        self.model = MedicalTextClassifier(**model_config)

        # Initialize loss function
        self.loss_fn = CombinedLoss(
            num_classes=model_config['num_classes'],
            lambda_f1=training_config.get('lambda_f1', 1.0),
            lambda_rmse=training_config.get('lambda_rmse', 0.1)
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_f1': [],
            'val_f1': [],
            'train_accuracy': [],
            'val_accuracy': []
        }

    def load_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test data.
        """
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        print(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples")
        print("Training label distribution:")
        print(train_df['condition_label'].value_counts())

        return train_df, test_df

    def preprocess_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Preprocess the data for training.
        """
        # Extract texts and labels
        train_texts = train_df['medical_abstract'].tolist()
        train_labels = train_df['condition_label'].tolist()

        test_texts = test_df['medical_abstract'].tolist()
        test_labels = test_df['condition_label'].tolist()

        # Fit tokenizer and label encoder on training data
        self.model.fit_tokenizer(train_texts)
        self.model.label_encoder.fit(train_labels)

        return train_texts, train_labels, test_texts, test_labels

    def train_epoch(self, texts: List[str], labels: List[str], batch_size: int) -> Dict[str, float]:
        """
        Train for one epoch.
        """
        epoch_loss = 0
        epoch_metrics = {'f1': 0, 'accuracy': 0}
        num_batches = 0

        # Shuffle data
        indices = np.random.permutation(len(texts))
        texts_shuffled = [texts[i] for i in indices]
        labels_shuffled = [labels[i] for i in indices]

        for i in range(0, len(texts), batch_size):
            batch_texts = texts_shuffled[i:i+batch_size]
            batch_labels = labels_shuffled[i:i+batch_size]

            # Forward pass
            embeddings, logits = self.model.model.forward(batch_texts)

            # Convert labels to indices
            label_indices = [self.model.label_encoder.transform([label])[0] for label in batch_labels]
            label_indices = np.array(label_indices)

            # Compute loss
            loss, loss_grad = self.loss_fn(logits, label_indices)

            # Backward pass
            gradients = self.model.model.backward(embeddings, logits, label_indices, loss_grad)

            # Update parameters
            self.model.model.update_parameters(gradients)

            # Accumulate metrics
            epoch_loss += loss
            predictions = np.argmax(logits, axis=1)
            batch_accuracy = np.mean(predictions == label_indices)
            epoch_metrics['accuracy'] += batch_accuracy

            num_batches += 1

        # Average metrics
        epoch_loss /= num_batches
        epoch_metrics['accuracy'] /= num_batches

        # Compute F1 on full epoch data (simplified)
        all_embeddings, all_logits = self.model.model.forward(texts[:1000])  # Subset for speed
        all_predictions = np.argmax(all_logits, axis=1)
        all_targets = self.model.label_encoder.transform(labels[:1000])
        metrics = compute_metrics(all_logits, all_targets)
        epoch_metrics['f1'] = metrics['f1_weighted']

        return {
            'loss': epoch_loss,
            'f1': epoch_metrics['f1'],
            'accuracy': epoch_metrics['accuracy']
        }

    def validate(self, texts: List[str], labels: List[str]) -> Dict[str, float]:
        """
        Validate the model.
        """
        embeddings, logits = self.model.model.forward(texts)
        predictions = np.argmax(logits, axis=1)
        targets = self.model.label_encoder.transform(labels)

        loss, _ = self.loss_fn(logits, targets)
        metrics = compute_metrics(logits, targets)

        return {
            'loss': loss,
            'f1': metrics['f1_weighted'],
            'accuracy': metrics['accuracy']
        }

    def train(self, train_texts: List[str], train_labels: List[str],
              val_texts: List[str], val_labels: List[str]) -> Dict[str, List[float]]:
        """
        Train the model.
        """
        num_epochs = self.training_config.get('num_epochs', 10)
        batch_size = self.training_config.get('batch_size', 32)
        patience = self.training_config.get('patience', 5)

        best_val_loss = float('inf')
        patience_counter = 0

        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch(train_texts, train_labels, batch_size)
            print(".4f"
                  ".4f")

            # Validate
            val_metrics = self.validate(val_texts, val_labels)
            print(".4f"
                  ".4f")

            # Store history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])

            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                # Save best model
                self.save_checkpoint(f"best_model_epoch_{epoch+1}.npz")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        return self.history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        os.makedirs('checkpoints', exist_ok=True)
        self.model.model.save_model(f"checkpoints/{filename}")

        # Save training state
        state = {
            'epoch': len(self.history['train_loss']),
            'model_config': self.model_config,
            'training_config': self.training_config,
            'history': self.history,
            'label_encoder_classes': self.model.label_encoder.classes_.tolist()
        }

        with open(f"checkpoints/{filename.replace('.npz', '_state.json')}", 'w') as f:
            json.dump(state, f, indent=2)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        self.model.model.load_model(f"checkpoints/{filename}")

        state_file = filename.replace('.npz', '_state.json')
        if os.path.exists(f"checkpoints/{state_file}"):
            with open(f"checkpoints/{state_file}", 'r') as f:
                state = json.load(f)
            self.history = state['history']

    def predict(self, texts: List[str]) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(texts)

    def evaluate(self, texts: List[str], labels: List[str]) -> Dict[str, float]:
        """Evaluate the model."""
        predictions = self.predict(texts)
        targets = np.array(labels)

        # Convert predictions to encoded format for metrics
        pred_encoded = self.model.label_encoder.transform(predictions)

        metrics = compute_metrics(pred_encoded.reshape(-1, 1), targets)

        return {
            'f1_macro': metrics['f1_macro'],
            'f1_weighted': metrics['f1_weighted'],
            'f1_micro': metrics['f1_micro'],
            'accuracy': metrics['accuracy'],
            'confusion_matrix': metrics['confusion_matrix'].tolist()
        }


def create_training_config() -> Dict:
    """
    Create default training configuration.
    """
    return {
        'num_epochs': 20,
        'batch_size': 32,
        'learning_rate': 0.001,
        'lambda_f1': 1.0,
        'lambda_rmse': 0.1,
        'patience': 5,
        'validation_split': 0.2
    }


def create_model_config(num_classes: int = 5) -> Dict:
    """
    Create default model configuration.
    """
    return {
        'vocab_size': 10000,
        'embedding_dim': 768,
        'max_seq_length': 512,
        'num_attention_heads': 12,
        'num_classes': num_classes,
        'learning_rate': 0.001
    }
