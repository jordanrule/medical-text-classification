"""
Custom Loss Functions for Medical Text Classification

Implements F1 loss and RMSE with backpropagation support.
"""

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from typing import Tuple, List, Optional, Dict


class F1Loss:
    """
    F1 Loss implementation that can be backpropagated through the sentence transformer.
    """

    def __init__(self, num_classes: int, average: str = 'weighted'):
        """
        Initialize F1 Loss.

        Args:
            num_classes: Number of classification classes
            average: Averaging method ('weighted', 'macro', 'micro')
        """
        self.num_classes = num_classes
        self.average = average

    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute F1 loss and gradients.

        Args:
            predictions: Model predictions (logits or probabilities)
            targets: True labels

        Returns:
            loss: F1 loss value
            gradients: Gradients w.r.t. predictions
        """
        # Convert logits to predictions if needed
        if predictions.shape[1] == self.num_classes:
            pred_classes = np.argmax(predictions, axis=1)
        else:
            pred_classes = predictions.astype(int)

        # Compute F1 score
        f1 = f1_score(targets, pred_classes, average=self.average)

        # F1 loss is 1 - F1 score
        loss = 1.0 - f1

        # Compute gradients (simplified for demonstration)
        # In a real implementation, this would compute exact gradients
        gradients = self._compute_f1_gradients(predictions, targets, pred_classes)

        return loss, gradients

    def _compute_f1_gradients(self, predictions: np.ndarray, targets: np.ndarray,
                             pred_classes: np.ndarray) -> np.ndarray:
        """
        Compute gradients for F1 loss w.r.t. predictions.
        """
        # Simplified gradient computation
        # Real F1 gradients would be more complex
        gradients = np.zeros_like(predictions)

        for i in range(len(targets)):
            if pred_classes[i] != targets[i]:
                # Penalize wrong predictions
                gradients[i, pred_classes[i]] += 1.0
                gradients[i, targets[i]] -= 1.0

        # Normalize gradients
        gradients /= len(targets)

        return gradients


class RMSELoss:
    """
    Root Mean Square Error Loss for regression tasks.
    """

    def __init__(self):
        pass

    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute RMSE loss and gradients.

        Args:
            predictions: Model predictions
            targets: True values

        Returns:
            loss: RMSE loss value
            gradients: Gradients w.r.t. predictions
        """
        errors = predictions - targets
        loss = np.sqrt(np.mean(errors ** 2))

        # Gradients w.r.t. predictions
        gradients = errors / (len(predictions) * loss + 1e-8)

        return loss, gradients


class CombinedLoss:
    """
    Combined loss that includes both classification and extraction components.
    """

    def __init__(self, num_classes: int, lambda_f1: float = 1.0, lambda_rmse: float = 0.1):
        """
        Initialize combined loss.

        Args:
            num_classes: Number of classification classes
            lambda_f1: Weight for F1 loss
            lambda_rmse: Weight for RMSE loss (for attention regularization)
        """
        self.f1_loss = F1Loss(num_classes)
        self.rmse_loss = RMSELoss()
        self.lambda_f1 = lambda_f1
        self.lambda_rmse = lambda_rmse

    def __call__(self, predictions: np.ndarray, targets: np.ndarray,
                  attention_weights: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
        """
        Compute combined loss.

        Args:
            predictions: Classification predictions
            targets: True labels
            attention_weights: Optional attention weights for regularization

        Returns:
            total_loss: Combined loss value
            gradients: Combined gradients
        """
        # F1 loss for classification
        f1_loss, f1_grad = self.f1_loss(predictions, targets)

        total_loss = self.lambda_f1 * f1_loss
        total_grad = self.lambda_f1 * f1_grad

        # RMSE loss for attention regularization (if provided)
        if attention_weights is not None:
            # Target smooth attention distribution
            target_attention = np.ones_like(attention_weights) / attention_weights.shape[1]
            rmse_loss, rmse_grad = self.rmse_loss(attention_weights, target_attention)

            total_loss += self.lambda_rmse * rmse_loss
            # For simplicity, we'll add RMSE gradients to classification gradients
            # In practice, this would need proper gradient routing
            total_grad += self.lambda_rmse * rmse_grad.mean(axis=1, keepdims=True)

        return total_loss, total_grad


class MedicalF1Loss:
    """
    Medical-specific F1 loss that accounts for class imbalance in medical datasets.
    """

    def __init__(self, num_classes: int, class_weights: Optional[np.ndarray] = None):
        """
        Initialize medical F1 loss.

        Args:
            num_classes: Number of classes
            class_weights: Optional class weights for imbalanced data
        """
        self.num_classes = num_classes
        self.class_weights = class_weights

    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute medical F1 loss with class weighting.
        """
        pred_classes = np.argmax(predictions, axis=1)

        # Compute per-class F1 scores
        f1_scores = []
        for class_idx in range(self.num_classes):
            # True positives, false positives, false negatives for this class
            tp = np.sum((pred_classes == class_idx) & (targets == class_idx))
            fp = np.sum((pred_classes == class_idx) & (targets != class_idx))
            fn = np.sum((pred_classes != class_idx) & (targets == class_idx))

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            f1_scores.append(f1)

        # Apply class weights if provided
        if self.class_weights is not None:
            weighted_f1 = np.average(f1_scores, weights=self.class_weights)
        else:
            weighted_f1 = np.mean(f1_scores)

        loss = 1.0 - weighted_f1

        # Compute gradients
        gradients = np.zeros_like(predictions)
        for i in range(len(targets)):
            if pred_classes[i] != targets[i]:
                # Weight the gradient by class importance
                weight = self.class_weights[targets[i]] if self.class_weights is not None else 1.0
                gradients[i, pred_classes[i]] += weight
                gradients[i, targets[i]] -= weight

        gradients /= len(targets)

        return loss, gradients


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive metrics for medical text classification.
    """
    pred_classes = np.argmax(predictions, axis=1) if predictions.ndim > 1 else predictions

    # F1 scores
    f1_macro = f1_score(targets, pred_classes, average='macro')
    f1_weighted = f1_score(targets, pred_classes, average='weighted')
    f1_micro = f1_score(targets, pred_classes, average='micro')

    # Per-class F1
    f1_per_class = f1_score(targets, pred_classes, average=None)

    # Confusion matrix
    cm = confusion_matrix(targets, pred_classes)

    return {
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_micro': f1_micro,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'accuracy': np.mean(pred_classes == targets)
    }
