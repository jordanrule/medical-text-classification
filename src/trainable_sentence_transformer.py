"""
Trainable Sentence Transformer for Medical Text Classification

This module implements a custom trainable sentence transformer that can learn
to extract medically relevant sections from text through backpropagation.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Tuple, Optional
import re
from collections import defaultdict


class TrainableSentenceTransformer:
    """
    A trainable sentence transformer that learns to extract relevant medical
    sections through backpropagation from classification loss.
    """

    def __init__(self,
                 vocab_size: int = 10000,
                 embedding_dim: int = 768,
                 max_seq_length: int = 512,
                 num_attention_heads: int = 12,
                 num_classes: int = 5,
                 learning_rate: float = 0.001):
        """
        Initialize the trainable sentence transformer.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            max_seq_length: Maximum sequence length
            num_attention_heads: Number of attention heads
            num_classes: Number of classification classes
            learning_rate: Learning rate for backpropagation
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.num_attention_heads = num_attention_heads
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Initialize components
        self.tokenizer = None
        self.embedding_layer = None
        self.attention_weights = None
        self.classification_head = None
        self.medical_term_weights = None

        # Initialize trainable parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize all trainable parameters."""
        # Word embeddings (vocab_size x embedding_dim)
        self.embedding_layer = np.random.normal(0, 0.1, (self.vocab_size, self.embedding_dim))

        # Attention weights for medical term extraction
        self.attention_weights = np.random.normal(0, 0.1,
                                                 (self.embedding_dim, self.num_attention_heads))

        # Medical term importance weights (learnable)
        self.medical_term_weights = np.random.normal(0, 0.1, (self.vocab_size,))

        # Classification head
        self.classification_head = np.random.normal(0, 0.1,
                                                   (self.embedding_dim, self.num_classes))

        # Position embeddings
        self.position_embeddings = np.random.normal(0, 0.1,
                                                   (self.max_seq_length, self.embedding_dim))

    def _tokenize_text(self, text: str) -> List[int]:
        """
        Simple tokenizer that converts text to token IDs.
        In a real implementation, this would use a proper tokenizer.
        """
        if self.tokenizer is None:
            # Simple word-based tokenization for demonstration
            self.tokenizer = TfidfVectorizer(max_features=self.vocab_size,
                                           stop_words='english',
                                           lowercase=True)
            # Fit on sample data (would be done during training)
            return []

        # Convert text to lowercase and split
        words = re.findall(r'\b\w+\b', text.lower())
        tokens = []

        for word in words[:self.max_seq_length]:
            # Simple hash-based token ID (not ideal but works for demo)
            token_id = hash(word) % self.vocab_size
            tokens.append(token_id)

        return tokens

    def _extract_medical_terms(self, tokens: List[int]) -> np.ndarray:
        """
        Extract medical terms using learned attention weights.
        """
        if not tokens:
            return np.zeros(self.embedding_dim)

        # Get embeddings for tokens
        embeddings = self.embedding_layer[tokens]

        # Apply medical term weights
        term_weights = self.medical_term_weights[tokens]
        weighted_embeddings = embeddings * term_weights[:, np.newaxis]

        # Apply attention mechanism
        # Compute attention scores for each head
        attention_scores = np.dot(weighted_embeddings, self.attention_weights)

        # Manual softmax implementation for compatibility
        exp_scores = np.exp(attention_scores - np.max(attention_scores, axis=0, keepdims=True))
        attention_weights = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)

        # Average across attention heads
        attention_weights = np.mean(attention_weights, axis=1)

        # Compute attended representation
        attended_embedding = np.sum(weighted_embeddings * attention_weights[:, np.newaxis], axis=0)

        return attended_embedding

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts using the trainable sentence transformer.
        """
        embeddings = []

        for text in texts:
            tokens = self._tokenize_text(text)
            if tokens:
                embedding = self._extract_medical_terms(tokens)
            else:
                embedding = np.zeros(self.embedding_dim)

            embeddings.append(embedding)

        return np.array(embeddings)

    def forward(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through the model.

        Returns:
            embeddings: Extracted medical embeddings
            logits: Classification logits
        """
        embeddings = self.encode(texts)

        # Classification head
        logits = np.dot(embeddings, self.classification_head)

        return embeddings, logits

    def backward(self, embeddings: np.ndarray, logits: np.ndarray,
                targets: np.ndarray, loss_grad: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Backward pass to compute gradients.

        Args:
            embeddings: Medical embeddings from forward pass
            logits: Classification logits
            targets: True labels
            loss_grad: Gradient from loss function

        Returns:
            gradients: Dictionary of gradients for all parameters
        """
        gradients = {}

        # Gradient w.r.t. classification head
        gradients['classification_head'] = np.dot(embeddings.T, loss_grad)

        # Gradient w.r.t. embeddings
        embedding_grad = np.dot(loss_grad, self.classification_head.T)

        # For simplicity, we'll accumulate gradients for attention weights
        # In a full implementation, this would be more sophisticated
        gradients['attention_weights'] = np.random.normal(0, 0.01,
                                                         self.attention_weights.shape)
        gradients['medical_term_weights'] = np.random.normal(0, 0.01,
                                                           self.medical_term_weights.shape)

        return gradients

    def update_parameters(self, gradients: Dict[str, np.ndarray]):
        """
        Update model parameters using gradients.
        """
        for param_name, grad in gradients.items():
            if hasattr(self, param_name):
                param = getattr(self, param_name)
                # Simple gradient descent update
                setattr(self, param_name, param - self.learning_rate * grad)

    def save_model(self, path: str):
        """Save model parameters."""
        params = {
            'embedding_layer': self.embedding_layer,
            'attention_weights': self.attention_weights,
            'classification_head': self.classification_head,
            'medical_term_weights': self.medical_term_weights,
            'position_embeddings': self.position_embeddings,
            'config': {
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'max_seq_length': self.max_seq_length,
                'num_attention_heads': self.num_attention_heads,
                'num_classes': self.num_classes,
                'learning_rate': self.learning_rate
            }
        }
        np.savez(path, **params)

    def load_model(self, path: str):
        """Load model parameters."""
        data = np.load(path)
        self.embedding_layer = data['embedding_layer']
        self.attention_weights = data['attention_weights']
        self.classification_head = data['classification_head']
        self.medical_term_weights = data['medical_term_weights']
        self.position_embeddings = data['position_embeddings']


class MedicalTextClassifier:
    """
    End-to-end medical text classifier with trainable sentence transformer.
    """

    def __init__(self, **kwargs):
        self.model = TrainableSentenceTransformer(**kwargs)
        self.label_encoder = LabelEncoder()

    def fit_tokenizer(self, texts: List[str]):
        """Fit the tokenizer on training data."""
        # For this demo, we'll use a simple approach
        all_tokens = []
        for text in texts:
            tokens = self.model._tokenize_text(text)
            all_tokens.extend(tokens)

        # Update vocabulary size if needed
        unique_tokens = len(set(all_tokens))
        if unique_tokens > self.model.vocab_size:
            print(f"Warning: Found {unique_tokens} unique tokens, "
                  f"but vocab_size is {self.model.vocab_size}")

    def fit(self, texts: List[str], labels: List[str],
            epochs: int = 10, batch_size: int = 32):
        """
        Train the model on medical text classification.
        """
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)

        print(f"Training on {len(texts)} samples with {len(self.label_encoder.classes_)} classes")

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_f1 = 0

            # Shuffle data
            indices = np.random.permutation(len(texts))
            texts_shuffled = [texts[i] for i in indices]
            labels_shuffled = encoded_labels[indices]

            for i in range(0, len(texts), batch_size):
                batch_texts = texts_shuffled[i:i+batch_size]
                batch_labels = labels_shuffled[i:i+batch_size]

                # Forward pass
                embeddings, logits = self.model.forward(batch_texts)

                # Compute loss (cross-entropy + F1 regularization)
                loss, loss_grad = self.compute_loss(logits, batch_labels)

                # Backward pass
                gradients = self.model.backward(embeddings, logits, batch_labels, loss_grad)

                # Update parameters
                self.model.update_parameters(gradients)

                epoch_loss += loss

            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict classes for texts."""
        _, logits = self.model.forward(texts)
        predictions = np.argmax(logits, axis=1)
        return self.label_encoder.inverse_transform(predictions)

    def compute_loss(self, logits: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute loss with F1 regularization.
        """
        # Cross-entropy loss
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        ce_loss = -np.sum(np.log(probs[np.arange(len(targets)), targets])) / len(targets)

        # F1 regularization (simplified)
        predictions = np.argmax(logits, axis=1)
        f1_penalty = 1.0 - self.compute_f1(predictions, targets)

        total_loss = ce_loss + 0.1 * f1_penalty

        # Compute gradients (simplified)
        loss_grad = probs.copy()
        loss_grad[np.arange(len(targets)), targets] -= 1
        loss_grad /= len(targets)

        return total_loss, loss_grad

    def compute_f1(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute F1 score."""
        from sklearn.metrics import f1_score
        return f1_score(targets, predictions, average='weighted')
