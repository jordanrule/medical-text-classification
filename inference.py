#!/usr/bin/env python3
"""
Inference Script for Medical Text Classification

Load a trained model and make predictions on medical abstracts.
"""

import argparse
import os
import sys
import json
import numpy as np
from typing import List, Dict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from training import MedicalTextTrainer, create_model_config
except ImportError:
    # For testing without proper package structure
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    from training import MedicalTextTrainer, create_model_config


def load_trained_model(model_path: str, config_path: str = None) -> MedicalTextTrainer:
    """
    Load a trained model from checkpoint.

    Args:
        model_path: Path to saved model file
        config_path: Path to configuration file (optional)

    Returns:
        Loaded trainer with trained model
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        model_config = config['model']
        training_config = config['training']
    else:
        # Use default configuration
        model_config = create_model_config()
        training_config = {'num_epochs': 10, 'batch_size': 32}

    # Initialize trainer
    trainer = MedicalTextTrainer(model_config, training_config)

    # Load model weights
    trainer.load_checkpoint(model_path)

    return trainer


def predict_medical_condition(text: str, trainer: MedicalTextTrainer) -> Dict:
    """
    Predict medical condition from text.

    Args:
        text: Medical abstract text
        trainer: Trained model trainer

    Returns:
        Prediction results
    """
    prediction = trainer.predict([text])[0]

    # Get model confidence (simplified)
    embeddings, logits = trainer.model.model.forward([text])
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    confidence = np.max(probabilities[0])

    return {
        'prediction': prediction,
        'confidence': float(confidence),
        'text_length': len(text),
        'extracted_features': embeddings.shape[1]  # Number of features extracted
    }


def batch_predict(texts: List[str], trainer: MedicalTextTrainer,
                 batch_size: int = 32) -> List[Dict]:
    """
    Make batch predictions.

    Args:
        texts: List of medical texts
        trainer: Trained model trainer
        batch_size: Batch size for inference

    Returns:
        List of prediction results
    """
    results = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        predictions = trainer.predict(batch_texts)

        # Get embeddings for feature analysis
        embeddings, logits = trainer.model.model.forward(batch_texts)
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        confidences = np.max(probabilities, axis=1)

        for j, (text, pred, conf) in enumerate(zip(batch_texts, predictions, confidences)):
            results.append({
                'text_index': i + j,
                'prediction': pred,
                'confidence': float(conf),
                'text_length': len(text),
            })

    return results


def analyze_medical_extraction(text: str, trainer: MedicalTextTrainer) -> Dict:
    """
    Analyze how the model extracts medical information from text.

    Args:
        text: Medical text to analyze
        trainer: Trained model

    Returns:
        Analysis of extraction process
    """
    # Get token-level analysis (simplified)
    tokens = trainer.model.model._tokenize_text(text)

    if not tokens:
        return {'error': 'Could not tokenize text'}

    # Get attention weights for medical terms
    embeddings = trainer.model.model._extract_medical_terms(tokens)

    # Find important tokens (simplified)
    medical_weights = trainer.model.model.medical_term_weights[tokens]
    important_indices = np.argsort(medical_weights)[-5:]  # Top 5 medical terms

    important_terms = []
    for idx in important_indices:
        if idx < len(tokens):
            # Convert token back to word (simplified)
            term = f"token_{tokens[idx]}"
            weight = medical_weights[idx]
            important_terms.append({'term': term, 'weight': float(weight)})

    return {
        'total_tokens': len(tokens),
        'embedding_dimension': len(embeddings),
        'important_medical_terms': important_terms,
        'extraction_confidence': float(np.linalg.norm(embeddings))
    }


def main():
    parser = argparse.ArgumentParser(description='Medical Text Classification Inference')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config-path', type=str, default=None,
                       help='Path to model configuration file')
    parser.add_argument('--text', type=str, default=None,
                       help='Single text to classify')
    parser.add_argument('--input-file', type=str, default=None,
                       help='File with texts to classify (one per line)')
    parser.add_argument('--output-file', type=str, default=None,
                       help='Output file for predictions')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze medical extraction process')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for inference')

    args = parser.parse_args()

    # Load trained model
    print(f"Loading model from: {args.model_path}")
    trainer = load_trained_model(args.model_path, args.config_path)

    if args.text:
        # Single text prediction
        print(f"\nClassifying text: {args.text[:100]}...")

        result = predict_medical_condition(args.text, trainer)
        print("\nPrediction Results:")
        print(f"  Condition: {result['prediction']}")
        print(".4f")
        print(f"  Text Length: {result['text_length']}")
        print(f"  Features Extracted: {result['extracted_features']}")

        if args.analyze:
            print("\nMedical Extraction Analysis:")
            analysis = analyze_medical_extraction(args.text, trainer)
            print(f"  Total Tokens: {analysis['total_tokens']}")
            print(f"  Embedding Dimension: {analysis['embedding_dimension']}")
            print(f"  Extraction Confidence: {analysis['extraction_confidence']:.4f}")
            print("  Important Medical Terms:")
            for term in analysis['important_medical_terms']:
                print(".4f")

    elif args.input_file:
        # Batch prediction from file
        print(f"Loading texts from: {args.input_file}")

        with open(args.input_file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]

        print(f"Processing {len(texts)} texts...")

        results = batch_predict(texts, trainer, args.batch_size)

        if args.output_file:
            # Save results to file
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output_file}")
        else:
            # Print summary
            predictions = [r['prediction'] for r in results]
            confidences = [r['confidence'] for r in results]

            print("
Batch Prediction Summary:")
            print(f"  Total Predictions: {len(results)}")
            print(f"  Unique Conditions Predicted: {len(set(predictions))}")
            print(".4f")
            print(".4f")

    else:
        print("Please provide either --text or --input-file")
        sys.exit(1)


if __name__ == '__main__':
    main()
