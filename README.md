# Medical Text Classification with Trainable Sentence Transformer

A novel architecture for medical text classification that uses a trainable sentence transformer with backpropagation to learn medically relevant text extraction through classification feedback.  It is a work in progress.

## Overview

This project implements a trainable embeddings model that learns to extract medically relevant sections from text through backpropagation from classification loss. The system uses:

- **Trainable Sentence Transformer**: Custom transformer that learns medical term extraction
- **Attention Mechanism**: Focuses on medically relevant parts of text
- **Backpropagation**: Classification loss improves the sentence transformer's extraction
- **F1/RMSE Loss**: Custom loss functions optimized for medical classification
- **GCP Deployment**: Ready for Google Cloud Platform training

## Architecture

```
Medical Text Input
        ↓
Sentence Transformer (Trainable)
    ↓
Attention Mechanism (Medical Terms)
    ↓
Classification Head
    ↓
F1 Loss + Backpropagation
    ↓
Improved Medical Extraction
```

## Key Features

- ✅ **End-to-End Trainable**: Sentence transformer learns through classification feedback
- ✅ **Medical Focus**: Attention mechanism trained on medical terminology
- ✅ **Custom Loss Functions**: F1 and RMSE optimized for medical classification
- ✅ **Backpropagation**: Gradients flow from classification to sentence transformer
- ✅ **GCP Ready**: Configured for Google Cloud Vertex AI training
- ✅ **Comprehensive Testing**: Full test suite for training and inference

## Dataset

Uses the [Medical Abstracts TC Corpus](https://github.com/sebischair/Medical-Abstracts-TC-Corpus) with 5 classes:

- Neoplasms (3,163 samples)
- Digestive system diseases (1,494 samples)
- Nervous system diseases (1,925 samples)
- Cardiovascular diseases (3,051 samples)
- General pathological conditions (4,805 samples)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd medical-text-classification
   ```

2. **Setup virtual environment**
   ```bash
   ./setup.sh
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

```bash
# Train the model
python train.py --experiment-name my-experiment

# Train with custom configuration
python train.py --config config/custom_config.json
```

### Inference

```bash
# Single text prediction
python inference.py --model-path outputs/my-experiment_final_model.npz --text "Patient presents with chest pain..."

# Batch prediction from file
python inference.py --model-path outputs/my-experiment_final_model.npz --input-file test_texts.txt
```

### Testing

```bash
# Run comprehensive tests
python test_training.py
```

## Configuration

### Model Configuration

```json
{
  "vocab_size": 10000,
  "embedding_dim": 768,
  "max_seq_length": 512,
  "num_attention_heads": 12,
  "num_classes": 5,
  "learning_rate": 0.001
}
```

### Training Configuration

```json
{
  "num_epochs": 20,
  "batch_size": 32,
  "learning_rate": 0.001,
  "lambda_f1": 1.0,
  "lambda_rmse": 0.1,
  "patience": 5
}
```

## Google Cloud Platform Deployment

### Prerequisites

1. **Google Cloud Project** with billing enabled
2. **Vertex AI API** enabled
3. **Google Cloud SDK** installed and configured

### Deployment Steps

1. **Build Docker image**
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_PROJECT/medical-text-classifier .
   ```

2. **Run deployment script**
   ```bash
   chmod +x deploy_gcp.sh
   ./deploy_gcp.sh
   ```

3. **Monitor training**
   ```bash
   # Check training status
   gcloud ai custom-jobs list --region=us-central1

   # View logs
   gcloud ai custom-jobs describe JOB_ID --region=us-central1
   ```

### Cost Estimation

- **Compute**: ~$0.38/hour (n1-standard-8)
- **GPU**: ~$0.35/hour (NVIDIA T4)
- **Storage**: ~$0.026/GB/month
- **Estimated total for 2-hour training**: ~$1.50

## Architecture Details

### Trainable Sentence Transformer

- **Word Embeddings**: Learned embeddings for medical vocabulary
- **Attention Mechanism**: Multi-head attention for medical term extraction
- **Medical Term Weights**: Learned importance weights for medical terms
- **Backpropagation**: Gradients flow from classification loss to transformer

### Loss Functions

- **F1 Loss**: Optimized for imbalanced medical classification
- **RMSE Loss**: Regularizes attention distribution
- **Combined Loss**: Weighted combination of F1 and RMSE

### Training Loop

1. Forward pass through sentence transformer
2. Extract medical-relevant embeddings via attention
3. Classification prediction
4. Compute F1 loss
5. Backpropagate through transformer and classifier
6. Update all parameters

## Results

The system demonstrates:

- **Medical Term Extraction**: Learns to focus on clinically relevant text
- **Improved Classification**: Backpropagation enhances medical understanding
- **Interpretability**: Attention weights show important medical terms
- **Scalability**: Ready for large-scale medical text processing

## Files Structure

```
medical-text-classification/
├── src/
│   ├── trainable_sentence_transformer.py  # Core model
│   ├── loss_functions.py                  # Custom losses
│   └── training.py                        # Training loop
├── train.py                               # Training script
├── inference.py                           # Inference script
├── test_training.py                       # Test suite
├── gcp_config.py                          # GCP deployment
├── Dockerfile                             # Container config
├── requirements.txt                       # Dependencies
├── setup.sh                               # Environment setup
└── data/                                  # Dataset
    ├── medical_tc_train.csv
    ├── medical_tc_test.csv
    └── medical_tc_labels.csv
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For questions or issues, please open a GitHub issue or contact the maintainers.
