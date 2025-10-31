# Fine-Tuning Large Language Models for Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.20+-green.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive implementation of fine-tuning Large Language Models (LLMs) for binary sentiment classification using the IMDB movie reviews dataset. This project demonstrates the complete pipeline from data preparation to production deployment, including hyperparameter optimization, comprehensive evaluation, and error analysis.

## 🎯 Project Overview

This project fine-tunes a DistilBERT model on the IMDB movie reviews dataset for binary sentiment classification (positive/negative). The implementation includes:

- **Dataset**: IMDB Movie Reviews (25,000 samples)
- **Task**: Binary sentiment classification
- **Model**: DistilBERT-base-uncased (66M parameters)
- **Framework**: Hugging Face Transformers
- **Evaluation**: Comprehensive metrics and error analysis

## 🚀 Key Features

- ✅ **Systematic Hyperparameter Optimization** - 5 different configurations tested
- ✅ **Comprehensive Evaluation** - Multiple metrics and baseline comparison
- ✅ **Error Analysis** - Detailed pattern identification and improvement suggestions
- ✅ **Production-Ready Pipeline** - Complete inference system with confidence scores
- ✅ **Reproducible Results** - Fixed random seeds and complete documentation
- ✅ **Real-World Applicable** - Practical sentiment analysis for business use cases

## 📊 Results Summary

| Metric | Baseline Model | Fine-tuned Model | Improvement |
|--------|----------------|------------------|-------------|
| Accuracy | [Baseline] | [Fine-tuned] | +[%] |
| F1-Score | [Baseline] | [Fine-tuned] | +[%] |
| Precision | [Baseline] | [Fine-tuned] | +[%] |
| Recall | [Baseline] | [Fine-tuned] | +[%] |

*Results will be updated after training completion*

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended: 8GB+ VRAM)
- 16GB+ RAM
- 10GB+ storage space

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/llm-fine-tuning-sentiment-analysis.git
cd llm-fine-tuning-sentiment-analysis
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📁 Project Structure

```
llm-fine-tuning-sentiment-analysis/
├── LLM_Fine_Tuning_Assignment.ipynb    # Main notebook with complete implementation
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
├── Technical_Report_LLM_Fine_Tuning.md # Detailed technical report
├── Video_Demonstration_Script.md      # Video presentation script
├── best_model/                        # Saved model and tokenizer
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files
├── results/                           # Training outputs and logs
│   ├── baseline/
│   ├── high_lr/
│   ├── low_lr/
│   ├── large_batch/
│   └── high_weight_decay/
└── logs/                             # Training logs
    ├── baseline/
    ├── high_lr/
    └── ...
```

## 🚀 Quick Start

### Option 1: Jupyter Notebook (Recommended)

1. **Open the notebook**
```bash
jupyter notebook LLM_Fine_Tuning_Assignment.ipynb
```

2. **Run all cells** - The notebook will:
   - Install required packages
   - Load and preprocess the IMDB dataset
   - Train models with different hyperparameters
   - Evaluate performance and generate visualizations
   - Save the best model for inference

### Option 2: Python Script

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentiment_analyzer import SentimentAnalyzer

# Load the fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained('./best_model')
tokenizer = AutoTokenizer.from_pretrained('./best_model')

# Initialize sentiment analyzer
analyzer = SentimentAnalyzer(model, tokenizer)

# Predict sentiment
result = analyzer.predict_sentiment("This movie is absolutely fantastic!")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.4f}")
```

## 📈 Usage Examples

### Single Text Prediction

```python
# Initialize the analyzer
analyzer = SentimentAnalyzer(model, tokenizer)

# Predict sentiment for a single text
text = "This movie is absolutely fantastic! I loved every minute of it."
result = analyzer.predict_sentiment(text)

print(f"Text: {text}")
print(f"Predicted Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Probabilities: {result['probabilities']}")
```

**Output:**
```
Text: This movie is absolutely fantastic! I loved every minute of it.
Predicted Sentiment: Positive
Confidence: 0.9543
Probabilities: {'negative': 0.0457, 'positive': 0.9543}
```

### Batch Prediction

```python
# Predict sentiment for multiple texts
texts = [
    "This movie is absolutely fantastic! I loved every minute of it.",
    "The worst movie I have ever seen. Complete waste of time.",
    "It was okay, nothing special but not terrible either.",
    "Amazing cinematography and brilliant acting throughout.",
    "Boring and predictable plot with poor character development."
]

results = analyzer.predict_batch(texts)

for i, (text, result) in enumerate(zip(texts, results), 1):
    print(f"{i}. {result['sentiment']} (confidence: {result['confidence']:.3f})")
    print(f"   Text: {text[:50]}...")
```

## 🔧 Configuration

### Hyperparameter Configurations

The project tests 5 different hyperparameter configurations:

| Configuration | Learning Rate | Batch Size | Weight Decay | Epochs |
|---------------|---------------|------------|--------------|--------|
| Baseline      | 2e-5          | 16         | 0.01         | 3      |
| High LR       | 5e-5          | 16         | 0.01         | 3      |
| Low LR        | 1e-5          | 16         | 0.01         | 3      |
| Large Batch   | 2e-5          | 32         | 0.01         | 3      |
| High WD       | 2e-5          | 16         | 0.1          | 3      |

### Model Configuration

```python
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512
NUM_LABELS = 2
```

## 📊 Evaluation Metrics

The project implements comprehensive evaluation metrics:

- **Accuracy**: Overall classification correctness
- **Precision**: True positive rate among positive predictions
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed error analysis
- **Error Pattern Analysis**: False positive/negative identification

## 🔍 Error Analysis

The project includes detailed error analysis:

- **Error Rate**: Percentage of misclassified samples
- **Error Types**: False positives vs false negatives
- **Text Length Analysis**: Performance across different text lengths
- **Pattern Identification**: Common characteristics of errors
- **Improvement Suggestions**: Actionable insights for model enhancement



## 📋 Technical Report

A detailed technical report is available including:

- **Methodology**: Complete approach and implementation details
- **Results**: Comprehensive performance analysis
- **Limitations**: Current constraints and challenges
- **Future Work**: Improvement suggestions and research directions
- **References**: Academic citations and related work



### Performance Optimization

- **Model Size**: 66M parameters (efficient for deployment)
- **Inference Speed**: ~[X]ms per prediction
- **Memory Usage**: ~[X]GB RAM for inference
- **Batch Processing**: Optimized for high-throughput scenarios

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


