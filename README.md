# Sentiment Analysis: LSTM vs DistilBERT

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

A comprehensive comparative study of **LSTM** and **DistilBERT** models for binary sentiment classification on IMDB movie reviews with an interactive Streamlit web application.

## 🎯 Project Overview

This project implements and evaluates two deep learning approaches for sentiment analysis:
- **LSTM**: Traditional RNN architecture trained from scratch
- **DistilBERT**: Modern Transformer-based pre-trained model

The analysis reveals how transfer learning and pre-training significantly improve NLP model performance.

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| LSTM | 86.04% | 85.33% | 86.75% | 0.8603 | 0.9204 |
| **DistilBERT** | **90.82%** ⭐ | **90.33%** | **91.31%** | **0.9082** | **0.9620** |

**Key Finding**: DistilBERT achieves **+4.78% higher accuracy** through transfer learning and pre-training.

## ✨ Features

- 🏠 **Home** - Project overview with statistics
- 🎯 **Demo** - Real-time predictions with example reviews
- 📊 **Comparison** - Detailed performance metrics and visualizations
- 📈 **Analysis** - Dataset statistics and 8 interactive charts
- ℹ️ **About** - Model architectures and recommendations

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/sarratebni-wq/sentiment-analysis-lstm-distilbert.git
cd sentiment-analysis-lstm-distilbert

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Download Models

**Models are too large for GitHub (~278 MB).** Download from your Kaggle notebook:

1. `lstm_sentiment_model.h5`
2. `distilbert_sentiment_model/` (folder)
3. `tokenizer.pkl`
4. `preprocessing_metadata.pkl`
5. `evaluation_results.pkl`

Place files in project root folder.

### Run Application

```bash
streamlit run app.py
```

Open browser to: `http://localhost:8501`

## 📁 Project Structure
sentiment-analysis-lstm-distilbert/
├── app.py                              # Streamlit application
├── requirements.txt                    # Dependencies
├── README.md                           # This file
├── .gitignore                          # Git ignore rules
│
├── Models/                             # Trained models (download)
│   ├── lstm_sentiment_model.h5
│   ├── distilbert_sentiment_model/
│   ├── tokenizer.pkl
│   ├── preprocessing_metadata.pkl
│   └── evaluation_results.pkl
│
└── Visualizations/                     # Analysis charts
├── 01_sentiment_distribution.png
├── 01_review_length_distribution.png
├── 01_sentiment_by_length.png
├── 01_top_words.png
├── 03_lstm_training_curves.png
├── 04_confusion_matrices.png
├── 04_metrics_comparison.png
└── 04_roc_curves.png

## 🧠 Model Architectures

### LSTM Model
Input → Embedding (128) → Bi-LSTM (128) → Dropout (0.5) →
Bi-LSTM (64) → Dropout (0.5) → Dense (64, ReLU) →
Dropout (0.3) → Sigmoid Output

**Config**: Adam optimizer, lr=0.001, batch=32, epochs=5  
**Parameters**: ~2.5M  
**Training Time**: 14 minutes

### DistilBERT Model
Text → Tokenizer → Embeddings → 6 Transformer Layers →
Classification Head → Softmax Output

**Config**: Adam optimizer, lr=5e-5, batch=16, epochs=3  
**Parameters**: ~67M (66M pre-trained)  
**Training Time**: 21 minutes

## 📊 Dataset

- **Source**: IMDB Movie Reviews (Kaggle)
- **Size**: 50,000 reviews
- **Classes**: Binary (Positive/Negative) - 50/50 split
- **Splits**: 70% train, 15% val, 15% test
- **Avg Length**: 230 words

## 🔄 How to Use

### Test with Examples

1. Go to **🎯 Demo** page
2. Click example buttons (Positive/Negative/Neutral)
3. Click **"🚀 Analyze Review"**
4. See real-time predictions from both models

### Enter Custom Review

1. Type your own review
2. Click **"🚀 Analyze Review"**
3. Get instant predictions with confidence scores

### Compare Models

1. Go to **📊 Comparison** page
2. View detailed performance metrics
3. See model agreement analysis

## 💡 Key Findings

1. **Pre-trained models outperform from-scratch training** (+4.78% accuracy)
2. **Transfer learning provides significant advantages** in NLP
3. **Accuracy vs Speed trade-off** exists between models
4. **Both models show excellent generalization** (minimal overfitting)
5. **Model ensemble could further improve** robustness

## 🎯 Recommendations

### Use DistilBERT When:
- ✅ Maximum accuracy needed
- ✅ Production systems with adequate resources
- ✅ Handling complex/nuanced sentiment
- ✅ GPU/TPU available for inference

### Use LSTM When:
- ✅ Speed/latency is critical
- ✅ Edge device or mobile deployment
- ✅ Limited computational resources
- ✅ Real-time processing needed

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | TensorFlow, Keras, PyTorch |
| NLP | Hugging Face Transformers |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Web Framework | Streamlit |

## 📋 Requirements
streamlit==1.28.1
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
tensorflow==2.13.0
torch==2.0.1
torchvision==0.15.2
transformers==4.31.0
scikit-learn==1.3.0
Pillow==10.0.0

## 🚀 Deployment

### Local
```bash
streamlit run app.py
```
## 📈 Performance Comparison

| Aspect | LSTM | DistilBERT |
|--------|------|-----------|
| Accuracy | 86.04% | 90.82% ⭐ |
| Inference Speed | 2ms | 8ms |
| Model Size | 10 MB | 268 MB |
| Parameters | 2.5M | 67M |
| Pre-trained | ❌ | ✅ |
| Best For | Speed | Accuracy |

## 🔍 Error Analysis

### False Positives
- Mixed sentiments in reviews
- Sarcasm and backhanded compliments
- Count: LSTM (348), DistilBERT (211)

### False Negatives
- Subtle positive language
- Comparative positives ("better than expected")
- Count: LSTM (475), DistilBERT (336)

## 🌟 Example Predictions

**Positive Review**: "This movie was absolutely amazing!"
- LSTM: 96.5% confidence
- DistilBERT: 97.1% confidence ✅

**Negative Review**: "Waste of time. Terrible movie."
- LSTM: 8.7% confidence (Negative)
- DistilBERT: 6.2% confidence (Negative) ✅

**Borderline Review**: "It was okay, average."
- LSTM: 52.3% (Positive)
- DistilBERT: 48.2% (Negative) - More accurate

## 🚀 Future Improvements

- [ ] Multilingual support
- [ ] Multi-class classification (positive/neutral/negative)
- [ ] Model ensemble approach
- [ ] REST API endpoint
- [ ] Batch prediction
- [ ] Explainability features (LIME, SHAP)
- [ ] Active learning implementation
- [ ] Mobile app version

## 📚 References

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Streamlit Docs](https://docs.streamlit.io/)

## 📄 License

MIT License - feel free to use this project for any purpose.

## 👤 Author

**Sarratebni-WQ**  
GitHub: [@sarratebni-wq](https://github.com/sarratebni-wq)

## 🤝 Contributing

Contributions welcome! Feel free to:
- Fork the repository
- Create a feature branch
- Submit pull requests
- Open issues for bugs/suggestions

## 💬 Support

- 📧 Open an issue on GitHub
- 💭 Use GitHub Discussions

---

**Made with ❤️ for the NLP community**  
**Last Updated**: June 2024 | **Status**: Complete ✅
