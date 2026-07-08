import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Sentiment Analysis Demo",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS (Cached for speed)
# ============================================================================

@st.cache_resource
def load_lstm_model():
    """Load LSTM model and tokenizer"""
    try:
        lstm_model = load_model('lstm_sentiment_model.h5')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        with open('preprocessing_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        return lstm_model, tokenizer, metadata, True
    except Exception as e:
        st.error(f"Error loading LSTM: {e}")
        return None, None, None, False

@st.cache_resource
def load_bert_model():
    """Load DistilBERT model and tokenizer"""
    try:
        bert_tokenizer = AutoTokenizer.from_pretrained('./distilbert_sentiment_model')
        bert_model = AutoModelForSequenceClassification.from_pretrained(
            './distilbert_sentiment_model'
        )
        bert_model.eval()
        return bert_model, bert_tokenizer, True
    except Exception as e:
        st.error(f"Error loading DistilBERT: {e}")
        return None, None, False

@st.cache_resource
def load_evaluation_results():
    """Load evaluation results"""
    try:
        with open('evaluation_results.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading results: {e}")
        return None

# Load all models
lstm_model, lstm_tokenizer, metadata, lstm_loaded = load_lstm_model()
bert_model, bert_tokenizer, bert_loaded = load_bert_model()
eval_results = load_evaluation_results()

models_ready = lstm_loaded and bert_loaded and eval_results is not None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def predict_lstm(text):
    """Get LSTM prediction"""
    try:
        sequence = lstm_tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=metadata['max_length'], padding='post')
        prediction = lstm_model.predict(padded, verbose=0)[0][0]
        return float(prediction)
    except Exception as e:
        st.error(f"LSTM Prediction Error: {e}")
        return None

def predict_bert(text):
    """Get DistilBERT prediction"""
    try:
        encodings = bert_tokenizer(
            text,
            padding='max_length',
            max_length=512,
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = bert_model(**encodings)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            prediction = probs[0][1].item()
        
        return float(prediction)
    except Exception as e:
        st.error(f"DistilBERT Prediction Error: {e}")
        return None

def get_sentiment(probability):
    """Convert probability to sentiment label"""
    if probability >= 0.5:
        return "POSITIVE", "😊"
    else:
        return "NEGATIVE", "😞"

def get_confidence(probability):
    """Get confidence level"""
    confidence = max(probability, 1 - probability)
    return confidence

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

with st.sidebar:
    st.title("🎬 Sentiment Analysis")
    st.markdown("---")
    
    page = st.radio(
        "Navigate to:",
        ["🏠 Home", "🎯 Demo", "📊 Comparison", "📈 Analysis", "ℹ️ About"],
        help="Select a page to navigate"
    )
    
    st.markdown("---")
    st.markdown("### Project Info")
    st.info("""
    **Task:** Binary Sentiment Classification
    
    **Dataset:** IMDB Movie Reviews (50,000)
    
    **Models:**
    - LSTM (RNN-based)
    - DistilBERT (Transformer)
    """)

# ============================================================================
# PAGE 1: HOME
# ============================================================================

if page == "🏠 Home":
    st.markdown("# 🎬 Movie Review Sentiment Analysis")
    st.markdown("### Compare LSTM vs DistilBERT Models")
    
    st.write("""
    Welcome to the Sentiment Analysis Demo! This interactive application compares two 
    deep learning models for analyzing movie review sentiments.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📚 Dataset Size",
            value="50,000",
            delta="Reviews"
        )
    
    with col2:
        st.metric(
            label="🤖 Models Trained",
            value="2",
            delta="LSTM & DistilBERT"
        )
    
    with col3:
        st.metric(
            label="🎯 Best Accuracy",
            value="~91%",
            delta="DistilBERT"
        )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔍 What is Sentiment Analysis?")
        st.write("""
        Sentiment analysis is a Natural Language Processing (NLP) task that determines 
        whether a piece of text expresses positive, negative, or neutral sentiment.
        
        **Use Cases:**
        - Review analysis
        - Social media monitoring
        - Customer feedback analysis
        - Brand sentiment tracking
        """)
    
    with col2:
        st.markdown("### 🚀 Getting Started")
        st.write("""
        1. Go to **🎯 Demo** to test your own reviews
        2. Check **📊 Comparison** for detailed metrics
        3. View **📈 Analysis** for dataset insights
        4. Read **ℹ️ About** for project details
        """)
    
    st.markdown("---")
    st.markdown("### 💡 Why Two Different Models?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **LSTM (Long Short-Term Memory)**
        - Traditional RNN architecture
        - Trained from scratch
        - Faster inference
        - Smaller model size (~10 MB)
        - ~86% accuracy
        """)
    
    with col2:
        st.markdown("""
        **DistilBERT (Transformer)**
        - Modern pre-trained model
        - Transfer learning approach
        - Better contextual understanding
        - Larger model size (~268 MB)
        - ~91% accuracy ⭐
        """)

# ============================================================================
# PAGE 2: DEMO
# ============================================================================

elif page == "🎯 Demo":
    st.markdown("# 🎯 Sentiment Prediction Demo")
    st.markdown("Test the models with your own movie reviews!")
    
    if not models_ready:
        st.error("❌ Models not fully loaded. Please check the sidebar for errors.")
    else:
        # Input section
        st.markdown("### 📝 Enter a Movie Review")
        
        user_review = st.text_area(
            "Write your movie review here:",
            height=150,
            placeholder="E.g., This movie was absolutely fantastic! The acting was superb...",
            label_visibility="collapsed"
        )
        
        # Example reviews
        st.markdown("### 📚 Or try an example:")
        example_col1, example_col2, example_col3 = st.columns(3)
        
        examples = {
            "😊 Positive": "This movie was absolutely amazing! The cinematography was breathtaking, the acting superb, and the storyline kept me engaged from start to finish. Highly recommended!",
            "😞 Negative": "Waste of time. Boring plot, terrible acting, and waste of special effects. Couldn't even finish watching it. One of the worst movies I've seen.",
            "😐 Neutral": "It was okay. Some parts were good, but overall it was average. Not worth watching again but not terrible either."
        }
        
        with example_col1:
            if st.button("😊 Positive Example", use_container_width=True):
                user_review = examples["😊 Positive"]
                st.rerun()
        
        with example_col2:
            if st.button("😞 Negative Example", use_container_width=True):
                user_review = examples["😞 Negative"]
                st.rerun()
        
        with example_col3:
            if st.button("😐 Neutral Example", use_container_width=True):
                user_review = examples["😐 Neutral"]
                st.rerun()
        
        # Predict button
        if st.button("🚀 Analyze Review", use_container_width=True, type="primary"):
            if not user_review.strip():
                st.warning("⚠️ Please enter a review text!")
            else:
                with st.spinner("🔄 Analyzing with both models..."):
                    # LSTM prediction
                    lstm_score = predict_lstm(user_review)
                    if lstm_score is not None:
                        lstm_sentiment, lstm_emoji = get_sentiment(lstm_score)
                        lstm_confidence = get_confidence(lstm_score)
                    
                    # DistilBERT prediction
                    bert_score = predict_bert(user_review)
                    if bert_score is not None:
                        bert_sentiment, bert_emoji = get_sentiment(bert_score)
                        bert_confidence = get_confidence(bert_score)
                
                if lstm_score is not None and bert_score is not None:
                    st.markdown("---")
                    st.markdown("### 📊 Analysis Results")
                    
                    # Display review
                    with st.expander("📖 Your Review", expanded=True):
                        st.write(user_review)
                    
                    # Results columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### LSTM")
                        st.metric(
                            "Sentiment",
                            f"{lstm_sentiment} {lstm_emoji}",
                            f"Confidence: {lstm_confidence:.1%}"
                        )
                        st.progress(lstm_score)
                        st.caption(f"Score: {lstm_score:.4f}")
                    
                    with col2:
                        st.markdown("### DistilBERT")
                        st.metric(
                            "Sentiment",
                            f"{bert_sentiment} {bert_emoji}",
                            f"Confidence: {bert_confidence:.1%}"
                        )
                        st.progress(bert_score)
                        st.caption(f"Score: {bert_score:.4f}")
                    
                    # Comparison
                    st.markdown("---")
                    st.markdown("### ⚖️ Model Comparison")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    agreement = "✅ AGREE" if (lstm_score >= 0.5) == (bert_score >= 0.5) else "❌ DISAGREE"
                    difference = abs(lstm_score - bert_score)
                    
                    with col1:
                        st.metric("Agreement", agreement)
                    
                    with col2:
                        st.metric("Difference", f"{difference:.4f}")
                    
                    with col3:
                        st.metric("Average", f"{(lstm_score + bert_score)/2:.4f}")
                    
                    # Insights
                    st.markdown("---")
                    st.markdown("### 📌 Insights")
                    
                    if difference < 0.1:
                        st.success(f"✓ Both models strongly agree on **{lstm_sentiment.lower()}** sentiment")
                    elif difference < 0.3:
                        st.info(f"⚠ Models show moderate agreement with {difference:.1%} difference")
                    else:
                        st.warning(f"❌ Models disagree significantly ({difference:.1%} difference)")
                    
                    # Visualization
                    st.markdown("---")
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    models = ['LSTM', 'DistilBERT']
                    scores = [lstm_score, bert_score]
                    colors = ['steelblue', 'coral']
                    
                    bars = ax.bar(models, scores, color=colors, edgecolor='black', alpha=0.8, width=0.6)
                    ax.set_ylabel('Positive Sentiment Score', fontsize=12)
                    ax.set_ylim([0, 1])
                    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Decision Boundary')
                    ax.legend()
                    ax.grid(axis='y', alpha=0.3)
                    
                    # Add value labels
                    for bar, score in zip(bars, scores):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{score:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
                    
                    st.pyplot(fig, use_container_width=True)

# ============================================================================
# PAGE 3: COMPARISON
# ============================================================================

elif page == "📊 Comparison":
    st.markdown("# 📊 Model Performance Comparison")
    
    if not models_ready:
        st.error("❌ Models not fully loaded.")
    else:
        # Extract metrics
        lstm_metrics = eval_results['lstm_metrics']
        bert_metrics = eval_results['bert_metrics']
        
        # Metrics comparison
        st.markdown("### 📈 Test Set Performance Metrics")
        
        comparison_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
            'LSTM': [
                lstm_metrics['accuracy'],
                lstm_metrics['precision'],
                lstm_metrics['recall'],
                lstm_metrics['f1'],
                lstm_metrics['auc']
            ],
            'DistilBERT': [
                bert_metrics['accuracy'],
                bert_metrics['precision'],
                bert_metrics['recall'],
                bert_metrics['f1'],
                bert_metrics['auc']
            ]
        }
        
        # Display as table
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison['Difference'] = df_comparison['DistilBERT'] - df_comparison['LSTM']
        df_comparison['Winner'] = df_comparison.apply(
            lambda row: '🥇 DistilBERT' if row['DistilBERT'] > row['LSTM'] else '🥇 LSTM',
            axis=1
        )
        
        st.dataframe(
            df_comparison.style.format({
                'LSTM': '{:.4f}',
                'DistilBERT': '{:.4f}',
                'Difference': '{:.4f}'
            }),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Metrics Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(comparison_data['Metric']))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, comparison_data['LSTM'], width, 
                          label='LSTM', color='steelblue', edgecolor='black', alpha=0.8)
            bars2 = ax.bar(x + width/2, comparison_data['DistilBERT'], width,
                          label='DistilBERT', color='coral', edgecolor='black', alpha=0.8)
            
            ax.set_ylabel('Score', fontsize=11)
            ax.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(comparison_data['Metric'], fontsize=10)
            ax.legend(fontsize=10)
            ax.set_ylim([0, 1.1])
            ax.grid(alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📌 Key Metrics")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric(
                    "LSTM Accuracy",
                    f"{lstm_metrics['accuracy']:.2%}",
                    delta=f"{(bert_metrics['accuracy'] - lstm_metrics['accuracy'])*100:+.1f}%"
                )
            
            with col_b:
                st.metric(
                    "DistilBERT Accuracy",
                    f"{bert_metrics['accuracy']:.2%}",
                )
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric(
                    "LSTM F1-Score",
                    f"{lstm_metrics['f1']:.4f}",
                )
            
            with col_b:
                st.metric(
                    "DistilBERT F1-Score",
                    f"{bert_metrics['f1']:.4f}",
                )
        
        st.markdown("---")
        
        # Detailed comparison
        st.markdown("### 🔍 Detailed Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### LSTM Model")
            st.markdown(f"""
            **Accuracy:** {lstm_metrics['accuracy']:.2%}
            
            **Precision:** {lstm_metrics['precision']:.2%}
            
            **Recall:** {lstm_metrics['recall']:.2%}
            
            **F1-Score:** {lstm_metrics['f1']:.4f}
            
            **AUC-ROC:** {lstm_metrics['auc']:.4f}
            
            **Advantages:**
            ✓ Simpler architecture
            ✓ Faster inference
            ✓ Smaller model (~10 MB)
            """)
        
        with col2:
            st.markdown("#### DistilBERT Model")
            st.markdown(f"""
            **Accuracy:** {bert_metrics['accuracy']:.2%} ⭐
            
            **Precision:** {bert_metrics['precision']:.2%}
            
            **Recall:** {bert_metrics['recall']:.2%}
            
            **F1-Score:** {bert_metrics['f1']:.4f}
            
            **AUC-ROC:** {bert_metrics['auc']:.4f}
            
            **Advantages:**
            ✓ State-of-the-art accuracy
            ✓ Pre-trained knowledge
            ✓ Better contextualization
            """)

# ============================================================================
# PAGE 4: ANALYSIS
# ============================================================================

elif page == "📈 Analysis":
    st.markdown("# 📈 Dataset & Analysis")
    
    st.markdown("### 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Total Reviews", value="50,000")
    
    with col2:
        st.metric(label="Training Set", value="35,000", delta="70%")
    
    with col3:
        st.metric(label="Validation Set", value="7,500", delta="15%")
    
    with col4:
        st.metric(label="Test Set", value="7,500", delta="15%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📚 Dataset Characteristics")
        st.markdown("""
        **Language:** English
        
        **Task:** Binary Sentiment Classification
        - Positive: ~25,000 (50%)
        - Negative: ~25,000 (50%)
        
        **Text Statistics:**
        - Average Review Length: ~230 words
        - Min Length: 10 words
        - Max Length: 2,470 words
        
        **Vocabulary:**
        - Unique Words: ~10,000
        """)
    
    with col2:
        st.markdown("### 🔧 Preprocessing Steps")
        st.markdown("""
        1. Removed HTML tags
        2. Removed URLs
        3. Removed special characters
        4. Normalized text
        5. Tokenization
        6. Padding/Truncation
        7. Train/Val/Test Split
        """)
    
    st.markdown("---")
    st.markdown("### 📊 Data Visualizations")
    
    # Try to load visualizations
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.checkbox("Show Sentiment Distribution", value=True):
                try:
                    img1 = plt.imread('01_sentiment_distribution.png')
                    st.image(img1, caption="Sentiment Distribution")
                except:
                    st.info("Image not found")
        
        with col2:
            if st.checkbox("Show Review Length Distribution", value=True):
                try:
                    img2 = plt.imread('01_review_length_distribution.png')
                    st.image(img2, caption="Review Length Distribution")
                except:
                    st.info("Image not found")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.checkbox("Show Top Words", value=True):
                try:
                    img3 = plt.imread('01_top_words.png')
                    st.image(img3, caption="Top 15 Most Common Words")
                except:
                    st.info("Image not found")
        
        with col2:
            if st.checkbox("Show Sentiment by Length", value=True):
                try:
                    img4 = plt.imread('01_sentiment_by_length.png')
                    st.image(img4, caption="Review Length by Sentiment")
                except:
                    st.info("Image not found")
    
    except Exception as e:
        st.warning(f"Could not load visualizations: {e}")

# ============================================================================
# PAGE 5: ABOUT
# ============================================================================

elif page == "ℹ️ About":
    st.markdown("# ℹ️ About This Project")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Project Goal")
        st.write("""
        This project compares two different deep learning approaches for sentiment analysis:
        
        1. **LSTM** - Traditional RNN architecture
        2. **DistilBERT** - Modern Transformer-based pre-trained model
        """)
    
    with col2:
        st.markdown("### 📚 Dataset")
        st.write("""
        **IMDB Movie Reviews Dataset**
        
        - Size: 50,000 movie reviews
        - Language: English
        - Task: Binary sentiment classification
        - Source: Kaggle
        """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🏗️ LSTM")
        st.markdown("""
        **Architecture:**
        - Embedding → Bi-LSTM → Dense
        
        **Performance:**
        - Accuracy: ~86%
        - Parameters: ~2.5M
        
        **Pros:**
        ✓ Faster
        ✓ Smaller
        ✓ Simpler
        """)
    
    with col2:
        st.markdown("### 🤖 DistilBERT")
        st.markdown("""
        **Architecture:**
        - Pre-trained Transformer
        - 6 layers, 768 hidden dims
        
        **Performance:**
        - Accuracy: ~91%
        - Parameters: ~67M
        
        **Pros:**
        ✓ Higher accuracy
        ✓ Pre-trained
        ✓ Contextual
        """)
    
    with col3:
        st.markdown("### 🔄 Comparison")
        st.markdown("""
        **LSTM Best For:**
        - Speed needed
        - Limited resources
        - Edge devices
        
        **DistilBERT Best For:**
        - Maximum accuracy
        - Production systems
        - Cloud deployment
        """)
    
    st.markdown("---")
    st.success("✅ Thank you for using Sentiment Analysis Demo!")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    <p>Sentiment Analysis Project | LSTM vs DistilBERT Comparison</p>
    <p>Dataset: IMDB Movie Reviews | 2024</p>
</div>
""", unsafe_allow_html=True)