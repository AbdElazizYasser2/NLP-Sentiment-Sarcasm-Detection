# 🧠 NLP: Sentiment & Sarcasm Detection Using BiLSTM and BERT

A deep learning NLP project that builds and compares models for **Sentiment Analysis** and **Sarcasm Detection** using two architectures: **Bidirectional LSTM** and **BERT (Transformers)**.

---

## 📌 Project Overview

This project tackles two challenging NLP tasks:

- **Sentiment Analysis** → Is the text Positive or Negative? (IMDB Reviews Dataset)
- **Sarcasm Detection** → Is the text Sarcastic or Not? (News Headlines Dataset)

Each task is solved using two different deep learning models:
- **BiLSTM** — Bidirectional Long Short-Term Memory (traditional deep learning)
- **BERT** — `bert-base-uncased` from Hugging Face (transformer-based)

---
## 📊 Datasets

| Dataset | Link |
|---|---|
| **IMDB Movie Reviews** | [Download from Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |
| **Sarcasm Headlines Dataset** | [Download from Kaggle](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection) |
---

## 🔧 Project Pipeline

### 1. 🧹 Text Cleaning
- Converted text to lowercase
- Removed URLs and special characters

### 2. 🏷️ Label Encoding
- Sentiment: `positive → 1`, `negative → 0`
- Sarcasm: `is_sarcastic` (already binary)

### 3. 🔤 Tokenization & Padding
- Used Keras Tokenizer (`num_words=5000`)
- Applied padding to fixed length (`maxlen=100`)

### 4. ✂️ Train/Test Split
- 80% Training / 20% Testing
- `random_state=42` for reproducibility

### 5. ⚖️ Handling Imbalanced Data
- Applied class weights for Sarcasm model to handle class imbalance

---

## 🤖 Models

### BiLSTM Models
```
Embedding → Bidirectional LSTM → Dropout → Dense (sigmoid)
```
- Loss: Binary Crossentropy
- Optimizer: Adam
- Epochs: 5

### BERT Models
```
bert-base-uncased → Fine-tuning → Classification Head
```
- Framework: PyTorch
- Optimizer: Adam (lr=2e-5)
- Mixed precision training (GradScaler)
- Epochs: 3
- Batch size: 16

---

## 📈 Evaluation Metrics

| Metric | Description |
|---|---|
| **Accuracy** | Overall correct predictions |
| **F1-Score** | Balanced metric (Precision + Recall) |
| **Confusion Matrix** | Detailed TP, TN, FP, FN breakdown |

---

## 📊 Model Comparison

| Model | Task |
|---|---|
| LSTM Sentiment | Sentiment Analysis |
| BERT Sentiment | Sentiment Analysis |
| LSTM Sarcasm | Sarcasm Detection |
| BERT Sarcasm | Sarcasm Detection |

> Full comparison visualized using bar charts (Accuracy + F1-Score)

---

## 🔍 Explanation Layer

Built a custom **Explanation Layer** that shows:
- Cleaned input text
- Keyword highlights
- Model predictions with confidence scores

Example:
```python
full_pipeline("Perfect! Just what I needed today, more problems.")
full_pipeline("it is so bad.")
full_pipeline("it is so smart.")
```

---

## 💾 Model Saving

Trained models saved using Hugging Face `save_pretrained()`:
```
bert_sentiment/
bert_sarcasm/
```

---

## 🛠️ Tech Stack

| Tool | Usage |
|---|---|
| **Python** | Programming Language |
| **TensorFlow / Keras** | BiLSTM Models |
| **PyTorch** | BERT Training |
| **Hugging Face Transformers** | BERT Model & Tokenizer |
| **Scikit-learn** | Metrics & Data Split |
| **Pandas & NumPy** | Data Handling |
| **Matplotlib & Seaborn** | Visualization |
| **Google Colab** | Development Environment |

---

## 📁 Project Structure

```
├── NLP.ipynb                    # Main Jupyter Notebook
├── IMDB Dataset.csv             # Sentiment Analysis Dataset
├── Sarcasm_Headlines_Dataset.json  # Sarcasm Detection Dataset
├── bert_sentiment/              # Saved BERT Sentiment Model
├── bert_sarcasm/                # Saved BERT Sarcasm Model
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repo
```bash
git clone https://github.com/AbdElazizYasser2/NLP-Sentiment-Sarcasm-Detection.git
cd NLP-Sentiment-Sarcasm-Detection
```

### 2. Install dependencies
```bash
pip install tensorflow torch transformers scikit-learn pandas numpy matplotlib seaborn
```

### 3. Run the notebook
```bash
jupyter notebook NLP.ipynb
```

> ⚡ Recommended to run on **Google Colab** with GPU for faster training.

---
