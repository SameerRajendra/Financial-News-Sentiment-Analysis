# Financial News Sentiment Analysis — GPT-Neo 125M Fine-Tuning

> Fine-tuned **GPT-Neo 125M** as a 3-class financial sentiment classifier using Hugging Face + PyTorch. Achieves **92% accuracy** and a **0.72 macro F1-score** on 36,000+ financial news articles, with custom class-imbalance handling via a `WeightedTrainer`.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Key Results](#key-results)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Requirements](#requirements)

---

## Overview

Financial news sentiment is highly imbalanced in the real world — positive articles dominate. This project addresses that challenge head-on:

- **GPT-Neo 125M** fine-tuned as a sequence classifier (Negative / Neutral / Positive)
- **QuestDB** pipeline for scalable article ingestion and querying
- **Custom `WeightedTrainer`** with per-class weighted loss to correct for severe class imbalance
- **On-demand inference function** for sentiment classification + confidence scoring

---

## Key Results

### Classification Report (Test Set: 36,835 articles)

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 0.75 | 0.52 | 0.62 | 2,381 |
| Neutral | 0.47 | 0.78 | 0.58 | 2,008 |
| Positive | 0.97 | 0.95 | 0.96 | 32,446 |
| **Accuracy** | | | **0.92** | 36,835 |
| **Macro Avg** | 0.73 | 0.75 | **0.72** | 36,835 |
| **Weighted Avg** | 0.93 | 0.92 | 0.92 | 36,835 |

### Confusion Matrix

| | Pred: Negative | Pred: Neutral | Pred: Positive |
|--|----------------|---------------|----------------|
| **Actual: Negative** | 1,246 | 544 | 591 |
| **Actual: Neutral** | 189 | 1,568 | 251 |
| **Actual: Positive** | 233 | 1,257 | 30,956 |

---

## Architecture

```text
  [ QuestDB ]
       │  Financial news articles
       ▼
  [ Data Pipeline (main.py) ]
       │  Preprocessing + Tokenization
       ▼
  [ GPT-Neo 125M (HuggingFace) ]
       │  Fine-tuned with WeightedTrainer
       │  Loss: Weighted Cross-Entropy
       │  Optimizer: AdamW + LR scheduler
       ▼
  [ Inference Function ]
       │  Sentiment label + confidence score
       ▼
  [ Output: Negative / Neutral / Positive ]
```

**Why class weighting?** The dataset has ~88% Positive articles. Without correction, the model collapses to predicting Positive for everything. A custom `WeightedTrainer` computes per-class weights inversely proportional to frequency, forcing the model to learn minority class boundaries.

---

## Repository Structure

```
Financial-News-Sentiment-Analysis/
├── main.py                          # Data ingestion pipeline (QuestDB)
├── questdb_table.sql                # QuestDB schema definition
├── final_finalest_notebook.ipynb    # Full training + evaluation notebook
├── Pattern_Report.pdf               # Full project report
└── README.md
```

---

## Setup & Installation

```bash
git clone https://github.com/SameerRajendra/Financial-News-Sentiment-Analysis.git
cd Financial-News-Sentiment-Analysis
pip install pandas datasets scikit-learn transformers numpy torch evaluate accelerate psycopg2-binary
```

### QuestDB Setup (for data pipeline)
```bash
# Run QuestDB via Docker
docker run -p 9000:9000 -p 8812:8812 questdb/questdb
# Then apply schema:
psql -h localhost -p 8812 -U admin -f questdb_table.sql
```

---

## Usage

```bash
# Step 1: Load and store data in QuestDB
python main.py

# Step 2: Fine-tune the model
# Open final_finalest_notebook.ipynb and run cells sequentially
jupyter notebook final_finalest_notebook.ipynb
```

### Inference Example
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained("./saved_model")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

def predict_sentiment(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    label = ["Negative", "Neutral", "Positive"][probs.argmax().item()]
    confidence = probs.max().item()
    return label, confidence

label, conf = predict_sentiment("Fed raises interest rates by 50 basis points.")
print(f"{label} ({conf:.1%} confidence)")
```

---

## Dataset

Financial PhraseBank dataset sourced from Hugging Face Datasets. Articles stored in QuestDB for structured querying.

- **Total articles:** 36,835 (test set)
- **Classes:** Negative (6.5%), Neutral (5.5%), Positive (88%)

---

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
scikit-learn>=1.3.0
psycopg2-binary>=2.9.0
accelerate>=0.20.0
```

---

*Course: Stevens Institute of Technology • Jan–May 2025*  
*Authors: [Sameer Rajendra](https://github.com/SameerRajendra), Amudhan Subramanian Manivasagam*  
*[📄 Read the Full Report](https://raw.githubusercontent.com/SameerRajendra/Financial-News-Sentiment-Analysis/main/Pattern_Report.pdf)*
