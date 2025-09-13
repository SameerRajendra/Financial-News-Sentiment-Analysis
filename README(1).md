# Financial News Sentiment Analysis with GPT-Neo 125M

## 📌 Overview
This project applies **transformer-based language models** to classify financial news into **positive, neutral, or negative sentiment**.  
The work was completed as part of **CPE646 - Pattern Recognition and Classification (Spring 2025)** at **Stevens Institute of Technology**, under the guidance of **Dr. Hong Man**.  

## 👨‍🎓 Contributors
- **Sameer Shashikant Rajendra**  
  *MS Applied Artificial Intelligence*  
  📧 srajendr1@stevens.edu  

- **Amudhan Subramanian Manivasagam**  
  *MS Applied AI & MBA (Dual-Degree)*  
  📧 asubrama1@stevens.edu  

---

## ⚙️ Installation & Setup
Ensure you have **Python 3.10+**. Install dependencies:
```bash
pip install pandas datasets scikit-learn transformers numpy torch evaluate accelerate psycopg2-binary
```

---

## 🚀 Usage
1. **Run preprocessing & data loading**
   - `main.py` fetches and stores financial news in **QuestDB**.
   - Tables are defined in `questdb_table.sql`.

2. **Fine-tune GPT-Neo 125M**
   - Open `final_finalest_notebook.ipynb`.
   - Run cells sequentially to:
     - Load dataset  
     - Tokenize text  
     - Train sentiment classifier  

3. **Evaluate results**
   - Model performance is measured with **accuracy, precision, recall, and F1 score**.  

---

## 📊 Results

### Classification Report
| Class     | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Negative  | 0.75      | 0.52   | 0.62     | 2,381   |
| Neutral   | 0.47      | 0.78   | 0.58     | 2,008   |
| Positive  | 0.97      | 0.95   | 0.96     | 32,446  |
| **Accuracy** | **0.92** | – | – | 36,835 |
| **Macro Avg** | 0.73 | 0.75 | 0.72 | 36,835 |
| **Weighted Avg** | 0.93 | 0.92 | 0.92 | 36,835 |

### Confusion Matrix
|            | Predicted Negative | Predicted Neutral | Predicted Positive |
|------------|--------------------|-------------------|--------------------|
| Actual Negative | 1246 | 544  | 591   |
| Actual Neutral  | 189  | 1568 | 251   |
| Actual Positive | 233  | 1257 | 30,956 |

---

## 📊 Model Details
- **Base Model**: GPT-Neo 125M (HuggingFace Transformers)  
- **Task**: Sequence classification (3 labels: Negative, Neutral, Positive)  
- **Loss Function**: Cross-Entropy  
- **Optimization**: AdamW with learning rate scheduling  
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1  

---

## 🔮 Future Work
- Experiment with **larger GPT-Neo variants (1.3B, 2.7B)**.  
- Integrate **real-time streaming financial data**.  
- Deploy model via **FastAPI + Docker** for production use.  
