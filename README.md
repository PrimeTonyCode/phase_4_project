
# 📊 Apple & Google Tweet Sentiment Analysis
![alt text](image.png)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

## 📖 Overview
This repository contains a complete **NLP pipeline** to classify tweet sentiment (**negative / neutral / positive**) about **Apple** and **Google** products.
The notebook walks through data cleaning, TF‑IDF features, classical ML models, **GridSearchCV** tuning, and evaluation.

---

## 🧭 Table of Contents
- [Dataset](#-dataset)
- [Reproducibility](#-reproducibility)
- [Tech Stack](#-tech-stack)
- [Quickstart](#-quickstart)
- [Results](#-results)
- [Insights](#-insights)
- [Limitations](#️-limitations)
- [Authors](#-authors)

---

## 🗂️ Dataset
- **Source:** CrowdFlower via data.world — *Brands and Product Emotions* (9,093 tweets)
- **Labels:** `negative`, `neutral`, `positive`
- **Notes:** Highly informal text (slang, emojis) and **class imbalance** (neutral dominates).

---

## 🔁 Reproducibility

**Train/Test split** 

**TF‑IDF vectorizer**

**NLTK assets:**
```python
import nltk
nltk.download('stopwords')
```

---

## ⚙️ Tech Stack
- Python 
- pandas, numpy, scikit‑learn, nltk, matplotlib, seaborn, wordcloud

---

## 🚀 Quickstart
```bash
git clone https://github.com/PrimeTonyCode/phase_4_project
cd Tweet_Sentiment_NLP
pip install -r requirements.txt
jupyter notebook Tweet_Sentiment_NLP.ipynb
```

---

## 📊 Results
| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| Logistic Regression (Multiclass) | 0.653 | 0.58 | 0.66 |
| Naive Bayes (Multiclass) | 0.589 | 0.52 | 0.61 |
| Random Forest (Multiclass) | 0.679 | 0.57 | 0.67 |
| SVM (Multiclass) | 0.653 | 0.56 | 0.67 |

**Best recorded model (by accuracy):** `Random Forest (Multiclass)` at **0.679**.

**Best hyperparameters found (from GridSearch in notebook):**
- `LogisticRegression`: {'C': 10, 'solver': 'saga'}
- `MultinomialNB`: {'alpha': 0.1}
- `SVM`: {'C': 8.424426408004217, 'kernel': 'rbf'}
- RandomForest tuned via GridSearchCV (best params printed in the notebook outputs).

---

## 🔍 Insights
- Apple tweets tend to be **more negative** (esp. around *iPad* references).
- Google tweets skew **more positive** (notably for *Maps* and *Search*).
- The **neutral** class is most frequent → consider class balancing, thresholding, or cost‑sensitive training.

---

## ⚠️ Limitations
- Historical dataset (pre‑2013) — results may not reflect current sentiment.
- Sarcasm, context outside the tweet, and multimodal cues are not captured.

---

## 📄 Authors
