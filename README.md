# 📊 Apple & Google Tweet Sentiment Analysis — End‑to‑End NLP Project
![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-brightgreen) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

![Alt text](output\combined_wordclouds.png)

## 📖 Overview
An end‑to‑end **Natural Language Processing** pipeline that classifies Twitter posts about **Apple** and **Google** products into **negative / neutral / positive** sentiments. The project demonstrates a complete workflow: data acquisition, **EDA**, rigorous **text preprocessing**, **TF‑IDF** feature extraction, traditional ML models (Logistic Regression, Naive Bayes, Random Forest, SVM), **hyperparameter tuning**, evaluation, and business‑oriented interpretation. Two tracks are covered: **Binary** (positive vs negative) and **Multiclass** (negative/neutral/positive) for nuanced monitoring.


## 💼 Stakeholders & Business Value
- **Product Teams:** monitor customer satisfaction, detect pain points, and prioritize fixes.
- **Marketing team:** track campaign impact and optimize messaging by sentiment trends.
- **Analysts:** study consumer behavior and topic salience on social media.
- **Investors:** derive competitive insights from public perception.
- **Customers:** benefit indirectly through data‑driven product and support improvements.
**Business value:** sentiment dashboards to rapidly identify issues, inform roadmap decisions, and guide engagement strategies.

## 🧭 Table of Contents
- [Project Structure](#️-project-structure)
- [Dataset](#️-dataset)
- [Exploratory Data Analysis](#-exploratory-data-analysis-eda)
- [Data Cleaning & Preprocessing](#-data-cleaning--preprocessing)
- [Feature Engineering](#-feature-engineering)
- [Problem Setups](#-problem-setups)
- [Training & Hyperparameter Tuning](#-training--hyperparameter-tuning)
- [Results](#-results-)
- [Interpretation & Insights](#-interpretation--insights)
- [Reproducibility](#-reproducibility)
- [Usage](#️-usage)
- [Roadmap/Future Work](#-roadmap--future-work)
- [Limitations & Ethics](#️-limitations--ethics)
- [Authors](#-authors)

## 🗂️ Project Structure
```
├── Tweet_Sentiment_NLP.ipynb                # Main analysis notebook
├── Tweet_Sentiment_Analysis_Presentation.pptx  # Slide deck (summary of findings)
├── README.md                                # (this file)
├── requirements.txt                         # Python dependencies
└── data/                                    # (optional) dataset storage (not included)
```

## 🗃️ Dataset
- **Source:** CrowdFlower via data.world — *Brands and Product Emotions* (**9,093** labeled tweets).
- **Labels:** `negative`, `neutral`, `positive`.
- **Text characteristics:** informal syntax, slang, emojis, hashtags, mentions.
- **Notebook reads from:** `judge-1377884607_tweet_product_company.csv`

- **Suitability:** good for supervised learning; reflects the challenges of real‑world social media data.


## 🔎 Exploratory Data Analysis (EDA)
- **Sentiment distribution:** neutral dominates (~5k), positives (~2.5–3k), negatives (~0.5–0.7k).
- **Linguistic contrast:** positives include *great, awesome*; negatives include *headache, fascist*.
- **Tweet length:** negative tweets tend to be longer on average than neutral/positive.
- **Brand/product field:** missing entries are filled with `Unknown` to retain samples.

## 🧹 Data Cleaning & Preprocessing
- Drop missing/duplicate tweets; ensures consistent text encoding.
- Text pipeline: Lemmatization/Stemming, Lowercasing, Regex cleanup (URLs, mentions, hashtags, punctuation), Stopword removal, Tokenization.
- Preserve emojis/hashtags when informative.
- Strip URLs/mentions during normalization

## 🧰 Feature Engineering
- **TF‑IDF** vectorization with unigrams/bigrams; vocabulary constrained for generalization.
- **Vectorizer parameters**
- **stop_words**
- **max_features** = 5000
- **ngram_range** = (1,2)

## 🔀 Problem Setups
### 1) Binary Classification
- Focus on **positive vs negative**; neutral/ambiguous removed for crisp decisions.
- **Split:** 80/20 with stratification.
- **Models:** Logistic Regression, Multinomial Naive Bayes, Random Forest, SVM; also **Extra Trees** and **XGBoost** in experimentation.

### 2) Multiclass Classification
- Three classes: **negative, neutral, positive**.
- **Class imbalance:** neutral majority; addressed via resampling and evaluation by per‑class metrics.
- **Models:** Logistic Regression, Multinomial Naive Bayes, Random Forest, SVM.
- **Balancing:** to improve minority class representation.

## 🧪 Training & Hyperparameter Tuning
- Stratified **train/test split** for fair evaluation.
- **Split:** test_size=0.2, random_state=42, stratify=True.
- **GridSearchCV** used to tune key hyperparameters.
- Best parameter snapshots are printed in the notebook outputs.

## 📊 Results 📈

### Results — Binary Classification
- **Accuracy (top):** Extra Trees ≈ **0.89**, XGBoost ≈ **0.88**.
- **Precision (positive):** Logistic Regression ≈ **0.93**.
- **Recall (positive):** MultinomialNB = **1.00**.
- **Macro‑F1 (best):** Logistic Regression & SVM ≈ **0.75**.
- **ROC‑AUC (best):** Extra Trees ≈ **0.90**.

- **Confusion‑matrix patterns:**
  - Logistic Regression: TN=74, FP=40, FN=62, TP=532
  - MultinomialNB: TN=9, FP=105, FN=0, TP=594
  - Random Forest: TN=27, FP=87, FN=5, TP=589
  - SVM: TN=72, FP=42, FN=61, TP=533
  - Extra Trees: TN=42, FP=72, FN=8, TP=586 

### Results — Multiclass Classification
| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| Logistic Regression (Multiclass) | 0.653 | 0.580 | 0.660 |
| Naive Bayes (Multiclass) | 0.589 | 0.520 | 0.610 |
| Random Forest (Multiclass) | 0.679 | 0.570 | 0.670 |
| SVM (Multiclass) | 0.653 | 0.560 | 0.670 |
**Best test accuracy:** `Random Forest (Multiclass)` → **0.679**.

### Notable tuning outcomes
- **Random Forest**: best test accuracy ≈ **0.684** with balanced metrics.
- **SVM**: best cross‑validation score ≈ **0.846** with `kernel='rbf'`, `C≈8.42`.

## 🔍 Interpretation & Insights
- **Neutral dominates**; overall accuracy can be misleading — inspect per‑class metrics.
- **Negative** tweets are hardest; all models struggle to distinguish them cleanly from neutral/positive.
- **Binary track** yields stronger decision quality for clear positive/negative detection.
- For production: Random Forest is a robust, interpretable default; SVM competitive but heavier to tune.
- Apple tweets tend to be **more negative** (esp. around *iPad* references).
- Google tweets skew **more positive** (notably for *Maps* and *Search*).

## 🔁 Reproducibility
- Pin dependencies via `requirements.txt`; use Python **3.8+**.
- Set seeds (e.g., `random_state=42`) for deterministic splits and models.
- Keep vectorizer config consistent (`max_features`, `ngram_range`, `stop_words`).

## ▶️ Usage
1) Install deps and open the notebook:
```bash
pip install -r requirements.txt
jupyter notebook Tweet_Sentiment_NLP.ipynb
```
2) Execute cells to reproduce preprocessing, modeling, and plots.

## 🧭 Roadmap / Future Work
- Replace TF‑IDF with **transformer embeddings**  or **Sentence‑BERT** for semantic lift.
- Add **topic modeling**  to pair sentiment with themes.
- Build a lightweight **inference API**  and a **dashboard**  for monitoring.
- Introduce **active learning** and **human‑in‑the‑loop** labeling for continuous improvement.
- Expand to **multilingual** tweets; handle code‑switching.

## ⚠️ Limitations & Ethics
- Historical dataset; may not reflect current brand sentiment.
- Sarcasm, humor, and context outside the tweet remain challenging.

## 📄 Authors
- **ANTONY Njoroge**
- **JEDIDA Muriira**
- **KEVINE Kimutai**
- **RACHEL Odhiambo**
- **WINNIE Amoit**
- **NAVROS Lewis**
