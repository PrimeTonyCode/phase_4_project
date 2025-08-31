# 📊 Apple and Google Tweet Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📖 Overview
This project applies **Natural Language Processing (NLP)** to analyze public sentiment toward **Apple** and **Google** products using a dataset of **9,093 labeled tweets**. Tweets are classified into **positive, negative, or neutral**, providing actionable insights into brand perception.  

We demonstrate a full **end-to-end ML workflow**:
- Text preprocessing
- Feature engineering with **TF-IDF** and embeddings
- Supervised learning with **Logistic Regression, Random Forest, Naive Bayes, and SVM**
- Hyperparameter tuning with **GridSearchCV**
- Model evaluation with classification reports, confusion matrices, and performance plots

---

## 🗂️ Dataset
- **Source:** [CrowdFlower via data.world](https://data.world/crowdflower/brands-and-product-emotions)
- **Size:** 9,093 tweets
- **Labels:** `positive`, `negative`, `neutral`
- **Features:**
  - Raw tweet text (hashtags, mentions, emojis, slang)
  - Optional brand/product target (e.g., *iPad*, *Google Maps*)

⚠️ **Limitations:**
- Collected pre-2013 → not fully reflective of modern trends
- Class imbalance (neutral tweets dominate)
- Informal language requires extensive preprocessing

---

## ⚙️ Tech Stack
- **Languages:** Python
- **Core Libraries:**
  - Data: `pandas`, `numpy`
  - NLP: `nltk`, `scikit-learn`, `wordcloud`
  - Visualization: `matplotlib`, `seaborn`
  - Modeling: `LogisticRegression`, `RandomForest`, `NaiveBayes`, `SVM`, `GridSearchCV`

---

## 📑 Table of Contents
1. [Installation](#-installation)
2. [Usage](#-usage)
3. [Results](#-results)
4. [Author](#-author)

---

## 🚀 Installation
Clone the repository:
```bash
git clone https://github.com/PrimeTonyCode/phase_4_project
cd Tweet_Sentiment_NLP
```

Install dependencies:
``

Run the notebook:
```bash
jupyter notebook Tweet_Sentiment_NLP.ipynb
```

---

## ▶️ Usage
- **Preprocessing:** Cleans tweets, removes URLs/mentions/hashtags, tokenizes, and lemmatizes.
- **Feature Engineering:** Converts text into TF-IDF vectors.
- **Modeling:** Trains and tunes multiple classifiers.
- **Evaluation:** Outputs classification reports, confusion matrices, and sentiment distribution plots.

---

## 📊 Results

| Model | Accuracy | Macro F1 | Notes |
|-------|----------|----------|-------|
| Logistic Regression (Tuned) | 66% | 0.58 | Balanced, interpretable |
| Naive Bayes (Tuned) | 62% | 0.54 | Good at negatives, weaker elsewhere |
| Random Forest (Tuned) | **68%** | 0.58 | Strongest recall for neutrals |
| SVM (Tuned) | **68%** | 0.56 | High precision on neutrals |

🔍 **Key Insights:**
- Apple received more **negative tweets**, especially about the *iPad*.
- Google showed higher **positive sentiment** for *Maps* and *Search*.
- Neutral tweets dominate, suggesting much of the conversation is informational.



---


---

## 👤 Authors
