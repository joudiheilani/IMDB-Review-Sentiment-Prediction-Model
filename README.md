# 🎬 IMDB Sentiment Analysis

Binary sentiment classification on 50,000 IMDB movie reviews using a TF-IDF + Logistic Regression pipeline. Built to explore classical NLP techniques, confidence-aware evaluation, and generalization behavior on a balanced dataset.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Confidence Analysis](#confidence-analysis)
- [Key Observations](#key-observations)
- [Getting Started](#getting-started)
- [Limitations & Future Work](#limitations--future-work)

---

## Overview

This project trains a sentiment classifier to label IMDB movie reviews as **positive** or **negative**. The goal was to build a clean, interpretable pipeline using classical ML — prioritizing reproducibility, honest evaluation, and insight into where and why the model fails.

**Dataset:** [IMDB Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) — 50,000 reviews, balanced (25k positive / 25k negative)

---

## Project Structure

```
imdb-sentiment-analysis/
├── data/
│   └── IMDB.csv              # ⚠️ Not included — see Dataset Setup below
├── model/
│   └── sentiment_model.pkl   # Serialized trained pipeline
├── src/
│   ├── train.py              # Training, cross-validation, model export
│   └── evaluate.py           # Full-dataset evaluation & error analysis
└── README.md
```

---

## Methodology

### Pipeline

```
Raw Text → TF-IDF Vectorizer → Logistic Regression → Sentiment Label
```

**TF-IDF Configuration:**
| Parameter | Value | Rationale |
|---|---|---|
| `max_features` | 15,000 | Caps vocabulary to reduce noise |
| `ngram_range` | (1, 2) | Captures unigrams and bigrams (e.g., "not good") |
| `stop_words` | `'english'` | Removes non-discriminative tokens |
| `max_df` | 0.8 | Ignores terms appearing in >80% of documents |
| `min_df` | 5 | Ignores terms appearing in fewer than 5 documents |

**Classifier:** `LogisticRegression(max_iter=1000)` with default L2 regularization

**Train/Test Split:** 80/20 stratified split (`random_state=42`)

---

## Results

### Cross-Validation (5-Fold, on training set)

| Fold | F1 Score |
|---|---|
| 1 | 0.8954 |
| 2 | 0.8947 |
| 3 | 0.8979 |
| 4 | 0.8907 |
| 5 | 0.8913 |
| **Mean** | **0.8940** |

Low variance across folds (std ≈ 0.003) indicates stable generalization.

---

### Held-Out Test Set (10,000 samples)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Negative | 0.91 | 0.88 | 0.89 | 4,961 |
| Positive | 0.89 | 0.91 | 0.90 | 5,039 |
| **Weighted Avg** | **0.90** | **0.90** | **0.90** | **10,000** |

**Test Accuracy: 90%**

---

### Full Dataset Evaluation (50,000 samples)

> `evaluate.py` runs on the full dataset to assess overall fit. Not a substitute for held-out test performance — included for completeness.

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Negative | 0.93 | 0.91 | 0.92 | 25,000 |
| Positive | 0.91 | 0.93 | 0.92 | 25,000 |
| **Weighted Avg** | **0.92** | **0.92** | **0.92** | **50,000** |

**Total Errors: 3,899 / 50,000 — Error Rate: 7.80%**

---

## Confidence Analysis

The model outputs class probabilities via `predict_proba`. Max probability is used as a confidence score.

| Prediction Outcome | Avg Confidence |
|---|---|
| ✅ Correct | 0.845 |
| ❌ Incorrect | 0.638 |

The ~20-point gap between correct and incorrect confidence is a meaningful signal — the model is, on average, appropriately less certain when it is wrong.

### High-Confidence Errors (>0.98 confidence, wrong label)

These are the most instructive failure cases:

| Snippet | True | Predicted | Confidence |
|---|---|---|---|
| *"this is a great movie. I love the series on tv..."* | Negative | Positive | **0.995** |
| *"This flick is sterling example of the state of..."* | Positive | Negative | **0.992** |
| *"This movie was pure genius. John Waters is bri..."* | Negative | Positive | **0.985** |

These errors likely involve **sarcasm, irony, or mixed-sentiment reviews** — cases where surface-level word features mislead the model with high confidence.

---

## Key Observations

- **Stable cross-validation** (F1 range: 0.891–0.898) suggests the model is not overfit to a single split.
- **Confidence correlates with correctness** — errors cluster at lower confidence scores (~0.64 avg vs ~0.84 for correct predictions).
- **High-confidence failures** point to the core limitation of bag-of-words models: they lack semantic and contextual understanding, making sarcasm and irony invisible to the feature space.
- **Low-confidence correct predictions** (confidence ≈ 0.50) represent genuinely ambiguous reviews — not model failure, but inherent label difficulty.

---

## Getting Started

### Dataset Setup

The dataset is not included in this repository due to file size. Download it from Kaggle before running anything:

1. Go to [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
2. Download `IMDB Dataset.csv`
3. Rename it to `IMDB.csv` and place it in the `data/` folder:

```
data/
└── IMDB.csv
```

> A Kaggle account is required to download the dataset. You can also use the [Kaggle CLI](https://www.kaggle.com/docs/api):
> ```bash
> kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
> ```

---

### Prerequisites

```bash
pip install pandas numpy scikit-learn joblib
```

### Train the model

```bash
python src/train.py
```

Outputs cross-validation scores, test set classification report, confidence analysis, and saves the model to `model/sentiment_model.pkl`.

### Evaluate on full dataset

```bash
python src/evaluate.py
```

Loads the saved model and reports full-dataset metrics, error rate, and edge case samples.

---

## Limitations & Future Work

| Limitation | Potential Improvement |
|---|---|
| Bag-of-words loses word order and context | Transformer-based models (e.g., DistilBERT fine-tuned on IMDB) |
| Sarcasm and irony cause high-confidence errors | Contextual embeddings or ensemble methods |
| No preprocessing beyond TF-IDF stop words | HTML tag removal, lemmatization |
| Binary classification only | Aspect-based or multi-class sentiment |
| `evaluate.py` runs on training data | Proper holdout-only evaluation script |

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-data-lightblue?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-array-blue?logo=numpy)

---

*Built as part of an ongoing ML portfolio. Feedback welcome.*
