# Hate Speech and Sentiment Analysis

This project demonstrates the application of machine learning and deep learning techniques for text classification. The primary focus is on analyzing tweets to detect hate speech and sentiment.

## Project Overview

The main goal is to compare the performance of:
- A **Support Vector Machine (SVM)** classifier trained on TF-IDF features.
- A fine-tuned **BERT** (Bidirectional Encoder Representations from Transformers) model for multi-class classification.

## Features

- **Data Preprocessing:**
  - Cleaning tweets using the `spaCy` library.
  - Removing noise, stopwords, and irrelevant characters.

- **Feature Engineering:**
  - TF-IDF vectorization for SVM.
  - Tokenization using the BERT tokenizer for the deep learning model.

- **Model Training:**
  - Training an SVM classifier using `scikit-learn`.
  - Fine-tuning a pre-trained BERT model using the `transformers` library.

- **Model Evaluation:**
  - Assessing accuracy and generating classification reports.
  - Comparing the performance of SVM and BERT.

## Results

| Model         | Accuracy | F1-Score (Weighted) |
|---------------|----------|---------------------|
| SVM           | **89.57%** | ~88%               |
| BERT (fine-tuned) | **90.43%** | ~90%             |

### Key Observations:
1. **BERT** outperformed **SVM** in terms of accuracy and F1-score, showcasing the strength of transformer-based models for text classification tasks.
2. **SVM** still achieved decent performance and remains a lightweight option for smaller datasets.

