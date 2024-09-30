# NLP Tweets Sentiment Analysis with Error Analysis

## Project Overview

This project focuses on sentiment analysis using natural language processing (NLP) techniques. The goal is to classify text data into positive or negative sentiments and analyze the performance of the model, including the identification of false positives and false negatives through error analysis.

### Key Features:

- **Sentiment Classification**: Implements machine learning classifiers to predict sentiment.
- **Error Analysis**: Detailed analysis of misclassifications, highlighting the most common issues leading to false positives and false negatives.
- **Preprocessing**: Involves extensive data preprocessing, including tokenization, stop-word removal, stemming, lemmatization, and the use of external libraries like TextBlob for sentiment correction.
- **Performance Metrics**: Model evaluation metrics include precision, recall, F1-score, and accuracy.

## Dataset

The dataset used consists of labeled text samples, where each sample is either positive or negative. The data is processed to extract key features and passed through various classifiers to predict the sentiment.

## Methodology

1. **Preprocessing**:

   - Tokenization, stop-word removal, stemming, and lemmatization were applied to clean the text data.
   - Use of TextBlob to handle complex sentiment patterns.
   - Feature extraction using N-grams, TF-IDF, and document length to improve classifier accuracy.

2. **Modeling**:

   - Implemented classifiers, including Support Vector Machine (SVM) and Logistic Regression, to predict sentiment.

3. **Error Analysis**:

   - Analyzed false positives and false negatives.
   - Identified patterns in misclassified data to improve model performance.

4. **Evaluation**:
   - Accuracy, precision, recall, and F1-score were used to evaluate the performance.
   - Achieved a best accuracy of 84% using optimized feature extraction techniques.

## Requirements

To run this project, install the following dependencies:

```
pandas==1.3.3
numpy==1.21.2
scikit-learn==0.24.2
nltk==3.6.2
textblob==0.15.3
```

Install the required libraries by running:

```bash
pip install -r requirements.txt
```

## Usage

1. **Preprocess the Data**: Use the provided Jupyter notebook to preprocess the text dataset.
2. **Train the Model**: Train the sentiment analysis model using the cleaned and preprocessed data.
3. **Evaluate Performance**: Evaluate the model using the provided performance metrics and conduct error analysis using the `error_analysis.txt` file.

## Results

- Best accuracy: 84% using SVM with optimized N-grams and document length features.
- Detailed error analysis reveals common sources of misclassification, particularly false positives in ambiguous or sarcastic text samples.

## Conclusion

This project provides a comprehensive approach to sentiment analysis, with a focus on both model performance and error analysis. Future improvements could involve exploring more complex models like deep learning and incorporating additional sentiment-specific features.
