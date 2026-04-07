import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def main():
    df = pd.read_csv("data/IMDB.csv")

    X_text = df['review']
    y = df['sentiment'].map({
        'negative': 0,
        'positive': 1
    })

    X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size = 0.2, random_state = 42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features = 15000, stop_words = 'english', ngram_range = (1, 2), max_df = 0.8, min_df = 5)),
        ('model', LogisticRegression(max_iter = 1000))
    ])

    print("Cross-Validation Results")
    c_val_scores = cross_val_score(pipeline, X_train_text, y_train, cv = 5, scoring = 'f1')
    print("CV scores:", c_val_scores)
    print("Mean CV F1:", c_val_scores.mean())

    pipeline.fit(X_train_text, y_train)

    y_prediction = pipeline.predict(X_test_text)

    print("\nTest Set Performance:")
    print(classification_report(y_test, y_prediction))

    print("\nConfidence Analysis Results")
    probabilities = pipeline.predict_proba(X_test_text)
    confidence = np.max(probabilities, axis = 1)

    results = pd.DataFrame({
        'review': X_test_text,
        'true': y_test,
        'prediction': y_prediction,
        'confidence': confidence
    })

    errors = results[results['true'] != results['prediction']]

    print("\nSample Errors:")
    print(errors.sample(5))

    print("\nAverage Confidence (Correct):", results[results['true'] == results['prediction']]['confidence'].mean())

    print("Average Confidence (Incorrect):", errors['confidence'].mean())

    joblib.dump(pipeline, "model/sentiment_model.pkl")
    print("\nModel saved to model/sentiment_model.pkl")

if __name__ == "__main__":
    main()