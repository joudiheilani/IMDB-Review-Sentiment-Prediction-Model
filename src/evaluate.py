import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report

def evaluate():
    df = pd.read_csv("data/IMDB.csv")

    X_text = df['review']
    y = df['sentiment'].map({
        'negative': 0,
        'positive': 1
    })

    model = joblib.load("model/sentiment_model.pkl")

    predictions = model.predict(X_text)

    print("Classification Report:\n")
    print(classification_report(y, predictions))

    probabilities = model.predict_proba(X_text)
    confidence = np.max(probabilities, axis = 1)

    results = pd.DataFrame({
        'review': X_text,
        'true': y,
        'pred': predictions,
        'confidence': confidence
    })

    correct = results[results['true'] == results['pred']]
    incorrect = results[results['true'] != results['pred']]

    print("\nAvg Confidence (Correct):", correct['confidence'].mean())
    print("Avg Confidence (Incorrect):", incorrect['confidence'].mean())

    print("\nTotal Samples:", len(results))
    print("Total Errors:", len(incorrect))
    print("Error Rate:", len(incorrect) / len(results))

    print("\nHigh Confidence Errors:")
    print(incorrect.sort_values(by = 'confidence', ascending = False).head(5))

    print("\nLow Confidence Correct Predictions:")
    print(correct.sort_values(by = 'confidence').head(5))

if __name__ == "__main__":
    evaluate()