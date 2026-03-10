import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

INPUT_FILE = "sentiment_system/data/training_dataset.csv"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "random_forest_model.pkl")


def main():

    print("Loading training dataset...")

    df = pd.read_csv(INPUT_FILE)

    # Features used for prediction
    features = [
        "sentiment_strength",
        "rolling_3_sentiment",
        "rolling_7_sentiment",
        "news_count_per_day"
    ]

    X = df[features]
    y = df["direction"]

    print("Dataset size:", len(df))

    print("Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("Evaluating model...")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nModel Accuracy:", accuracy)

    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))

    # Show feature importance (good for explaining model)
    importance = model.feature_importances_

    print("\nFeature Importance:")

    for f, score in zip(features, importance):
        print(f"{f}: {round(score,4)}")

    # Ensure models folder exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\nSaving model...")

    joblib.dump(model, MODEL_FILE)

    print("Model saved to:", MODEL_FILE)


if __name__ == "__main__":
    main()