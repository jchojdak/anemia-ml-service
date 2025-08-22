#!/usr/bin/env python3
"""
Model training script based on the notebook analysis.
This script trains multiple ML models and saves them as joblib files.
"""

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


def load_and_prepare_data():
    """Load and prepare the anemia datasets"""
    print("[#] Loading datasets...")

    df_list = [
        pd.read_csv("data/01-biswaranjanrao-anemia-dataset.csv"),
        pd.read_csv("data/02-ragishehab-anemia-dataset.csv")
    ]

    dataset = pd.concat(df_list, ignore_index=True)
    print(f"[#] Combined dataset shape: {dataset.shape}")

    initial_shape = dataset.shape
    dataset = dataset.drop_duplicates()
    dataset = dataset.dropna()

    print(f"[#] After cleaning:")
    print(f"- Removed {initial_shape[0] - dataset.shape[0]} duplicate/missing rows")
    print(f"- Final dataset shape: {dataset.shape}")

    return dataset


def train_and_save_models(dataset):
    """Train all models and save them to files"""

    X = dataset.drop(columns=['Result'])
    y = dataset['Result']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs("models", exist_ok=True)

    models = {
        'logistic': LogisticRegression(random_state=42, max_iter=1000),
        'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
        'svm': SVC(random_state=42, probability=True),
        'knn': KNeighborsClassifier(),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'naive_bayes': GaussianNB()
    }

    models_need_scaling = ['logistic', 'svm', 'knn']

    print("[#] Training and saving models...")
    results = {}

    for name, model in models.items():
        print(f"Training {name}...")

        if name in models_need_scaling:
            scaler = StandardScaler()
            X_train_processed = scaler.fit_transform(X_train)
            X_test_processed = scaler.transform(X_test)

            # Save scaler
            scaler_path = f"models/{name}_scaler.joblib"
            joblib.dump(scaler, scaler_path)
            print(f"  - Saved scaler: {scaler_path}")
        else:
            X_train_processed = X_train
            X_test_processed = X_test

        model.fit(X_train_processed, y_train)

        y_pred = model.predict(X_test_processed)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy

        model_path = f"models/{name}_model.joblib"
        joblib.dump(model, model_path)
        print(f"  - Saved model: {model_path}")
        print(f"  - Test accuracy: {accuracy:.4f}")

    return results


def main():
    """Main training script"""
    print("Anemia Detection Model Training Script")
    print("=" * 50)

    try:
        dataset = load_and_prepare_data()

        results = train_and_save_models(dataset)

        print("\n[#] TRAINING COMPLETED")
        print("=" * 30)
        print("Model performance summary:")
        for model_name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model_name}: {accuracy:.4f}")

        print(f"\n[#] Best performing model: {max(results.items(), key=lambda x: x[1])[0]}")
        print("[#] All models saved to 'models/' directory")

    except Exception as e:
        print(f"Error during training: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
