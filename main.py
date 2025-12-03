import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap
import os

MODEL_PATH = "model.joblib"
EXPLAINER_PATH = "shap_explainer.joblib"

def train_and_save():
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = [f.decode() if isinstance(f, bytes) else f for f in data.feature_names]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, preds))
    print("Classification report:\n", classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    # Save model and metadata
    joblib.dump({"model": clf, "feature_names": feature_names}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    # Create a SHAP TreeExplainer and save only compact payload
    # (saving full explainer can be large). We store expected_value + small data sample.
    explainer = shap.TreeExplainer(clf, X_train, feature_names=feature_names)
    explainer_payload = {
        "expected_value": explainer.expected_value,
        "data_sample": X_train[:200],  # small subset for faster loading in app
        "feature_names": feature_names,
    }
    joblib.dump(explainer_payload, EXPLAINER_PATH)
    print(f"Saved SHAP explainer payload to {EXPLAINER_PATH}")

if __name__ == "__main__":
    train_and_save()