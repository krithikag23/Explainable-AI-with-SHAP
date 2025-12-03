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
