import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
import shap
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(layout="wide")
st.title("XAI Demo — Breast Cancer (RandomForest + SHAP)")

# Load model
try:
    payload = joblib.load("model.joblib")
    model = payload["model"]
    feature_names = payload["feature_names"]
except Exception:
    st.error("Model not found. Please run `python main.py` first to train and save the model.")
    st.stop()

data = load_breast_cancer()
X, y = data.data, data.target

st.sidebar.header("Select sample and options")
idx = st.sidebar.number_input("Sample index (0..n-1)", min_value=0, max_value=len(X)-1, value=0)
show_all = st.sidebar.checkbox("Show top 10 SHAP features table", value=True)
show_waterfall = st.sidebar.checkbox("Show waterfall/force plot", value=True)

sample = X[idx:idx+1]

# Load SHAP explainer payload
explainer = None
try:
    explainer_payload = joblib.load("shap_explainer.joblib")
    explainer = shap.TreeExplainer(model, explainer_payload["data_sample"], feature_names=explainer_payload["feature_names"])
except Exception:
    st.warning("SHAP explainer payload not found or failed to load. Creating TreeExplainer from full training data (may be slower).")
    explainer = shap.TreeExplainer(model, X, feature_names=feature_names)
