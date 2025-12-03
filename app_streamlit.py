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