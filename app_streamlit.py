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

# Compute shap values
shap_values = explainer.shap_values(sample)
# For binary classification with tree models shap_values is usually a list: [neg_class, pos_class]
if isinstance(shap_values, list):
    shap_for_pos = shap_values[1]  # positive class contributions
else:
    shap_for_pos = shap_values

# Table of top features
df = pd.DataFrame({
    "feature": feature_names,
    "value": sample.flatten(),
    "shap": shap_for_pos.flatten()
}).astype({'feature': str})
df["abs_shap"] = df["shap"].abs()
df_sorted = df.sort_values("abs_shap", ascending=False).head(10).drop(columns="abs_shap")
if show_all:
    st.subheader("Top 10 features (by absolute SHAP value)")
    st.table(df_sorted.set_index("feature"))

# Waterfall / force plot (static fallback)
if show_waterfall:
    st.subheader("SHAP waterfall (static fallback)")
    try:
        # Preferred: use shap.plots.waterfall if available
        fig = plt.figure(figsize=(10, 4))
        try:
            # shap 0.40+ has shap.plots.waterfall
            shap.plots.waterfall(explainer.expected_value[1] if isinstance(explainer.expected_value, (list, tuple, np.ndarray)) else explainer.expected_value, shap_for_pos[0], feature_names=feature_names, show=False)
            st.pyplot(fig)
        except Exception:
            # fallback to legacy waterfall or bar plot
            # Bar plot of top contributions:
            top = df_sorted.copy()
            top = top.iloc[::-1]
            ax = top.plot.barh(x="feature", y="shap", legend=False)
            ax.set_xlabel("SHAP value (contribution to prediction)")
            plt.tight_layout()
            st.pyplot(ax.figure)
    except Exception as e:
        st.write("Could not render SHAP waterfall/plot in this environment. Try running locally or see README for interactive instructions.")
        st.write(str(e))

st.info("Tip: For interactive SHAP force/JS plots, run a Jupyter notebook and use `shap.initjs()` + `shap.plots.force`.")
