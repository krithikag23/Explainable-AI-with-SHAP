
import joblib

def load_model(path="model.joblib"):
    payload = joblib.load(path)
    return payload["model"], payload.get("feature_names", None)

def load_explainer_payload(path="shap_explainer.joblib"):
    payload = joblib.load(path)
    return payload