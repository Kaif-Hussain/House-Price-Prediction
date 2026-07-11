import pickle
import pandas as pd
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

with open(ARTIFACTS_DIR / "model.pkl", "rb") as f:
    model = pickle.load(f)

with open(ARTIFACTS_DIR / "model_meta.json") as f:
    meta = json.load(f)

def predict(features: dict):
    df = pd.DataFrame([features])[meta["features"]]
    pred = model.predict(df)[0]
    return round(float(pred), 4)