import os
import zipfile
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

ZIP_PATH = "AssessorExportCSV.zip"
EXTRACT_DIR = "data_assessor"

def unzip_and_get_csv(zip_path: str, extract_dir: str) -> str:
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    csv_files = []
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    if not csv_files:
        raise FileNotFoundError("No CSV found inside zip.")

    return max(csv_files, key=lambda p: os.path.getsize(p))

@st.cache_data
def load_data() -> pd.DataFrame:
    csv_path = unzip_and_get_csv(ZIP_PATH, EXTRACT_DIR)
    df = pd.read_csv(csv_path, low_memory=False)
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Clean target
    df["APPRAISED_VALUE"] = (
        df["APPRAISED_VALUE"].astype(str).str.replace(r"[\$,]", "", regex=True)
    )
    df["APPRAISED_VALUE"] = pd.to_numeric(df["APPRAISED_VALUE"], errors="coerce")
    df = df.dropna(subset=["APPRAISED_VALUE"])
    df = df[df["APPRAISED_VALUE"] > 0].copy()

    # Residential filter
    type_cols = ["PROPERTY_CLASS", "CURRENT_USE_CODE_DESC", "CURRENT_USE_CODE"]
    res_col = next((c for c in type_cols if c in df.columns), None)

    if res_col is not None:
        pattern = "RES|RESIDENT|SINGLE|DWELL|CONDO|TOWN|APART|MULTI|1-FAM|2-FAM|3-FAM|1 FAMILY|2 FAMILY|3 FAMILY"
        df = df[df[res_col].astype(str).str.contains(pattern, case=False, na=False)].copy()

    return df

def train_model(df: pd.DataFrame):
    # Pick a small, stable set of features if available; otherwise fall back to numeric columns.
    preferred = ["YEAR_BUILT", "BEDROOMS", "FULL_BATH", "HALF_BATH", "TOTAL_LIVING_AREA", "LOT_SIZE", "ACRES"]
    features = [c for c in preferred if c in df.columns]

    if not features:
        # fallback: numeric predictors (excluding target)
        features = [c for c in df.select_dtypes(include=np.number).columns if c != "APPRAISED_VALUE"]

    # Build X/y
    y = df["APPRAISED_VALUE"].copy()
    X_raw = df[features].copy()

    # One-hot encode (in case some are not numeric)
    X = pd.get_dummies(X_raw, drop_first=True)
    X = X.select_dtypes(include=[np.number, "bool"]).astype(float)
    X = X.fillna(X.median(numeric_only=True))

    # Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X.columns.tolist(), df

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Hamilton County Property Value Predictor")
st.write("Predicts **APPRAISED_VALUE** using a Linear Regression model trained on the assessor dataset.")
st.warning("Disclaimer: Educational use only.")

df = load_data()
df = preprocess(df)

if df.empty:
    st.error("No data available after preprocessing (check residential filter / APPRAISED_VALUE cleaning).")
    st.stop()

model, feature_names, df_clean = train_model(df)

st.subheader("Enter Inputs")

user_vals = {}
for feat in feature_names:
    # Use median of available column if the raw column exists; otherwise default 0
    base_name = feat.split("_")[0]
    default = 0.0
    if base_name in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[base_name]):
        default = float(df_clean[base_name].median())
    user_vals[feat] = st.number_input(feat, value=float(default))

X_user = pd.DataFrame([user_vals])[feature_names]
pred = model.predict(X_user)[0]

st.subheader("Predicted APPRAISED_VALUE")
st.write(f"**${pred:,.0f}**")
