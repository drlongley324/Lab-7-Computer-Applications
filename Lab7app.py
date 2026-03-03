import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl")

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Clean APPRAISED_VALUE
    df = df.copy()
    df["APPRAISED_VALUE"] = (
        df["APPRAISED_VALUE"]
        .astype(str)
        .str.replace(r"[\$,]", "", regex=True)
    )
    df["APPRAISED_VALUE"] = pd.to_numeric(df["APPRAISED_VALUE"], errors="coerce")
    df = df.dropna(subset=["APPRAISED_VALUE"])
    df = df[df["APPRAISED_VALUE"] > 0]

    # Residential filter
    type_cols = ["PROPERTY_CLASS", "CURRENT_USE_CODE_DESC", "CURRENT_USE_CODE"]
    res_col = next((c for c in type_cols if c in df.columns), None)

    if res_col is not None:
        residential_keywords = [
            "RES", "RESIDENT", "SINGLE", "DWELL", "CONDO", "TOWN",
            "APART", "MULTI", "1-FAM", "2-FAM", "3-FAM", "1 FAMILY", "2 FAMILY", "3 FAMILY"
        ]
        pattern = "|".join(residential_keywords)
        df = df[df[res_col].astype(str).str.contains(pattern, case=False, na=False)].copy()

    return df

def build_model(df: pd.DataFrame):
    # Use numeric predictors only
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Build X, y
    y = df["APPRAISED_VALUE"]
    X = df[numeric_cols].drop(columns=["APPRAISED_VALUE"], errors="ignore")

    data = pd.concat([X, y], axis=1).dropna()
    X = data.drop(columns=["APPRAISED_VALUE"])
    y = data["APPRAISED_VALUE"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X.columns

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Hamilton County Property Value Predictor")
st.write(
    "This app predicts **APPRAISED_VALUE** using a simple Linear Regression model trained on the "
    "Hamilton County Assessor dataset."
)

st.info("Disclaimer: Educational use only. Predictions may be inaccurate.")

# Load + preprocess
data_path = "Housing_Hamilton_County.xlsx"
df = load_data(data_path)
df = preprocess(df)

if "APPRAISED_VALUE" not in df.columns:
    st.error("APPRAISED_VALUE column not found in dataset.")
    st.stop()

model, feature_names = build_model(df)

st.subheader("Enter Feature Values")

user_input = {}
for col in feature_names:
    # Use median as a reasonable default
    default_val = float(df[col].median()) if col in df.columns else 0.0
    user_input[col] = st.number_input(col, value=default_val)

X_user = pd.DataFrame([user_input])

pred = model.predict(X_user)[0]
st.subheader("Predicted Appraised Value")
st.write(f"**${pred:,.0f}**")
