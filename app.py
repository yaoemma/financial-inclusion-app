# app.py
import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# -------------------------
# CONFIG
# -------------------------
DATA_PATH = "https://raw.githubusercontent.com/yaoemma/financial-inclusion-app/main/data/Financial_inclusion_dataset.csv"

MODEL_PATH = "financial_inclusion_model.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_CANDIDATES = [
    "bank_account",
    "has_bank_account",
    "bankaccount",
    "has_account",
    "HasBankAccount",
    "BankAccount",
    "bank_account_flag",
    "bank"
]

# -------------------------
# UTIL : find target column
# -------------------------
def detect_target(df: pd.DataFrame):
    for cand in TARGET_CANDIDATES:
        if cand in df.columns:
            return cand
    return df.columns[-1]  # fallback


# -------------------------
# LOAD DATA (FIXED)
# -------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str):
    import requests, io
    response = requests.get(path)
    response.raise_for_status()   # erreur si URL morte
    content = response.content
    df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    return df


# -------------------------
# PREPROCESS
# -------------------------
def build_preprocessor(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, cat_cols)
        ],
        remainder="drop"
    )
    return preprocessor, numeric_cols, cat_cols


# -------------------------
# TRAIN / SAVE MODEL
# -------------------------
def train_and_save_model(df: pd.DataFrame, target_col: str, model_path: str):
    st.info("Entraînement du modèle — ceci peut prendre quelques secondes...")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    mask = y.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    preprocessor, numeric_cols, cat_cols = build_preprocessor(X)

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=y if y.nunique() > 1 else None
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except:
        auc = None

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True) if y.nunique() > 1 else None

    joblib.dump({
        "pipeline": clf,
        "meta": {
            "numeric_cols": numeric_cols,
            "cat_cols": cat_cols,
            "target_col": target_col
        }
    }, model_path)

    return {"accuracy": acc, "roc_auc": auc, "report": report}


# -------------------------
# LOAD MODEL
# -------------------------
def load_model(path: str):
    if not Path(path).exists():
        return None
    return joblib.load(path)


# -------------------------
# PREDICT SINGLE
# -------------------------
def predict_single(pipeline, input_df: pd.DataFrame):
    pred = pipeline.predict(input_df)
    try:
        proba = pipeline.predict_proba(input_df)[:, 1]
    except:
        proba = None
    return pred, proba


# -------------------------
# STREAMLIT UI
# -------------------------
def main():
    st.set_page_config(page_title="Inclusion financière — Demo", layout="wide")
    st.title("Projet : Inclusion financière en Afrique")

    st.markdown("Application Streamlit pour explorer le dataset, entraîner un modèle et réaliser des prédictions.")

    # -----------------------------
    # FIX: charger le dataset sans Path()
    # -----------------------------
    try:
        df = load_data(DATA_PATH)
    except Exception as e:
        st.error(f"Impossible de charger le dataset depuis : {DATA_PATH}\n\nErreur : {e}")
        st.stop()

    st.sidebar.header("Actions")
    action = st.sidebar.selectbox("Choisis une action", [
        "Voir aperçu des données",
        "Profilage rapide (pandas profiling)",
        "Nettoyage & entraînement du modèle",
        "Charger modèle enregistré",
        "Interface de prédiction (formulaire)"
    ])

    target_col = detect_target(df)

    if action == "Voir aperçu des données":
        st.subheader("Aperçu du dataset")
        st.write(f"{df.shape[0]} lignes — {df.shape[1]} colonnes")
        st.write(df.head())

    elif action == "Profilage rapide (pandas profiling)":
        st.subheader("Profilage du dataset")
        try:
            from ydata_profiling import ProfileReport
            profile = ProfileReport(df, minimal=True)
            st.components.v1.html(profile.to_html(), height=800, scrolling=True)
        except Exception as e:
            st.warning("Profilage non disponible. Erreur : " + str(e))

    elif action == "Nettoyage & entraînement du modèle":
        st.subheader("Nettoyage & Entraînement")
        if st.button("Lancer l'entraînement"):
            df_clean = df.drop_duplicates().reset_index(drop=True)
            results = train_and_save_model(df_clean, target_col, MODEL_PATH)
            st.success("Modèle entraîné")
            st.write(results)

    elif action == "Charger modèle enregistré":
        model_data = load_model(MODEL_PATH)
        if model_data is None:
            st.warning("Aucun modèle trouvé.")
        else:
            st.success("Modèle chargé !")
            st.write(model_data["meta"])

    elif action == "Interface de prédiction (formulaire)":
        model_data = load_model(MODEL_PATH)
        if model_data is None:
            st.warning("Aucun modèle trouvé.")
            st.stop()

        pipeline = model_data["pipeline"]
        meta = model_data["meta"]
        numeric_cols = meta["numeric_cols"]
        cat_cols = meta["cat_cols"]

        with st.form("predict_form"):
            inputs = {}

            for col in numeric_cols:
                default = float(df[col].median())
                inputs[col] = st.number_input(col, value=default)

            for col in cat_cols:
                choices = df[col].dropna().unique().tolist()
                if len(choices) <= 50:
                    inputs[col] = st.selectbox(col, choices)
                else:
                    inputs[col] = st.text_input(col)

            submitted = st.form_submit_button("Prédire")

        if submitted:
            input_df = pd.DataFrame([inputs])
            pred, proba = predict_single(pipeline, input_df)
            st.success(f"Prédiction : {pred[0]}")
            if proba is not None:
                st.write(f"Probabilité : {proba[0]:.4f}")

    st.markdown("---")
    st.caption("Application Gomycode — Inclusion financière.")


if __name__ == "__main__":
    main()
