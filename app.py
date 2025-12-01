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
DATA_PATH = "https://raw.githubusercontent.com/yaoemma/financial-inclusion-app/refs/heads/main/Financial_inclusion_dataset.csv"
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
    # fallback: last column (common in some datasets)
    return df.columns[-1]

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str):
    df = pd.read_csv(path, low_memory=False)
    return df

# -------------------------
# PREPROCESS
# -------------------------
def build_preprocessor(X: pd.DataFrame):
    # identify numeric and categorical features
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Simple imputers + encoders
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

    # remove rows with missing target
    mask = y.notna()
    X = X.loc[mask, :].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    preprocessor, numeric_cols, cat_cols = build_preprocessor(X)

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y if y.nunique()>1 else None)

    clf.fit(X_train, y_train)

    # metrics
    y_pred = clf.predict(X_test)
    try:
        y_proba = clf.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = None

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True) if y_test.nunique() > 1 else None

    # save model and metadata
    meta = {
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "target_col": target_col
    }
    joblib.dump({"pipeline": clf, "meta": meta}, model_path)
    return {"accuracy": acc, "roc_auc": auc, "report": report, "model_path": model_path}

# -------------------------
# LOAD MODEL
# -------------------------
def load_model(path: str):
    if not Path(path).exists():
        return None
    data = joblib.load(path)
    return data

# -------------------------
# PREDICT SINGLE
# -------------------------
def predict_single(pipeline, input_df: pd.DataFrame):
    proba = None
    pred = pipeline.predict(input_df)
    try:
        proba = pipeline.predict_proba(input_df)[:,1]
    except Exception:
        proba = None
    return pred, proba

# -------------------------
# STREAMLIT UI
# -------------------------
def main():
    st.set_page_config(page_title="Inclusion financière — Demo", layout="wide")
    st.title("Projet : Inclusion financière en Afrique")
    st.markdown("Application Streamlit pour explorer le dataset, entraîner un modèle et réaliser des prédictions.")

    # Load data
    if not Path(DATA_PATH).exists():
        st.error(f"Le fichier de données n'a pas été trouvé à l'emplacement : `{DATA_PATH}`. Vérifie le chemin.")
        st.stop()
    df = load_data(DATA_PATH)

    st.sidebar.header("Actions")
    action = st.sidebar.selectbox("Choisis une action", [
        "Voir aperçu des données",
        "Profilage rapide (pandas profiling)",
        "Nettoyage & entraînement du modèle",
        "Charger modèle enregistré",
        "Interface de prédiction (formulaire)"
    ])

    # Automatic detection of target column
    target_col = detect_target(df)

    if action == "Voir aperçu des données":
        st.subheader("Aperçu du dataset")
        st.write(f"Nombre de lignes : {df.shape[0]}, nombre de colonnes : {df.shape[1]}")
        st.write("Colonnes :", df.columns.tolist())
        st.dataframe(df.head(200))

        with st.expander("Statistiques descriptives (numériques)"):
            st.write(df.describe(include=[np.number]).T)

        with st.expander("Statistiques descriptives (catégorielles)"):
            st.write(df.describe(include=["object", "category"]).T)

        st.markdown("**Détection automatique de la colonne cible (target)**")
        st.write(f"Colonne sélectionnée comme target : **{target_col}** (tu peux la changer dans le code si besoin)")

    elif action == "Profilage rapide (pandas profiling)":
        st.subheader("Profilage (pandas-profiling / ydata_profiling)")
        try:
            from ydata_profiling import ProfileReport
            profile = ProfileReport(df, minimal=True)
            st_profile = profile.to_html()
            st.components.v1.html(st_profile, height=800, scrolling=True)
        except Exception as e:
            st.warning("pandas-profiling (ydata_profiling) n'est pas installé ou a échoué. Voir instructions dans README. Erreur: " + str(e))

    elif action == "Nettoyage & entraînement du modèle":
        st.subheader("Nettoyage & Entraînement")
        st.markdown("On va traiter les valeurs manquantes, supprimer les doublons, et entraîner un RandomForest.")
        if st.button("Lancer l'entraînement"):
            # basic cleaning
            df_clean = df.copy()
            before = df_clean.shape[0]
            df_clean = df_clean.drop_duplicates().reset_index(drop=True)
            after = df_clean.shape[0]
            st.write(f"Suppression des doublons : {before-after} lignes supprimées.")

            # train
            results = train_and_save_model(df_clean, target_col, MODEL_PATH)
            st.success("Modèle entraîné et sauvegardé.")
            st.write("Metrics :")
            st.write(f"Accuracy (test) : {results['accuracy']}")
            if results["roc_auc"] is not None:
                st.write(f"ROC AUC (test) : {results['roc_auc']}")
            if results["report"] is not None:
                st.write(pd.DataFrame(results["report"]).transpose())

    elif action == "Charger modèle enregistré":
        st.subheader("Charger modèle")
        model_data = load_model(MODEL_PATH)
        if model_data is None:
            st.warning("Aucun modèle trouvé. Entraîne un modèle d'abord (Nettoyage & entraînement).")
        else:
            st.success(f"Modèle chargé depuis `{MODEL_PATH}`")
            st.write("Méta :")
            st.write(model_data["meta"])
            st.write("Tu peux maintenant aller dans 'Interface de prédiction (formulaire)' pour faire des prédictions.")

    elif action == "Interface de prédiction (formulaire)":
        st.subheader("Formulaire de prédiction")
        model_data = load_model(MODEL_PATH)
        if model_data is None:
            st.warning("Aucun modèle sauvegardé trouvé. Entraîne le modèle d'abord via 'Nettoyage & entraînement du modèle'.")
            st.stop()

        pipeline = model_data["pipeline"]
        meta = model_data["meta"]
        numeric_cols = meta.get("numeric_cols", [])
        cat_cols = meta.get("cat_cols", [])

        st.markdown("Remplis les champs ci-dessous pour faire une prédiction :")
        with st.form("predict_form"):
            input_data = {}
            # For numeric columns
            for col in numeric_cols:
                # determine a sensible default from dataset if possible
                default = float(df[col].median()) if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) else 0.0
                input_data[col] = st.number_input(f"{col} (numérique)", value=float(default))

            # For categorical columns
            for col in cat_cols:
                if col in df.columns:
                    opts = df[col].dropna().unique().tolist()
                    if len(opts) > 0 and len(opts) <= 50:
                        choice = st.selectbox(f"{col} (catégorie)", options=opts)
                    else:
                        # too many unique values: free text
                        choice = st.text_input(f"{col} (catégorie - libre)")
                else:
                    choice = st.text_input(f"{col} (catégorie - libre)")
                input_data[col] = choice

            submitted = st.form_submit_button("Prédire")
        if submitted:
            input_df = pd.DataFrame([input_data])
            pred, proba = predict_single(pipeline, input_df)
            st.write("Résultat de la prédiction :")
            st.write(f"Classe prédite : **{pred[0]}**")
            if proba is not None:
                st.write(f"Probabilité (classe positive) : **{proba[0]:.4f}**")
            else:
                st.write("Probabilité indisponible pour ce modèle (predict_proba non supporté).")

    # Footer
    st.markdown("---")
    st.caption("App générée — adapte les colonnes cibles / chemin dataset selon ton jeu de données réel.")

if __name__ == "__main__":
    main()
