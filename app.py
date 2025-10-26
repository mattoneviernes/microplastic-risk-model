# app.py
# Predictive Risk Modeling for Microplastic Pollution using Data Mining Techniques
# Streamlit application for:
# - loading CSV datasets
# - preprocessing (missing values, encoding, scaling)
# - training & evaluating classification models (RandomForest, LogisticRegression, DecisionTree)
# - 10-fold cross validation with Accuracy, Precision, Recall, F1
# - visualizations: risk distribution, confusion matrix, feature importance
# - predicting risk levels (Low / Medium / High) and exporting results
#
# Author: Assistant (adapted for mattoneviernes thesis)
# Date: 2025-10-26

import io
import math
import base64
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate, train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

st.set_page_config(page_title="Microplastic Risk Model", layout="wide")
st.title("Predictive Risk Modeling for Microplastic Pollution")
st.caption("Thesis: Predictive Risk Modeling for Microplastic Pollution using Data Mining Techniques")

# -------------------------
# Helper functions
# -------------------------
def read_csv(uploaded_file):
    """Read uploaded CSV file -> DataFrame. Handles bytes and file-like objects."""
    try:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin1")


def discretize_target_by_quantiles(series, bins=3, labels=None):
    """Discretize numeric target into bins (default 3 quantile bins)."""
    if labels is None:
        labels = ["Low", "Medium", "High"][:bins]
    # Use qcut (quantiles) but fall back to cut if duplicates
    try:
        return pd.qcut(series, q=bins, labels=labels, duplicates="drop")
    except Exception:
        return pd.cut(series, bins=bins, labels=labels, duplicates="drop")


def get_feature_names_from_preprocessor(preprocessor, numeric_features, categorical_features):
    """Return list of feature names after ColumnTransformer preprocessing."""
    feature_names = []
    # Numeric features remain as-is after numeric transformer
    if len(numeric_features) > 0:
        feature_names.extend(numeric_features)
    # Categorical features will be one-hot encoded
    if len(categorical_features) > 0:
        # fetch the OneHotEncoder inside the categorical transformer
        try:
            ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            cat_names = ohe.get_feature_names_out(categorical_features).tolist()
            feature_names.extend(cat_names)
        except Exception:
            # fallback: use raw categorical names
            feature_names.extend(categorical_features)
    return feature_names


def build_preprocessor(X, target_col=None):
    """Create a ColumnTransformer which imputes & scales numeric data and imputes & encodes categorical data."""
    # Exclude target column from features selection if present
    drop_cols = []
    if target_col is not None:
        drop_cols = [target_col]

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [c for c in numeric_features if c not in drop_cols]
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_features = [c for c in categorical_features if c not in drop_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_features, categorical_features


def summarize_cv_results(cv_res):
    """Return a tidy DataFrame summarizing cross_validate results (mean +/- std)."""
    metrics = {}
    for key, vals in cv_res.items():
        # skip fit_time and score_time for summary table
        if key in ("fit_time", "score_time"):
            continue
        metrics[key] = (np.mean(vals), np.std(vals))
    # Build df
    rows = []
    for k, (mean_v, std_v) in metrics.items():
        rows.append({"metric": k, "mean": mean_v, "std": std_v})
    return pd.DataFrame(rows)


def plot_feature_importance(importances, feature_names, top_n=20):
    """Barplot for feature importances."""
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(8, min(6, len(fi) * 0.25 + 1)))
    sns.barplot(x=fi.values, y=fi.index, palette="viridis")
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()


def plot_risk_distribution(y_series):
    """Plot distribution of risk categories."""
    plt.figure(figsize=(6, 4))
    counts = y_series.value_counts().sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values, palette="rocket")
    plt.title("Risk Distribution")
    plt.xlabel("Risk Level")
    plt.ylabel("Count")
    for i, v in enumerate(counts.values):
        plt.text(i, v + max(1, v * 0.01), str(int(v)), ha="center")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()


def plot_confusion_matrix(y_true, y_pred, labels):
    """Heatmap of confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()


def allow_download_df(df, filename="predictions.csv"):
    """Provide a download button for a dataframe (CSV)."""
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label="Download predictions (CSV)", data=csv, file_name=filename, mime="text/csv")


# -------------------------
# UI: Sidebar - Upload & settings
# -------------------------
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload dataset (CSV)", type=["csv"], help="Upload CSV file with microplastic data")
if uploaded_file is None:
    st.sidebar.info("Please upload a CSV file to begin.")
    st.stop()

try:
    df_raw = read_csv(uploaded_file)
except Exception as e:
    st.sidebar.error(f"Unable to read file: {e}")
    st.stop()

st.sidebar.success("File loaded")
st.write("Data preview (first 5 rows):")
st.dataframe(df_raw.head())

# -------------------------
# UI: Target selection / discretization
# -------------------------
st.sidebar.subheader("Target (risk) settings")
columns = df_raw.columns.tolist()
target_col = st.sidebar.selectbox("Select target column (existing risk label or numeric measure)", options=[None] + columns, index=0)

discretize_numeric = False
bins = 3
if target_col is None:
    st.sidebar.warning("Select a target column so the app can train models.")
else:
    # Determine if target is numeric
    if pd.api.types.is_numeric_dtype(df_raw[target_col]):
        discretize_numeric = st.sidebar.checkbox(
            "Discretize numeric target into risk levels (Low, Medium, High) using quantiles?", value=True
        )
        if discretize_numeric:
            bins = st.sidebar.radio("Number of bins", options=[3], index=0)  # fix to 3 as requested

# Choose model
st.sidebar.subheader("Model selection")
model_name = st.sidebar.selectbox("Choose classifier", options=["Random Forest", "Logistic Regression", "Decision Tree"])

# Cross validation folds
st.sidebar.subheader("Cross-validation")
requested_folds = st.sidebar.number_input("Number of folds (suggested 10)", min_value=2, max_value=50, value=10, step=1)

# Random seed
random_state = st.sidebar.number_input("Random state", min_value=0, value=42, step=1)

# Button to start training
train_btn = st.sidebar.button("Train & Evaluate")

# -------------------------
# Main: Preprocessing & Training
# -------------------------
if train_btn:
    # Basic checks
    if target_col is None:
        st.error("Please select a target column in the sidebar before training.")
        st.stop()

    df = df_raw.copy()

    # Drop rows where entire row is empty
    df.dropna(how="all", inplace=True)

    # If numeric and user wants discretization -> create labels
    y_raw = df[target_col]
    if pd.api.types.is_numeric_dtype(y_raw) and discretize_numeric:
        y = discretize_target_by_quantiles(y_raw, bins=bins, labels=["Low", "Medium", "High"])
        y = y.astype(str)  # qcut can return categorical; ensure string
    else:
        # Treat as categorical: fillna with 'unknown' and convert to str
        y = y_raw.fillna("unknown").astype(str)

    # Show class counts and ensure we have at least 2 classes
    class_counts = y.value_counts()
    st.subheader("Target class distribution")
    st.write(class_counts)

    if class_counts.shape[0] < 2:
        st.error("Need at least 2 classes in the target to train a classifier.")
        st.stop()

    # Prepare X (drop target)
    X = df.drop(columns=[target_col])

    # Build preprocessor
    preprocessor, numeric_features, categorical_features = build_preprocessor(X, target_col=None)

    # Fit preprocessor to get feature names (safe operation)
    try:
        preprocessor.fit(X)
        feature_names = get_feature_names_from_preprocessor(preprocessor, numeric_features, categorical_features)
    except Exception as e:
        st.error(f"Failed to fit preprocessor: {e}")
        st.stop()

    # Select estimator
    if model_name == "Random Forest":
        estimator = RandomForestClassifier(n_estimators=200, random_state=int(random_state))
    elif model_name == "Logistic Regression":
        estimator = LogisticRegression(max_iter=1000, random_state=int(random_state))
    else:
        estimator = DecisionTreeClassifier(random_state=int(random_state))

    # Create pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", estimator)])

    # Cross-validation setup: use StratifiedKFold when possible
    n_samples = X.shape[0]
    folds = int(requested_folds)
    if folds > n_samples:
        st.warning(f"Requested {folds} folds but only {n_samples} samples available. Reducing folds to {max(2, n_samples)}.")
        folds = max(2, n_samples)

    # Check class minimum counts for stratification
    min_class_count = int(class_counts.min())
    if min_class_count < 2:
        st.warning(
            f"Smallest class has {min_class_count} samples; StratifiedKFold with many folds may be invalid. "
            "The code will use StratifiedKFold if possible and reduce folds when required."
        )

    # Ensure folds do not exceed min_class_count
    if folds > min_class_count:
        safe_folds = max(2, min_class_count)
        if safe_folds < folds:
            st.info(f"Reducing folds from {folds} to {safe_folds} to satisfy stratification constraints.")
            folds = safe_folds

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(random_state))

    # Cross-validate with multiple scoring metrics
    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    st.info(f"Running cross-validation with {folds} folds. This may take a few moments.")

    try:
        cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
    except ValueError as e:
        st.error(f"Cross-validation failed: {e}")
        st.stop()

    # Summarize results
    summary_df = summarize_cv_results(cv_results)
    # Rename metrics for readability
    # display in a wide layout
    st.subheader("Cross-Validation results (mean ± std)")
    # Convert metric keys to friendly names and present percent where appropriate
    display_rows = []
    for _, row in summary_df.iterrows():
        metric_name = row["metric"]
        mean_v = row["mean"]
        std_v = row["std"]
        display_rows.append(
            {
                "Metric": metric_name.replace("test_", ""),
                "Mean": f"{mean_v:.3f}",
                "Std": f"{std_v:.3f}",
            }
        )
    st.table(pd.DataFrame(display_rows))

    # Fit final model on train/test split for extra evaluation and confusion matrix
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=int(random_state))
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=int(random_state))

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    st.subheader("Hold-out evaluation (20% test split)")
    st.write("Classification report (on test split):")
    st.text(classification_report(y_test, y_pred, zero_division=0))
    labels = sorted(list(y.unique()))
    plot_confusion_matrix(y_test, y_pred, labels=labels)

    # Feature importance visualization (if available)
    st.subheader("Feature importance / coefficients")
    try:
        clf = pipeline.named_steps["classifier"]
        # For tree-based: feature_importances_
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            plot_feature_importance(importances, feature_names, top_n=30)
        # For linear models: coefficients
        elif hasattr(clf, "coef_"):
            coef = np.squeeze(clf.coef_)
            # handle multiclass (coef_ shape: n_classes x n_features)
            if coef.ndim == 2:
                # For multiclass, take absolute mean across classes
                coef_abs_mean = np.mean(np.abs(coef), axis=0)
                plot_feature_importance(coef_abs_mean, feature_names, top_n=30)
            else:
                plot_feature_importance(np.abs(coef), feature_names, top_n=30)
        else:
            st.info("Selected model does not provide feature importances/coefficients.")
    except Exception as e:
        st.warning(f"Unable to compute feature importance: {e}")

    # Risk distribution plot on entire labeled data
    st.subheader("Risk distribution (full labeled data)")
    plot_risk_distribution(y)

    # -------------------------
    # Prediction for new/unlabeled data
    # -------------------------
    st.subheader("Predict risk for new data")
    st.write("You can upload a CSV of new samples (same feature columns) to predict risk levels using the trained model.")

    pred_file = st.file_uploader("Upload CSV for prediction (optional)", type=["csv"], key="pred_uploader")
    if pred_file is not None:
        try:
            df_new = read_csv(pred_file)
            st.write("Preview of new data:")
            st.dataframe(df_new.head())

            # Ensure columns match the training feature set (X columns)
            missing_cols = [c for c in X.columns if c not in df_new.columns]
            if missing_cols:
                st.warning(f"New data is missing {len(missing_cols)} columns required for prediction. Missing columns: {missing_cols[:10]}")
                # Try to add missing columns with NaN so preprocessing pipeline can impute
                for c in missing_cols:
                    df_new[c] = np.nan

            # Keep only training feature columns (order matters)
            df_new = df_new[X.columns.tolist()]

            # Make predictions
            preds = pipeline.predict(df_new)
            # If classifier has predict_proba, include max probability
            proba = None
            if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
                proba_arr = pipeline.predict_proba(df_new)
                proba = np.max(proba_arr, axis=1)

            df_new_out = df_new.copy()
            df_new_out["predicted_risk"] = preds
            if proba is not None:
                df_new_out["predicted_prob"] = proba

            st.success("Predictions complete")
            st.dataframe(df_new_out.head(200))
            allow_download_df(df_new_out, filename="predictions.csv")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.info("Upload new samples to get predictions, or use the dataset above after removing the target column.")

    st.success("Training and evaluation complete. Inspect the results, visualizations, and download predictions if needed.")
else:
    st.info("Configure settings in the sidebar and click 'Train & Evaluate' to run modeling.")
