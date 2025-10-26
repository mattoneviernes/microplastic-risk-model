# ==============================================================
# Predictive Risk Modeling for Microplastic Pollution
# Updated by: Assistant for mattoneviernes
# Objectives:
# 1. Preprocess the data
# 2. Design and implement model architecture using classification and clustering
# 3. Validate the model architecture
# 4. Cross-validate using K-Fold
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)

st.set_page_config(page_title="Microplastic Pollution Risk Model", layout="wide")
st.title("🌊 Predictive Risk Modeling for Microplastic Pollution")
st.caption("Developed by: Magdaluyo & Viernes | Agusan del Sur State College of Agriculture and Technology")

st.markdown(
    """
This web app automates:
- Data preprocessing (missing values, encoding, scaling)
- Unsupervised learning (Clustering)
- Supervised learning (Classification & Regression)
- Model validation and K-Fold cross-validation
"""
)

# -------------------------
# Helpers
# -------------------------
def safe_read_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin1")


def describe_df(df):
    st.write("Shape:", df.shape)
    st.write("Columns and dtypes:")
    st.dataframe(pd.DataFrame(df.dtypes, columns=["dtype"]).T)


# -------------------------
# STEP 1: Upload
# -------------------------
st.header("STEP 1: Upload Dataset")
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if not uploaded_file:
    st.warning("📂 Please upload your dataset to start preprocessing and modeling.")
    st.stop()

df = safe_read_csv(uploaded_file)
st.success("✅ Dataset uploaded successfully.")
st.subheader("Preview")
st.dataframe(df.head())

# -------------------------
# STEP 2: Preprocessing
# -------------------------
st.header("STEP 2: Data Cleaning and Preprocessing")
st.write(f"Initial records: {len(df)}")

# Basic cleaning
df = df.copy()
df.drop_duplicates(inplace=True)
st.write(f"After removing duplicates: {len(df)}")

st.subheader("Missing values per column")
missing = df.isnull().sum()
st.dataframe(missing[missing > 0].sort_values(ascending=False))

# Fill missing values
# Numeric -> median, Categorical -> mode
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

if len(num_cols) > 0:
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

for c in cat_cols:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].mode().iloc[0])

st.success("✅ Missing values handled.")

st.subheader("Automatic type hints")
describe_df(df)

# Encode categorical columns for modeling (features)
# For tree models, one-hot or pd.get_dummies is fine; keep label encoders for target if needed.
st.write("Encoding categorical feature columns using pd.get_dummies (one-hot).")
df_encoded = pd.get_dummies(df, drop_first=True)
st.write(f"Encoded shape: {df_encoded.shape}")
st.dataframe(df_encoded.head())

# Keep original df for target selection if user wants original column
original_df = df.copy()

# -------------------------
# STEP 3: Clustering (Unsupervised)
# -------------------------
st.header("STEP 3: Clustering Analysis (Unsupervised Learning)")

with st.expander("Clustering settings and results", expanded=True):
    if df_encoded.shape[0] < 2 or df_encoded.shape[1] < 2:
        st.warning("Not enough data/features to perform clustering.")
    else:
        # Let user select features to use for clustering (defaults to numeric features)
        clustering_features = st.multiselect(
            "Select features for clustering (at least 2)", options=df_encoded.columns.tolist(), default=df_encoded.select_dtypes(include=[np.number]).columns.tolist()[:4]
        )

        if len(clustering_features) >= 2:
            # Scaling helps KMeans
            scaler = StandardScaler()
            X_cluster = scaler.fit_transform(df_encoded[clustering_features])

            n_clusters = st.slider("Number of clusters (k)", min_value=2, max_value=10, value=3)
            kmeans = KMeans(n_clusters=int(n_clusters), random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_cluster)
            df["Cluster"] = cluster_labels

            st.success(f"✅ K-Means clustering complete ({n_clusters} clusters).")
            st.write("Cluster counts:")
            st.write(df["Cluster"].value_counts().sort_index())

            # silhouette score (requires >1 cluster and >1 sample per cluster)
            try:
                sil = silhouette_score(X_cluster, cluster_labels)
                st.write(f"Silhouette Score: {sil:.3f}")
            except Exception:
                st.write("Silhouette Score: Not available for this configuration.")

            # Scatter plot of first two features
            plt.figure(figsize=(7, 5))
            sns.scatterplot(
                x=df_encoded[clustering_features[0]],
                y=df_encoded[clustering_features[1]],
                hue=cluster_labels,
                palette="Set2",
                legend="full",
            )
            plt.title("KMeans Clustering")
            plt.xlabel(clustering_features[0])
            plt.ylabel(clustering_features[1])
            st.pyplot(plt.gcf())
            plt.clf()
        else:
            st.info("Select at least 2 features to run clustering.")

# -------------------------
# STEP 4: Supervised Modeling (Classification / Regression)
# -------------------------
st.header("STEP 4: Supervised Modeling (Classification & Regression)")

st.write("Select a target column for supervised learning. The app will detect whether it looks like a classification target (categorical / small number of unique values) or a regression target (continuous).")
target_col = st.selectbox("Select Target Column", options=original_df.columns.tolist())

if target_col is None:
    st.warning("Select a target column to continue.")
    st.stop()

# Determine task type heuristically
y_series = original_df[target_col]
unique_vals = y_series.nunique(dropna=True)
is_numeric_target = pd.api.types.is_numeric_dtype(y_series)

# Heuristic: treat as classification if non-numeric OR numeric with small unique counts (<20)
if (not is_numeric_target) or (is_numeric_target and unique_vals <= 20):
    task = "classification"
else:
    task = "regression"

st.write(f"Detected task type: {task}")

# Prepare X and y
# Use df_encoded for features (one-hot encoding applied)
features = [c for c in df_encoded.columns if c != target_col and not c.startswith(f"{target_col}_")]
X = df_encoded.drop(columns=[c for c in df_encoded.columns if c == target_col or c.startswith(f"{target_col}_")], errors="ignore")

# For y, if original target is categorical, encode it; if numeric then use original numeric
label_encoder = None
if task == "classification":
    if not is_numeric_target:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(original_df[target_col].astype(str))
    else:
        # If numeric but small unique values, treat as categorical
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(original_df[target_col].astype(str))
else:
    # regression
    y = original_df[target_col].astype(float).values

st.write(f"Feature matrix shape: {X.shape} | Target shape: {y.shape}")

# Optionally scale numeric features for regression / clustering; RandomForest doesn't require it but LinearRegression does.
scaler_for_reg = StandardScaler()
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

# Model training and validation
if task == "classification":
    st.subheader("Classification: Random Forest (default)")

    test_size = st.slider("Test set proportion", 0.05, 0.5, 0.2, step=0.05)
    stratify_flag = True if unique_vals > 1 else False

    try:
        if stratify_flag:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        clf = RandomForestClassifier(random_state=42, n_estimators=200)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.success(f"✅ Model trained. Test Accuracy: {acc*100:.2f}%")

        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred, zero_division=0))

        cm = confusion_matrix(y_test, y_pred)
        st.subheader("Confusion Matrix")
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(plt.gcf())
        plt.clf()

        # Feature importances
        try:
            fi = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
            st.subheader("Top feature importances")
            st.write(fi)
            plt.figure(figsize=(8, 4))
            sns.barplot(x=fi.values, y=fi.index)
            plt.title("Feature Importances (Top 20)")
            st.pyplot(plt.gcf())
            plt.clf()
        except Exception:
            pass

        # Cross-validation (StratifiedKFold)
        folds = st.slider("K-Folds for cross-validation", 2, 10, 5)
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        st.write(f"Cross-Validation Accuracy scores: {np.round(cv_scores, 3)}")
        st.success(f"Average CV accuracy: {np.mean(cv_scores)*100:.2f}% (+/- {np.std(cv_scores)*100:.2f}%)")

        # Plot CV scores
        plt.figure(figsize=(6, 4))
        sns.lineplot(x=range(1, len(cv_scores) + 1), y=cv_scores, marker="o")
        plt.title("Cross-Validation Accuracy per Fold")
        plt.xlabel("Fold")
        plt.ylabel("Accuracy")
        st.pyplot(plt.gcf())
        plt.clf()
    except Exception as e:
        st.error(f"Error during classification training: {e}")

else:
    st.subheader("Regression: Linear Regression (default)")

    # Prepare X (scale numeric for regression)
    X_reg = X.copy()
    if len(numeric_features) > 0:
        X_reg[numeric_features] = scaler_for_reg.fit_transform(X_reg[numeric_features])

    test_size = st.slider("Test set proportion", 0.05, 0.5, 0.2, step=0.05)

    try:
        X_train, X_test, y_train, y_test = train_test_split(X_reg, y, test_size=test_size, random_state=42)

        reg_choice = st.selectbox("Regression algorithm", options=["LinearRegression", "RandomForestRegressor"])

        if reg_choice == "LinearRegression":
            model = LinearRegression()
        else:
            model = RandomForestRegressor(n_estimators=200, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        st.success("✅ Regression model trained.")
        st.write(f"MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.3f}")

        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=y_test, y=y_pred)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted")
        st.pyplot(plt.gcf())
        plt.clf()

        # Cross-validation (KFold)
        folds = st.slider("K-Folds for cross-validation (regression)", 2, 10, 5)
        cv = KFold(n_splits=folds, shuffle=True, random_state=42)
        # Use R² as scoring for regression and also show MAE via neg_mean_absolute_error
        cv_scores_r2 = cross_val_score(model, X_reg, y, cv=cv, scoring="r2")
        cv_scores_mae = cross_val_score(model, X_reg, y, cv=cv, scoring="neg_mean_absolute_error")
        st.write(f"CV R² scores: {np.round(cv_scores_r2, 3)}")
        st.write(f"CV MAE scores: {np.round(-cv_scores_mae, 3)}")
        st.success(f"Average CV R²: {np.mean(cv_scores_r2):.3f} | Average CV MAE: {-np.mean(cv_scores_mae):.3f}")
    except Exception as e:
        st.error(f"Error during regression training: {e}")

# -------------------------
# STEP 5: Model validation summary
# -------------------------
st.header("STEP 5: Model Validation Summary")
st.info(
    """
    This app performed:
    - Data preprocessing: duplicated rows removed, missing values imputed (median for numeric, mode for categorical), categorical features one-hot encoded.
    - Clustering: KMeans with user-selectable features and number of clusters, silhouette score shown when available.
    - Classification: RandomForest classifier with test-set evaluation, confusion matrix, feature importances and Stratified K-Fold cross-validation.
    - Regression: LinearRegression or RandomForestRegressor with test-set evaluation and K-Fold cross-validation (R² and MAE).
    """
)

st.caption("© 2025 Magdaluyo & Viernes | Predictive Risk Modeling for Microplastic Pollution")
