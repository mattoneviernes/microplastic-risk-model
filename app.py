# ==============================================================
# Predictive Risk Modeling for Microplastic Pollution
# Updated by: Assistant for mattoneviernes
# Objectives (updates):
# - Sidebar upload that accepts CSV and Excel files
# - Robust file parsing and sheet selection for Excel
# - Thorough error handling and fixes for earlier edge-cases
# - Handle stratification / cross-validation edge-cases when classes are rare
# ==============================================================

import re
import math
import io
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
- Robust data preprocessing (cleaning, normalization, unit parsing)
- Standardizing microplastic taxonomy (types, polymers, shapes, colors)
- Derived features (size bins, log concentration, density flags)
- Quality checks (outliers, unrealistic values)
- Unsupervised learning (Clustering)
- Supervised learning (Classification & Regression)
"""
)

# -------------------------
# Helpers: File reading, Cleaning & Standardization
# -------------------------
def safe_read_file(uploaded_file, sheet_name=0):
    """Read CSV or Excel from Streamlit uploader. Returns DataFrame or raises."""
    try:
        name = getattr(uploaded_file, "name", "")
        # Work with BytesIO so pandas can read multiple times if needed
        uploaded_file.seek(0)
        content = uploaded_file.read()
        uploaded_file.seek(0)
        if name.lower().endswith((".xls", ".xlsx")):
            # read_excel supports bytes buffer
            return pd.read_excel(io.BytesIO(content), sheet_name=sheet_name)
        else:
            # try default CSV reading, try latin1 if default fails
            try:
                return pd.read_csv(io.BytesIO(content))
            except Exception:
                uploaded_file.seek(0)
                return pd.read_csv(io.BytesIO(content), encoding="latin1")
    except Exception as e:
        raise RuntimeError(f"Unable to read uploaded file: {e}")


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: lower-case, strip, replace spaces/specials with underscores."""
    df = df.copy()
    new_cols = {}
    for c in df.columns:
        nc = str(c).strip().lower()
        nc = re.sub(r"[^\w\s]", "", nc)  # remove punctuation
        nc = re.sub(r"\s+", "_", nc)
        new_cols[c] = nc
    df.rename(columns=new_cols, inplace=True)
    return df


def find_column(df: pd.DataFrame, keywords):
    """Return the first column name that matches any of the keywords in its name (case-insensitive)."""
    cols = df.columns.tolist()
    for kw in keywords:
        for c in cols:
            if kw.lower() in c.lower():
                return c
    return None


def extract_numeric(value):
    """Extract numeric value from string that may contain units or separators."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        try:
            return float(value)
        except Exception:
            return np.nan
    s = str(value).strip()
    # Remove commas used as thousands separator but keep decimal points
    s_clean = s.replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s_clean)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return np.nan
    return np.nan


def standardize_microplastic_type(val):
    """Normalize microplastic type strings to a small controlled vocabulary."""
    if pd.isna(val):
        return np.nan
    v = str(val).strip().lower()
    mappings = {
        "fiber": ["fiber", "fibre", "filament", "fibres", "fibers"],
        "fragment": ["fragment", "flake", "fragments", "flakes", "particle", "fragmented"],
        "film": ["film", "sheet", "film/sheet"],
        "bead": ["bead", "sphere", "spherical", "microbead", "beads"],
        "pellet": ["pellet", "nurdle", "pellets", "nurdles"],
        "foam": ["foam"],
        "granule": ["granule", "granular"],
        "unknown": ["unknown", "n/a", "na", "unspecified", ""],
    }
    for std, variants in mappings.items():
        for variant in variants:
            if variant in v:
                return std
    # fallback: keep first token as 'other' or 'other' label
    token = re.sub(r"[^a-z0-9]", "", v)
    if token in ["fiber", "fragment", "film", "bead", "pellet", "foam", "granule"]:
        return token
    return "other"


def standardize_polymer(val):
    """Normalize polymer names to canonical forms."""
    if pd.isna(val):
        return np.nan
    v = str(val).strip().lower()
    if re.search(r"\bpe\b|polyethylene", v):
        return "polyethylene (PE)"
    if re.search(r"\bpp\b|polypropylene", v):
        return "polypropylene (PP)"
    if re.search(r"\bps\b|polystyrene", v):
        return "polystyrene (PS)"
    if re.search(r"\bpvc\b|polyvinyl", v):
        return "polyvinyl chloride (PVC)"
    if re.search(r"\bpet\b|polyethylene terephthalate", v):
        return "polyethylene terephthalate (PET)"
    if re.search(r"\bnylon\b", v):
        return "nylon"
    if v in ["unknown", "n/a", "", "na", "unspecified"]:
        return "unknown"
    return v[:50]


def limit_cardinality_ser(s: pd.Series, top_n=50):
    """Keep top_n most frequent categories; replace others with 'other'."""
    if s.dtype.name == "category" or s.dtype == object:
        top = s.value_counts().nlargest(top_n).index
        return s.where(s.isin(top), other="other")
    return s


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features commonly useful for microplastic models."""
    df = df.copy()

    # Standard size parsing: detect columns that look like 'size', 'diameter', 'length'
    size_col = find_column(df, ["size", "diameter", "length"])
    if size_col:
        df[f"{size_col}_num"] = df[size_col].apply(extract_numeric)
        # Heuristic: if the median value of detected size seems large (>100), interpret as micrometers (µm)
        median_size = df[f"{size_col}_num"].median(skipna=True)
        if pd.notna(median_size) and median_size > 100:
            # convert µm -> mm for a normalized size_mm
            df["particle_size_mm"] = df[f"{size_col}_num"].apply(lambda x: x / 1000.0 if pd.notna(x) else np.nan)
        else:
            # assume values are already in mm (or small µm)
            df["particle_size_mm"] = df[f"{size_col}_num"]

        # Create size categories based on µm scale (use original numeric when available)
        def size_bucket(x):
            if pd.isna(x):
                return "unknown"
            # Interpret x as µm if median_size > 100 else interpret as mm -> convert to µm
            if pd.notna(median_size) and median_size > 100:
                um = x
            else:
                # x interpreted as mm -> mm * 1000 to get µm
                um = x * 1000 if pd.notna(x) else np.nan
            if pd.isna(um):
                return "unknown"
            if um > 1000:
                return "macro_>1000um"
            if um > 100:
                return "micro_100-1000um"
            if um > 1:
                return "micro_1-100um"
            return "nano_<1um"

        df["size_category"] = df[f"{size_col}_num"].apply(size_bucket)
    else:
        df["size_category"] = "unknown"

    # Concentration parsing: detect 'concentration' or 'conc' columns
    conc_col = find_column(df, ["concentration", "conc", "count", "abundance", "particles"])
    if conc_col:
        df[f"{conc_col}_num"] = df[conc_col].apply(extract_numeric)
        # Log transform (add small constant)
        df[f"{conc_col}_log1p"] = df[f"{conc_col}_num"].apply(lambda x: np.log1p(x) if pd.notna(x) else x)

    # Derived density or per-area features if area/volume columns exist
    area_col = find_column(df, ["area", "surface", "m2", "m^2", "m2"])
    if area_col and conc_col:
        try:
            df["conc_per_area"] = df[f"{conc_col}_num"] / df[area_col].apply(extract_numeric)
        except Exception:
            df["conc_per_area"] = np.nan

    # Flags
    df["has_size"] = int(1) if any(col.endswith("_num") and "size" in col for col in df.columns) and df[[c for c in df.columns if c.endswith("_num") and "size" in c]].notna().any(axis=1).any() else 0
    # Per-row flags for presence of size/conc
    if size_col and f"{size_col}_num" in df.columns:
        df["has_size"] = df[f"{size_col}_num"].notna().astype(int)
    else:
        df["has_size"] = 0

    if conc_col and f"{conc_col}_num" in df.columns:
        df["has_conc"] = df[f"{conc_col}_num"].notna().astype(int)
    else:
        df["has_conc"] = 0

    return df


def quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Run sanity checks: negative removal, simple IQR outlier flagging (not dropping by default)."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    checks = {}
    # Remove obviously negative physical measures (size, concentration)
    size_like = [c for c in numeric_cols if "size" in c or "diameter" in c or "length" in c]
    conc_like = [c for c in numeric_cols if "conc" in c or "concentration" in c or "count" in c or "particles" in c]

    for c in set(size_like + conc_like):
        if c in df.columns:
            negs = (df[c] < 0).sum()
            if negs > 0:
                checks[f"removed_negatives_in_{c}"] = int(negs)
                df.loc[df[c] < 0, c] = np.nan

    # IQR-based outlier detection (flag columns)
    outlier_flags = pd.DataFrame(index=df.index)
    for c in numeric_cols:
        ser = df[c].dropna()
        if ser.shape[0] < 4:
            continue
        q1 = ser.quantile(0.25)
        q3 = ser.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_flags[f"{c}_outlier"] = ((df[c] < lower) | (df[c] > upper)).astype(int)

    # Attach a summary column (number of numeric outliers per row)
    if not outlier_flags.empty:
        df = pd.concat([df, outlier_flags], axis=1)
        df["numeric_outlier_count"] = outlier_flags.sum(axis=1)
    else:
        df["numeric_outlier_count"] = 0

    # Report checks to Streamlit
    if checks:
        st.write("Quality fixes applied (negative value handling):")
        st.write(checks)
    st.write("Outlier flags added as '..._outlier' boolean numeric columns and 'numeric_outlier_count'.")

    return df


def standardize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize several microplastic-related categorical columns (type, polymer, shape, color)."""
    df = df.copy()
    # Candidate columns
    type_col = find_column(df, ["type", "shape", "morphology"])
    polymer_col = find_column(df, ["polymer", "material", "composition"])
    color_col = find_column(df, ["color", "colour", "hue"])
    shape_col = find_column(df, ["shape", "form", "morphology"])

    if type_col:
        df[type_col] = df[type_col].astype(str).apply(standardize_microplastic_type)
    if polymer_col:
        df[polymer_col] = df[polymer_col].astype(str).apply(standardize_polymer)
    if color_col:
        df[color_col] = df[color_col].astype(str).str.strip().str.lower().replace({"nan": np.nan})
    if shape_col and shape_col != type_col:
        df[shape_col] = df[shape_col].astype(str).str.strip().str.lower().replace({"nan": np.nan})

    # Reduce cardinality of textual categorical columns
    for c in df.select_dtypes(include=["object", "category"]).columns:
        df[c] = limit_cardinality_ser(df[c], top_n=60)

    return df


def finalize_encoding(df: pd.DataFrame, target_col: str = None):
    """Encode categorical features (one-hot) while locking down final feature columns.
    Returns encoded df and the list of final feature columns (for modeling)."""
    df = df.copy()
    # Identify categorical features excluding target
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target_col:
        cat_cols = [c for c in cat_cols if c != target_col]

    # One-hot encode but drop_first to avoid collinearity
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    # Drop columns that are exact duplicates of the target (rare) or accidentally created
    if target_col and target_col in df_encoded.columns:
        df_encoded.drop(columns=[target_col], inplace=True, errors="ignore")

    # Lock features: exclude identifier-like columns
    id_like = [c for c in df_encoded.columns if re.search(r"^(id|index|sampleid|sample_id|siteid)", c)]
    final_features = [c for c in df_encoded.columns if c not in id_like]
    return df_encoded, final_features


def describe_df(df):
    st.write("Shape:", df.shape)
    st.write("Columns and dtypes:")
    st.dataframe(pd.DataFrame(df.dtypes, columns=["dtype"]).T)


# -------------------------
# STEP 1: Sidebar Upload (CSV & Excel)
# -------------------------
st.sidebar.header("STEP 1: Upload Dataset")
st.sidebar.write("Supported formats: CSV, XLSX, XLS")

uploaded_file = st.sidebar.file_uploader(
    "Choose a file",
    type=["csv", "xls", "xlsx"],
    help="Upload CSV or Excel. If Excel has multiple sheets, select sheet below.",
)

# Optional Excel sheet selection (only shown after upload and only for Excel)
selected_sheet = 0
if uploaded_file is not None and getattr(uploaded_file, "name", "").lower().endswith((".xls", ".xlsx")):
    # Try to preview sheet names
    try:
        # Read bytes and inspect Excel sheet names
        uploaded_file.seek(0)
        excel_bytes = uploaded_file.read()
        xls = pd.ExcelFile(io.BytesIO(excel_bytes))
        sheets = xls.sheet_names
        # Put selection widget in sidebar
        selected_sheet = st.sidebar.selectbox("Select sheet", options=sheets, index=0)
        # Reset file pointer
        uploaded_file.seek(0)
    except Exception:
        # If unable to inspect sheet names, fallback to first sheet
        selected_sheet = 0

if not uploaded_file:
    st.sidebar.info("📂 Upload a CSV or Excel file in the sidebar to start preprocessing and modeling.")
    st.stop()

# Read file with robust reader
try:
    df_raw = safe_read_file(uploaded_file, sheet_name=selected_sheet)
    st.sidebar.success("✅ File recognized and read.")
except Exception as e:
    st.sidebar.error(f"Failed to read uploaded file: {e}")
    st.stop()

st.subheader("Preview (first 5 rows)")
st.dataframe(df_raw.head())

# -------------------------
# STEP 2: Preprocessing (Cleaning, Standardization, Derivation)
# -------------------------
st.header("STEP 2: Data Cleaning, Standardization & Feature Engineering")

st.write(f"Initial records: {len(df_raw)}")

# Make a working copy
df = df_raw.copy()

# Clean column names
df = clean_column_names(df)
st.write("Normalized column names to snake_case for consistency.")
describe_df(df)

# Drop exact duplicate rows
before_dup = len(df)
df.drop_duplicates(inplace=True)
after_dup = len(df)
st.write(f"Removed duplicates: {before_dup - after_dup} rows dropped. Remaining: {after_dup}")

# Trim whitespace on string columns
for c in df.select_dtypes(include=["object"]).columns:
    df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

# Basic missing-value overview
st.subheader("Missing values per column (before imputation)")
missing = df.isnull().sum()
st.dataframe(missing[missing > 0].sort_values(ascending=False))

# Standardize category values (taxonomy)
df = standardize_categories(df)
st.success("✅ Categorical taxonomy standardized (type, polymer, color, shape where detected).")

# Parse numeric tokens from mixed-type columns and derive features
df = derive_features(df)
st.success("✅ Derived features created (size_category, log concentration, presence flags, etc.).")

# Handle missing values with sensible defaults:
# Numeric -> median; Categorical -> mode (after taxonomy reduction)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

if num_cols:
    medians = df[num_cols].median()
    df[num_cols] = df[num_cols].fillna(medians)

for c in cat_cols:
    if df[c].isnull().any():
        modes = df[c].mode()
        if not modes.empty:
            df[c] = df[c].fillna(modes.iloc[0])
        else:
            df[c] = df[c].fillna("unknown")

st.success("✅ Missing values imputed (median for numeric, mode for categorical).")

# Run quality checks (negatives -> NaN -> imputed; outlier flags added)
df = quality_checks(df)

st.subheader("Preview after preprocessing")
st.dataframe(df.head())

# Keep original for target selection
original_df = df.copy()

# -------------------------
# STEP 2b: Encoding & Locking Feature Set
# -------------------------
st.header("STEP 2B: Encoding & Feature Lock")
st.write("We'll prepare a locked-down feature matrix for modeling (one-hot categorical encoding, limited cardinality).")

# Ask user if they want to exclude identifier columns
exclude_identifiers = st.checkbox("Automatically drop ID-like columns (sample_id, id, index)?", value=True)

# Final encoding (target excluded later)
df_encoded, locked_features = finalize_encoding(df, target_col=None)
if exclude_identifiers:
    locked_features = [c for c in locked_features if not re.search(r"^(id|index|sampleid|sample_id|siteid)", c)]

st.write(f"Encoded shape: {df_encoded.shape}")
st.write(f"Number of locked features available for modeling: {len(locked_features)}")
st.dataframe(pd.DataFrame({"locked_features": locked_features}).head(200))

# -------------------------
# STEP 3: Clustering (Unsupervised)
# -------------------------
st.header("STEP 3: Clustering Analysis (Unsupervised Learning)")

with st.expander("Clustering settings and results", expanded=True):
    if df_encoded.shape[0] < 2 or len(locked_features) < 2:
        st.warning("Not enough data/features to perform clustering.")
    else:
        clustering_features = st.multiselect(
            "Select features for clustering (at least 2). Choose from locked features",
            options=locked_features,
            default=locked_features[:4],
        )

        if len(clustering_features) >= 2:
            scaler = StandardScaler()
            X_cluster = scaler.fit_transform(df_encoded[clustering_features])

            n_clusters = st.slider("Number of clusters (k)", min_value=2, max_value=10, value=3)
            kmeans = KMeans(n_clusters=int(n_clusters), random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_cluster)
            df["cluster"] = cluster_labels

            st.success(f"✅ K-Means clustering complete ({n_clusters} clusters).")
            st.write("Cluster counts:")
            st.write(df["cluster"].value_counts().sort_index())

            # Silhouette
            try:
                sil = silhouette_score(X_cluster, cluster_labels)
                st.write(f"Silhouette Score: {sil:.3f}")
            except Exception:
                st.write("Silhouette Score: Not available for this configuration.")

            # Scatter plot of first two selected features
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
st.write("Select a target column for supervised learning. The app will detect whether it looks like a classification target or regression target.")

target_col = st.selectbox("Select Target Column", options=original_df.columns.tolist())

if target_col is None:
    st.warning("Select a target column to continue.")
    st.stop()

# Determine task type heuristically
y_series = original_df[target_col]
unique_vals = y_series.nunique(dropna=True)
is_numeric_target = pd.api.types.is_numeric_dtype(y_series)

if (not is_numeric_target) or (is_numeric_target and unique_vals <= 20):
    task = "classification"
else:
    task = "regression"

st.write(f"Detected task type: {task}")

# Prepare X and y: ensure we don't include target-derived columns in X
df_encoded_with_target, _ = finalize_encoding(original_df, target_col=target_col)
X = df_encoded_with_target.drop(columns=[c for c in df_encoded_with_target.columns if c == target_col or c.startswith(f"{target_col}_")], errors="ignore")
# Restrict to locked features intersection to avoid unexpected columns
X = X[[c for c in locked_features if c in X.columns]]

# Prepare y
label_encoder = None
if task == "classification":
    try:
        if not is_numeric_target:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(original_df[target_col].astype(str))
        else:
            # numeric but small unique values -> categorical label encode
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(original_df[target_col].astype(str))
    except Exception as e:
        st.error(f"Failed to encode classification target: {e}")
        st.stop()
else:
    # regression
    try:
        y = original_df[target_col].apply(extract_numeric).astype(float).values
    except Exception as e:
        st.error(f"Failed to parse regression target as numeric: {e}")
        st.stop()

# Validate shapes
if X.shape[0] != len(y):
    st.error("Feature and target lengths do not match after preprocessing. Check the dataset for missing rows or parsing issues.")
    st.stop()

st.write(f"Feature matrix shape: {X.shape} | Target shape: {y.shape}")

# Optionally scale numeric features for regression; RandomForest not required
scaler_for_reg = StandardScaler()
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

# Modeling
if task == "classification":
    st.subheader("Classification: Random Forest (default)")

    test_size = st.slider("Test set proportion", 0.05, 0.5, 0.2, step=0.05)

    # Decide whether to stratify splits based on class counts
    y_counts = pd.Series(y).value_counts()
    min_class_count = int(y_counts.min()) if not y_counts.empty else 0
    total_samples = X.shape[0]
    # If any class has <2 samples, stratification for train_test_split or StratifiedKFold is not safe.
    stratify_allowed = min_class_count >= 2

    if not stratify_allowed:
        st.warning(
            f"Stratified splitting is disabled because at least one class has fewer than 2 samples (min class count = {min_class_count}). "
            "The split and cross-validation will use non-stratified K-Fold methods."
        )

    try:
        if stratify_allowed:
            # stratify only if safe (each class has >=2 samples)
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

        # Cross-validation (choose StratifiedKFold when safe; otherwise KFold)
        folds = st.slider("K-Folds for cross-validation", 2, 10, 5)

        # ensure folds no larger than total samples
        if folds > total_samples:
            st.warning(f"Number of folds reduced from {folds} to {total_samples} (can't have more folds than samples).")
            folds = total_samples if total_samples >= 2 else 2

        # If stratified requested but min_class_count < folds, reduce folds or fallback
        if stratify_allowed:
            if min_class_count < folds:
                st.warning(f"Reducing number of folds from {folds} to {min_class_count} because the smallest class only has {min_class_count} samples.")
                folds = min_class_count if min_class_count >= 2 else 2
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=folds, shuffle=True, random_state=42)

        # compute cv scores (wrap in try to surface errors)
        try:
            cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
            st.write(f"Cross-Validation Accuracy scores: {np.round(cv_scores, 3)}")
            st.success(f"Average CV accuracy: {np.mean(cv_scores)*100:.2f}% (+/- {np.std(cv_scores)*100:.2f}%)")

            plt.figure(figsize=(6, 4))
            sns.lineplot(x=range(1, len(cv_scores) + 1), y=cv_scores, marker="o")
            plt.title("Cross-Validation Accuracy per Fold")
            plt.xlabel("Fold")
            plt.ylabel("Accuracy")
            st.pyplot(plt.gcf())
            plt.clf()
        except Exception as e:
            st.error(f"Cross-validation failed: {e}. You may need to reduce folds or disable stratification.")
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
        if folds > X_reg.shape[0]:
            st.warning(f"Number of folds reduced from {folds} to {max(2, X_reg.shape[0])} because there are fewer samples than folds.")
            folds = max(2, X_reg.shape[0])
        cv = KFold(n_splits=folds, shuffle=True, random_state=42)
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
    - Data cleaning: normalized column names, trimmed whitespace, removed duplicate rows.
    - Taxonomy standardization: microplastic types, polymers, color, shape were normalized to controlled vocabularies.
    - Derived features: size categories, log-transformed concentrations, presence flags (has_size, has_conc), conc_per_area when possible.
    - Quality checks: negative physical values handled, IQR outlier flags added.
    - Encoding: categorical features cardinality limited and one-hot encoded; feature set locked for modeling.
    - Clustering & Modeling: KMeans clustering, RandomForest classifier/regressor, LinearRegression, and K-Fold cross-validation.
    """
)

st.caption("© 2025 Magdaluyo & Viernes | Predictive Risk Modeling for Microplastic Pollution")
