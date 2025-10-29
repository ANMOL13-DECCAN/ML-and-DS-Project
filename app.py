# File: app.py
"""
Interactive Streamlit ML + EDA app (fixed indentation error)
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Optional: xgboost (fixed indentation block)
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="ML + EDA Streamlit App")

st.title("Interactive ML & Data Science App")
st.write("Upload a CSV dataset, pick a target column and an algorithm, then train and evaluate.")

# --- sidebar: upload and options ---
with st.sidebar:
    st.header("Upload & Settings")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])    
    test_size = st.slider("Test set proportion", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.number_input("Random state (seed)", value=42, step=1)
    show_pairplot = st.checkbox("Show pairplot (may be slow)", value=False)

# --- load data ---
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()
else:
    st.info("Please upload a CSV file to get started.")
    st.stop()

# --- basic EDA ---
st.header("Data preview & EDA")

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Preview")
    st.dataframe(df.head())
    st.write(f"Shape: {df.shape}")
    if st.button("Show full column list"):
        st.write(list(df.columns))

with col2:
    st.subheader("Missing values")
    miss = df.isnull().sum()
    st.dataframe(miss[miss > 0].sort_values(ascending=False))

st.subheader("Column data types")
st.write(df.dtypes)

st.subheader("Target selection")
all_columns = list(df.columns)
target_col = st.selectbox("Select target column", options=all_columns)

@st.cache_data
def infer_problem(y_series):
    if pd.api.types.is_object_dtype(y_series) or pd.api.types.is_categorical_dtype(y_series):
        return 'classification'
    n_unique = y_series.nunique(dropna=True)
    if pd.api.types.is_integer_dtype(y_series) and n_unique <= 20:
        return 'classification'
    if n_unique <= 10 and n_unique < 0.05 * len(y_series):
        return 'classification'
    if pd.api.types.is_numeric_dtype(y_series):
        return 'regression'
    return 'classification'

problem_type = infer_problem(df[target_col])
st.info(f"Detected problem type: {problem_type}")

if problem_type == 'classification':
    st.write(df[target_col].value_counts())
else:
    st.write(df[target_col].describe())

st.subheader("Visualizations")
eda_col1, eda_col2 = st.columns(2)
with eda_col1:
    st.write("Histogram of numeric columns")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        sel = st.selectbox("Choose numeric column to plot histogram", options=numeric_cols)
        fig, ax = plt.subplots()
        ax.hist(df[sel].dropna(), bins=30)
        ax.set_title(f"Histogram: {sel}")
        st.pyplot(fig)

with eda_col2:
    st.write("Correlation heatmap (numeric columns)")
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, fmt='.2f', ax=ax)
        st.pyplot(fig)

if show_pairplot:
    try:
        st.write("Pairplot (this can be slow for many columns)")
        pp_fig = sns.pairplot(df.select_dtypes(include=[np.number]).dropna().sample(n=min(200, len(df))))
        st.pyplot(pp_fig)
    except Exception as e:
        st.write("Pairplot failed:", e)

st.header("Modeling")
X = df.drop(columns=[target_col])
y = df[target_col]

num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
st.write(f"Numeric features: {len(num_features)} | Categorical features: {len(cat_features)}")

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy='median')),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy='most_frequent')),
    ("onehot", OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ],
    remainder='drop'
)

if problem_type == 'classification':
    model_options = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Machine': SVC(probability=True)
    }
    if HAS_XGB:
        model_options['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
else:
    model_options = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(n_estimators=100),
        'K-Nearest Neighbors': KNeighborsRegressor(),
        'Support Vector Regressor': SVR()
    }
    if HAS_XGB:
        model_options['XGBoost'] = XGBRegressor()

model_name = st.selectbox("Choose model", options=list(model_options.keys()))
model = model_options[model_name]

if st.button("Train model"):
    with st.spinner("Training..."):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state),
            stratify=(y if problem_type=='classification' and y.nunique()>1 else None)
        )

        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        try:
            pipe.fit(X_train, y_train)
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

        y_pred = pipe.predict(X_test)
        st.success("Training finished")

        if problem_type == 'classification':
            acc = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{acc:.4f}")
            st.subheader("Classification report")
            st.text(classification_report(y_test, y_pred))
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            st.pyplot(fig)
        else:
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            st.metric("R2", f"{r2:.4f}")
            st.metric("MAE", f"{mae:.4f}")
            st.metric("RMSE", f"{rmse:.4f}")
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.6)
            st.pyplot(fig)

        st.subheader("Model details")
        try:
            mdl = pipe.named_steps['model']
            if hasattr(mdl, 'feature_importances_'):
                st.write("Feature importances (top 20)")
                try:
                    ohe = pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
                    cat_names = ohe.get_feature_names_out(cat_features)
                except Exception:
                    cat_names = []
                feature_names = list(num_features) + list(cat_names)
                importances = mdl.feature_importances_
                fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20)
                st.bar_chart(fi)
        except Exception as e:
            st.write("Could not show model details:", e)

        st.subheader("Make predictions on new data")
        upload_pred = st.file_uploader("Upload CSV with same features to predict", key='pred')
        if upload_pred is not None:
            try:
                to_pred = pd.read_csv(upload_pred)
                preds = pipe.predict(to_pred)
                out = to_pred.copy()
                out['prediction'] = preds
                st.dataframe(out.head())
                csv = out.to_csv(index=False).encode('utf-8')
                st.download_button("Download predictions as CSV", csv, file_name='predictions.csv')
            except Exception as e:
                st.error(f"Failed to run predictions: {e}")
