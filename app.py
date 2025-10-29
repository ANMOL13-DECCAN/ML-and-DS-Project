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


# Optional: xgboost (will be available if installed)
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
st.error(f"Failed to run predictions: {e}")
