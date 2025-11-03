import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
# Add all other necessary imports here (StandardScaler, OneHotEncoder, your classifiers, etc.)
# from sklearn.compose import ColumnTransformer 
# from sklearn.linear_model import LogisticRegression 
# from sklearn.ensemble import RandomForestClassifier

# Use @st.cache_resource for heavy, non-data-specific objects like models and transformers
@st.cache_resource
def train_model(_preprocessor, X_train, y_train, classifier):
    """
    Trains the ML model pipeline using a preprocessor.
    
    The leading underscore on '_preprocessor' tells Streamlit NOT to hash this 
    non-hashable ColumnTransformer object, thereby fixing the error.
    """
    
    # Inside the function, you must refer to the argument using the underscore
    model_pipeline = Pipeline(steps=[
        # Use the corrected argument name here: _preprocessor
        ('preprocessor', _preprocessor), 
        ('classifier', classifier)
    ])
    
    # Fit the entire pipeline
    model_pipeline.fit(X_train, y_train)
    
    return model_pipeline

# --- Example Usage (How you call the function) ---
# Assuming you have defined your preprocessor, training data, and classifier:
# preprocessor = ColumnTransformer(...) 
# X_train, y_train = ...
# log_reg = LogisticRegression(max_iter=1000)

# The function call remains the same, passing the preprocessor object:
# trained_lr_model = train_model(preprocessor, X_train, y_train, log_reg) 
# st.write("Model trained successfully!")
