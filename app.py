import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# --- 1. Data Loading and Preparation (Cached) ---

@st.cache_data
def load_and_preprocess_data():
    """Mocks the loading and initial filtering of the Loksabha data."""
    
    # Mock Data Creation (Replace with actual data loading if 'cleaned_loksabha_elections.csv' is available)
    data = {
        'year': [2019, 2019, 2019, 2014, 2014, 2019, 2019, 2014, 2019, 2019],
        'state_name': ['MAHARASHTRA', 'UP', 'BIHAR', 'MAHARASHTRA', 'UP', 'MAHARASHTRA', 'BIHAR', 'UP', 'KARNATAKA', 'TAMIL NADU'],
        'pc_name': ['Pune', 'Varanasi', 'Patna Sahib', 'Pune', 'Varanasi', 'Nagpur', 'Nalanda', 'Amethi', 'Bangalore South', 'Chennai North'],
        'pc_type': ['GEN', 'GEN', 'GEN', 'GEN', 'GEN', 'GEN', 'GEN', 'GEN', 'GEN', 'GEN'],
        'votes': [568000, 674000, 395000, 400000, 500000, 600000, 350000, 450000, 700000, 450000],
        'margin': [100000, 480000, 55000, 50000, 300000, 350000, 15000, 100000, 200000, 50000],
        'party': ['BJP', 'BJP', 'JDU', 'INC', 'BJP', 'BJP', 'JDU', 'INC', 'BJP', 'DMK'],
        'candidate_type': ['MALE', 'MALE', 'MALE', 'MALE', 'MALE', 'MALE', 'MALE', 'FEMALE', 'FEMALE', 'MALE'],
        'sex': ['M', 'M', 'M', 'M', 'M', 'M', 'M', 'F', 'F', 'M']
    }
    df = pd.DataFrame(data)

    # Filter to only top 10 parties (mocked to just 4 for simplicity)
    top_parties = df['party'].value_counts().nlargest(4).index.tolist()
    df_filtered = df[df['party'].isin(top_parties)].copy()

    # Define features and target
    numerical_features = ['year', 'votes', 'margin']
    categorical_features = ['state_name', 'pc_name', 'pc_type', 'candidate_type', 'sex']
    
    X = df_filtered[numerical_features + categorical_features]
    y = df_filtered['party']
    
    # Define Preprocessor (ColumnTransformer)
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return preprocessor, X_train, y_train, X_test, y_test, numerical_features, categorical_features


# --- 2. Model Training (Fixes the Caching Error) ---

# Use @st.cache_resource for heavy, non-data-specific objects like models and transformers.
@st.cache_resource
def train_model(_preprocessor, X_train, y_train, classifier):
    """
    Trains the ML model pipeline using a preprocessor.
    
    The leading underscore on '_preprocessor' tells Streamlit NOT to hash the 
    non-hashable ColumnTransformer object, thereby fixing the error.
    """
    
    # Inside the function, you must refer to the argument using the underscore
    model_pipeline = Pipeline(steps=[
        ('preprocessor', _preprocessor), 
        ('classifier', classifier)
    ])
    
    # Fit the entire pipeline
    model_pipeline.fit(X_train, y_train)
    
    return model_pipeline

# --- 3. Streamlit App Layout ---

def main():
    st.set_page_config(
        page_title="Loksabha Winner Predictor (Top 4 Parties)",
        layout="centered",
        initial_sidebar_state="auto"
    )

    st.title("🇮🇳 Loksabha Winner Prediction System")
    st.markdown("---")
    
    # Load and prepare data (cached)
    preprocessor, X_train, y_train, X_test, y_test, num_feats, cat_feats = load_and_preprocess_data()
    
    st.sidebar.header("Model Selection & Training")
    
    # Define classifier
    classifier_name = st.sidebar.selectbox(
        'Select Classifier (LR proved optimal)',
        ('Logistic Regression', 'Random Forest')
    )

    if classifier_name == 'Logistic Regression':
        classifier = LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=42)
    else:
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model (Calling the fixed function)
    st.sidebar.info(f"Training {classifier_name} (Cached)...")
    try:
        model = train_model(preprocessor, X_train, y_train, classifier)
        
        # Evaluate performance
        accuracy = model.score(X_test, y_test)
        st.sidebar.success(f"{classifier_name} Trained Successfully!")
        st.sidebar.metric("Test Accuracy (Mocked)", f"{accuracy:.2%}")
        st.markdown(f"### Trained Model: {classifier_name}")
        st.write(f"This model achieved a competitive $\\mathbf{{77.22\\%}}$ accuracy in the original project, demonstrating the linearity of the top 10 party classification problem.")
        
    except Exception as e:
        st.error(f"Error during model training or prediction: {e}")
        st.warning("Please ensure your dataset ('cleaned_loksabha_elections.csv') is available if mocking fails.")
        return

    st.markdown("---")
    
    # --- User Input Form ---
    st.header("Predict Winner for a New Constituency")
    
    with st.form("prediction_form"):
        # Get unique values for select boxes from the data (mocked)
        state_options = ['MAHARASHTRA', 'UP', 'BIHAR', 'KARNATAKA', 'TAMIL NADU']
        pc_options = ['Pune', 'Varanasi', 'Patna Sahib', 'Nagpur', 'Nalanda', 'Amethi', 'Bangalore South', 'Chennai North']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            input_year = st.slider("Election Year", 2014, 2024, 2019)
            input_state = st.selectbox("State Name", state_options)
        
        with col2:
            input_pc = st.selectbox("Constituency Name", pc_options)
            input_type = st.selectbox("PC Type", ['GEN', 'ST', 'SC'])
            
        with col3:
            # Note: Votes and Margin are the most important features
            input_votes = st.number_input("Votes (Winner)", min_value=100000, max_value=1000000, value=550000, step=10000)
            input_margin = st.number_input("Margin (Winner)", min_value=1000, max_value=500000, value=100000, step=1000)

        # Assuming male candidate and GEN type for simplicity
        input_candidate_type = 'MALE'
        input_sex = 'M'
        
        submitted = st.form_submit_button("Predict Winning Party")

    if submitted:
        # 1. Create Input DataFrame
        input_data = pd.DataFrame([{
            'year': input_year, 
            'votes': input_votes, 
            'margin': input_margin,
            'state_name': input_state, 
            'pc_name': input_pc, 
            'pc_type': input_type,
            'candidate_type': input_candidate_type,
            'sex': input_sex
        }])
        
        # 2. Predict
        try:
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # 3. Display Results
            st.success(f"### The Predicted Winning Party is: **{prediction}**")
            
            st.subheader("Probability Breakdown")
            proba_df = pd.DataFrame({
                'Party': model.classes_,
                'Probability': prediction_proba
            }).sort_values(by='Probability', ascending=False)
            
            st.dataframe(proba_df, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"Prediction failed. Error: {e}")
            st.warning("This is often due to the one-hot encoder not seeing a value in the training data (e.g., a specific PC Name or State).")

if __name__ == "__main__":
    main()
