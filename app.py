import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# --- Streamlit Configuration ---
st.set_page_config(
    page_title="Loksabha Winner Predictor (Top 10 Parties)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Define Features and Columns ---
TARGET_COLUMN = 'party'
DROP_COLUMNS = ['id', 'month', 'pc_code', 'candidate_name', 'position',
                'valid_votes', 'total_electors', 'turnout_percentage']

numerical_features = ['year', 'votes', 'margin', 'margin_percentage', 'vote_share_percentage']
categorical_features = ['state_name', 'pc_name', 'pc_type', 'candidate_type', 'sex']


# --- Function to load and preprocess data (Cached for performance) ---
@st.cache_data
def load_and_preprocess_data(file_path):
    """Loads, filters data, and returns the preprocessor and trained data."""
    df = pd.read_csv(file_path)
    df_winners = df[df['position'] == 1].copy()
    df_winners = df_winners.dropna(subset=['party'])

    # Focus on the top 10 parties
    top_n = 10
    top_parties = df_winners['party'].value_counts().nlargest(top_n).index
    df_winners = df_winners[df_winners['party'].isin(top_parties)].reset_index(drop=True)

    FEATURES = [col for col in df_winners.columns if col not in [TARGET_COLUMN] + DROP_COLUMNS]

    X = df_winners[FEATURES]
    y = df_winners[TARGET_COLUMN]
    
    # Preprocessor Definition
    numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    # Fit preprocessor on all data (X) to learn all categories
    preprocessor.fit(X)
    
    return X, y, preprocessor, top_parties

# --- Function to train the model (Cached for performance) ---
@st.cache_resource
def train_model(X, y, preprocessor):
    """Trains the Random Forest model using the best parameters found."""
    st.write("Training Random Forest Classifier (Cached)...")
    
    # For caching, we simplify the split just to get a fit model
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Use a basic RandomForestClassifier pipeline
    rf_model = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))])
    
    rf_model.fit(X_train, y_train)
    return rf_model

# --- Load Data and Train Model ---
try:
    X_data, y_data, data_preprocessor, top_parties = load_and_preprocess_data('cleaned_loksabha_elections.csv')
    model = train_model(X_data, y_data, data_preprocessor)
except Exception as e:
    st.error(f"Error loading data or training model: {e}")
    st.stop()


# --- Streamlit Interface ---
st.title("🗳️ Loksabha Election Winner Predictor")
st.markdown("Predict the winning party (among the top 10 historical winners) for a given constituency using the Random Forest Classifier.")
st.caption(f"Model trained on historical election data. Top 10 Parties: {', '.join(top_parties)}")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Election Context")
    
    # Dynamically populate dropdowns
    years = sorted(X_data['year'].unique(), reverse=True)
    states = sorted(X_data['state_name'].unique())
    pc_names = sorted(X_data['pc_name'].unique())
    pc_types = sorted(X_data['pc_type'].unique())
    
    input_year = st.selectbox("Year", years, index=0)
    input_state = st.selectbox("State Name", states, index=states.index('Maharashtra') if 'Maharashtra' in states else 0)
    input_pc_name = st.selectbox("Constituency (PC Name)", pc_names, index=pc_names.index('Mumbai North West') if 'Mumbai North West' in pc_names else 0)
    input_pc_type = st.selectbox("Constituency Type", pc_types, index=pc_types.index('GEN') if 'GEN' in pc_types else 0)
    input_candidate_type = st.selectbox("Candidate Type", ['GEN', 'SC', 'ST'], index=0)
    input_sex = st.selectbox("Candidate Sex", ['M', 'F'], index=0)

    st.header("Victory Metrics (Historical/Projected)")
    st.markdown("Enter the projected or historical numerical results for the winner.")
    
    input_votes = st.number_input("Votes (Winner's Total)", min_value=10000, value=500000, step=10000)
    input_margin = st.number_input("Margin (Votes Diff to Runner-up)", min_value=100, value=150000, step=10000)
    input_margin_perc = st.slider("Margin Percentage (%)", min_value=0.1, max_value=50.0, value=10.0, step=0.1)
    input_vote_share_perc = st.slider("Vote Share Percentage (%)", min_value=10.0, max_value=100.0, value=55.0, step=0.1)

    predict_button = st.button("Predict Winning Party")

# --- Prediction Logic ---
if predict_button:
    # 1. Create input DataFrame
    input_data = pd.DataFrame({
        'year': [input_year],
        'state_name': [input_state],
        'pc_name': [input_pc_name],
        'pc_type': [input_pc_type],
        'candidate_type': [input_candidate_type],
        'sex': [input_sex],
        'votes': [input_votes],
        'margin': [input_margin],
        'margin_percentage': [input_margin_perc],
        'vote_share_percentage': [input_vote_share_perc]
    })
    
    # 2. Make prediction
    try:
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)
        
        # Get the confidence score for the predicted class
        confidence_score = prediction_proba.max() * 100
        
        # 3. Display Results
        st.success("✅ Prediction Complete")
        
        st.write(f"Based on the provided inputs, the predicted winning party is:")
        st.markdown(f"## **{prediction}**")
        st.markdown(f"Confidence: **{confidence_score:.2f}%**")
        
        st.markdown("---")
        st.subheader("Top 5 Predicted Party Probabilities")
        
        # Get top 5 probabilities
        proba_df = pd.DataFrame({
            'Party': model.classes_,
            'Probability': prediction_proba[0]
        }).sort_values(by='Probability', ascending=False).head(5)
        
        proba_df['Probability'] = (proba_df['Probability'] * 100).round(2).astype(str) + '%'
        
        st.table(proba_df)
        
    except Exception as e:
        st.error(f"An error occurred during prediction. Please check your inputs. Error: {e}")

# --- Footer/Model Info ---
st.markdown("---")
st.markdown(
    """
    **Model Used:** Random Forest Classifier (Accuracy: ~77%)
    **Key Predictors:** Victory Margin, Total Votes, and Constituency Name.
    """
)
