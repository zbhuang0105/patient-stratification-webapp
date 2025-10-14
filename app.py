import streamlit as st
import pandas as pd
import joblib
import pickle
import shap

# --- Page Configuration ---
st.set_page_config(
    page_title="Patient Stratification Dashboard",
    page_icon="🩺",
    layout="wide"
)

# --- Data Loading (Cached for performance) ---
@st.cache_resource
def load_models_and_explainer():
    model = joblib.load('streamlit_data/xgb_model.joblib')
    subcluster_model = joblib.load('streamlit_data/subcluster_model.joblib')
    
    # 关键改动：在运行时创建 explainer，而不是从文件中加载
    # 我们需要先加载特征数据来创建它
    X_df_for_explainer = pd.read_csv('streamlit_data/all_features.csv')
    explainer = shap.Explainer(model, X_df_for_explainer)
    
    return model, subcluster_model, explainer

@st.cache_data
def load_data():
    X_df = pd.read_csv('streamlit_data/all_features.csv')
    Y = pd.read_csv('streamlit_data/all_labels.csv')['Group'].values
    mod_clustered_df = pd.read_csv('streamlit_data/mod_patients_clustered.csv')
    
    # 关键改动：现在只从 pkl 文件中加载 shap_values
    with open('streamlit_data/shap_data.pkl', 'rb') as f:
        shap_values = pickle.load(f)['shap_values']
        
    return X_df, Y, mod_clustered_df, shap_values

# --- Load all data ---
model, subcluster_model, explainer = load_models_and_explainer()
X_df, Y, mod_clustered_df, shap_values = load_data()

# --- Store data in session_state for sharing across pages ---
st.session_state['model'] = model
st.session_state['subcluster_model'] = subcluster_model
st.session_state['explainer'] = explainer
st.session_state['X_df'] = X_df
st.session_state['Y'] = Y
st.session_state['mod_clustered_df'] = mod_clustered_df
st.session_state['shap_values'] = shap_values
st.session_state['class_names'] = ['Mild', 'Moderate', 'Severe']
st.session_state['feature_names'] = X_df.columns.tolist()


# --- Main Page Content ---
st.title("🩺 Patient Stratification & ML Explainability Platform")

st.markdown("""
Welcome to the platform. This tool is designed for real-time patient risk stratification.

Please navigate to the **New Patient Prediction** page using the sidebar on the left to begin.

### Core Functionality:
*   **New Patient Prediction**: Input a new patient's data to get a real-time prediction of their severity level (Mild, Moderate, or Severe) and a detailed explanation of the prediction.
""")

st.info("Please select a page from the sidebar to begin.")
