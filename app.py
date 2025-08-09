import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import time
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# --- App Configuration ---
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Model Training & Loading Functions (Cached for performance) ---
@st.cache_resource
def train_and_save_model(data_path='predictive_maintenance.csv', model_path='rf_model.joblib'):
    st.info(f"Training model with data from '{data_path}'...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"Error: Dataset '{data_path}' not found. Please place it in the same directory.")
        return None, None, None
    features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    target = 'Machine failure'
    if not all(col in df.columns for col in features + [target]):
        st.error("Dataset is missing required columns.")
        return None, None, None
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    st.success(f"Model training complete. Accuracy: {accuracy:.2%}")
    joblib.dump({'model': model, 'features': features, 'dataframe': df}, model_path)
    st.success(f"Model and data saved to '{model_path}'")
    return model, features, df

@st.cache_resource
def load_model_and_data(model_path='rf_model.joblib'):
    if not os.path.exists(model_path):
        return None, None, None
    saved_dict = joblib.load(model_path)
    return saved_dict['model'], saved_dict['features'], saved_dict['dataframe']

# --- Helper function to get a specific machine's data ---
def get_machine_scenario(df, udi):
    """Returns the data sequence for a specific machine UDI."""
    scenario_df = df[df['UDI'] == udi].sort_values(by='Tool wear [min]').reset_index(drop=True)
    if scenario_df.empty:
        return None
    return scenario_df

# --- Main Application Logic ---
MODEL_FILE = 'rf_model.joblib'
DATA_FILE = 'predictive_maintenance.csv'

model, features, df = load_model_and_data(MODEL_FILE)
if model is None:
    model, features, df = train_and_save_model(DATA_FILE, MODEL_FILE)
    if model is None:
        st.stop()

# --- Initialize Session State ---
if 'simulation_mode' not in st.session_state:
    st.session_state.simulation_mode = 'inactive'
    st.session_state.replay_df = None
    st.session_state.replay_index = 0
    st.session_state.history = pd.DataFrame(columns=['Time'] + features)

# --- Sidebar ---
st.sidebar.title("Control Panel")
st.sidebar.header("Live Simulation")

# --- NEW: Select a specific machine for replay ---
st.sidebar.subheader("Forensic Analysis")
st.sidebar.markdown("Select a specific machine to replay its historical data.")
machine_list = df['UDI'].unique()
selected_udi = st.sidebar.selectbox("Select Machine UDI", options=machine_list)

if st.sidebar.button(f"🎬 Replay Machine {selected_udi}"):
    scenario = get_machine_scenario(df, selected_udi)
    if scenario is not None:
        st.session_state.replay_df = scenario
        st.session_state.replay_index = 0
        st.session_state.simulation_mode = 'replay'
        st.session_state.history = pd.DataFrame(columns=['Time'] + features)
    else:
        st.sidebar.error(f"No data found for machine UDI {selected_udi}.")

if st.sidebar.button("⏹️ Stop Simulation"):
    st.session_state.simulation_mode = 'inactive'

st.sidebar.header("Manual 'What-If' Analysis")
st.sidebar.markdown("Use sliders for manual analysis when simulation is stopped.")
is_disabled = st.session_state.simulation_mode != 'inactive'
air_temp = st.sidebar.slider('Air Temperature [K]', 295.0, 305.0, 300.1, 0.1, disabled=is_disabled)
process_temp = st.sidebar.slider('Process temperature [K]', 305.0, 315.0, 310.1, 0.1, disabled=is_disabled)
rotational_speed = st.sidebar.slider('Rotational speed [rpm]', 1100, 3000, 1500, disabled=is_disabled)
torque = st.sidebar.slider('Torque [Nm]', 3.0, 80.0, 40.5, 0.1, disabled=is_disabled)
tool_wear = st.sidebar.slider('Tool wear [min]', 0, 260, 108, disabled=is_disabled)

# --- Main Dashboard ---
st.title("⚙️ Real-Time Predictive Maintenance Dashboard")
status_placeholder = st.empty()
charts_placeholder = st.empty()

# --- Main Loop ---
if st.session_state.simulation_mode == 'replay':
    if st.session_state.replay_index < len(st.session_state.replay_df):
        current_data = st.session_state.replay_df.iloc[st.session_state.replay_index]
        input_df = pd.DataFrame([current_data[features]])
        st.session_state.replay_index += 1
        sleep_time = 0.2
        
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        new_row_data = input_df.iloc[0].to_dict()
        new_row_data['Time'] = pd.Timestamp.now()
        new_row = pd.DataFrame([new_row_data])
        st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True).tail(100)

        with status_placeholder.container():
            udi_being_replayed = st.session_state.replay_df['UDI'].iloc[0]
            st.header(f"Live System Status (Replaying Machine UDI: {udi_being_replayed})")
            col1, col2 = st.columns([1, 2])
            with col1:
                if prediction[0] == 0:
                    st.success("✅ NORMAL")
                else:
                    st.error("🚨 FAILURE IMMINENT")
                failure_prob = prediction_proba[0][1] * 100
                st.metric(label="Failure Probability", value=f"{failure_prob:.1f}%")
                st.progress(int(failure_prob))
            with col2:
                st.dataframe(input_df)

        with charts_placeholder.container():
            st.header("Live Sensor Data")
            c1, c2 = st.columns(2)
            with c1:
                fig_temp = px.line(st.session_state.history, x='Time', y=['Air temperature [K]', 'Process temperature [K]'], title="Temperature Trend")
                st.plotly_chart(fig_temp, use_container_width=True)
                fig_torque = px.line(st.session_state.history, x='Time', y='Torque [Nm]', title="Torque Trend")
                st.plotly_chart(fig_torque, use_container_width=True)
            with c2:
                fig_speed = px.line(st.session_state.history, x='Time', y='Rotational speed [rpm]', title="Rotational Speed Trend")
                st.plotly_chart(fig_speed, use_container_width=True)
                fig_wear = px.line(st.session_state.history, x='Time', y='Tool wear [min]', title="Tool Wear Trend")
                st.plotly_chart(fig_wear, use_container_width=True)
        
        time.sleep(sleep_time)
        st.rerun()
    else:
        st.success("Scenario replay finished.")
        st.session_state.simulation_mode = 'inactive'
        time.sleep(2)
        st.rerun()

else: # --- Inactive/Manual Mode ---
    st.header("Manual 'What-If' Analysis")
    st.markdown("Use the sliders in the sidebar for manual analysis, or start a live simulation.")
    input_data = pd.DataFrame({'Air temperature [K]': [air_temp], 'Process temperature [K]': [process_temp], 'Rotational speed [rpm]': [rotational_speed], 'Torque [Nm]': [torque], 'Tool wear [min]': [tool_wear]})[features]
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Prediction")
        if prediction[0] == 0:
            st.success("✅ NORMAL")
        else:
            st.error("🚨 FAILURE IMMINENT")
    with col2:
        st.subheader("Probability")
        failure_prob = prediction_proba[0][1] * 100
        st.metric(label="Probability of Failure", value=f"{failure_prob:.2f}%")
        st.progress(int(failure_prob))
    st.subheader("Model Insights: Feature Importance")
    feature_importances = pd.DataFrame(model.feature_importances_, index=features, columns=['importance']).sort_values('importance', ascending=False)
    fig = px.bar(feature_importances, x=feature_importances.index, y='importance', title='Feature Importance for Failure Prediction')
    st.plotly_chart(fig, use_container_width=True)
 
