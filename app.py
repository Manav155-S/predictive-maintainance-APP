import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import warnings
import re

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# --- App Configuration ---
st.set_page_config(
    page_title="Predictive Maintenance & Anomaly Detection",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Model Training & Loading Functions (Cached for performance) ---
@st.cache_resource
def train_and_save_model(data_path='predictive_maintenance.csv', model_path='xgb_model_tuned.joblib'):
    st.info(f"Training new XGBoost model with Hyperparameter Tuning...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"Error: Dataset '{data_path}' not found. Please place it in the same directory.")
        return None, None, None

    failure_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    df['Failure Type'] = 0
    for i, col in enumerate(failure_cols):
        df.loc[df[col] == 1, 'Failure Type'] = i + 1

    original_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    target = 'Failure Type'

    if not all(col in df.columns for col in original_features + [target]):
        st.error("Dataset is missing required columns.")
        return None, None, None

    X = df[original_features]
    y = df[target]

    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    X.columns = [regex.sub("_", col).replace(" ", "_") for col in X.columns]
    cleaned_features = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    param_grid = {
        'n_estimators': [100, 200], 'max_depth': [5, 7], 'learning_rate': [0.1, 0.2],
        'subsample': [0.8, 0.9], 'colsample_bytree': [0.8, 0.9]
    }

    xgb = XGBClassifier(objective='multi:softmax', use_label_encoder=False, eval_metric='mlogloss')
    random_search = RandomizedSearchCV(xgb, param_distributions=param_grid, n_iter=5, scoring='accuracy', n_jobs=-1, cv=3, verbose=1, random_state=42)
    
    with st.spinner("Finding best model parameters... (This may take a minute on first run)"):
        random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    st.success(f"Best parameters found: {random_search.best_params_}")

    accuracy = accuracy_score(y_test, best_model.predict(X_test))
    st.success(f"Tuned XGBoost model training complete. Accuracy: {accuracy:.2%}")

    joblib.dump({'model': best_model, 'features': cleaned_features, 'dataframe': df}, model_path)
    st.success(f"New tuned model saved to '{model_path}'")
    return best_model, cleaned_features, df

@st.cache_resource
def load_model_and_data(model_path='xgb_model_tuned.joblib'):
    if not os.path.exists(model_path): return None, None, None
    try:
        saved_dict = joblib.load(model_path)
        if 'dataframe' not in saved_dict: return None, None, None
        return saved_dict['model'], saved_dict['features'], saved_dict['dataframe']
    except Exception: return None, None, None


# --- Main Application Logic ---
MODEL_FILE = 'xgb_model_tuned.joblib'
DATA_FILE = 'predictive_maintenance.csv'

model, features, df = load_model_and_data(MODEL_FILE)
if model is None:
    model, features, df = train_and_save_model(DATA_FILE, MODEL_FILE)
    if model is None: st.stop()

original_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
normal_averages = df[df['Failure Type'] == 0][original_features].mean()

if 'fleet_state' not in st.session_state:
    fleet_sample = df.sample(100).set_index('UDI')
    fleet_sample['Health'] = 100.0
    st.session_state.fleet_state = fleet_sample
    st.session_state.at_risk_machines = pd.DataFrame()
    st.session_state.selected_machine_udi = None

failure_map = {
    0: '✅ NORMAL', 1: '🚨 TOOL WEAR FAILURE', 2: '🚨 HEAT DISSIPATION FAILURE',
    3: '🚨 POWER FAILURE', 4: '🚨 OVERSTRAIN FAILURE', 5: '🚨 RANDOM FAILURE'
}

# --- Sidebar Controls ---
st.sidebar.title("Control Panel")
st.sidebar.header("Fleet Monitoring")

if st.sidebar.button("▶️ Advance Time & Scan Fleet", type="primary"):
    fleet_state = st.session_state.fleet_state
    
    fleet_state['Tool wear [min]'] += 0.1
    noise = np.random.randn(fleet_state.shape[0], len(original_features)) * [0.05, 0.05, 2, 0.2, 0.01]
    fleet_state[original_features] += noise

    degrading_machines = fleet_state[fleet_state['Health'] < 80].index
    fleet_state.loc[degrading_machines, 'Torque [Nm]'] += np.random.uniform(0.5, 1.5, len(degrading_machines))
    fleet_state.loc[degrading_machines, 'Process temperature [K]'] += np.random.uniform(0.1, 0.3, len(degrading_machines))

    if np.random.rand() < 0.2:
        machine_to_stress = np.random.choice(fleet_state.index)
        fleet_state.loc[machine_to_stress, 'Health'] -= 25
        fleet_state.loc[machine_to_stress, 'Torque [Nm]'] *= 1.2
        st.sidebar.warning(f"Stress event on Machine {machine_to_stress}!")

    fleet_state[original_features] = fleet_state[original_features].clip(lower=0)
    st.session_state.fleet_state = fleet_state

    fleet_to_predict = fleet_state[original_features]
    fleet_cleaned = fleet_to_predict.copy()
    fleet_cleaned.columns = features
    
    predictions = model.predict(fleet_cleaned)
    probabilities = model.predict_proba(fleet_cleaned)

    at_risk_indices = np.where(predictions > 0)[0]
    
    if len(at_risk_indices) > 0:
        at_risk_data = []
        for idx in at_risk_indices:
            machine_udi = fleet_state.index[idx]
            pred_class = predictions[idx]
            confidence = probabilities[idx][pred_class]
            current_readings = fleet_state.iloc[idx]
            deviations = ((current_readings[original_features] - normal_averages) / normal_averages) * 100
            root_cause = deviations.abs().idxmax()
            at_risk_data.append({
                "UDI": machine_udi,
                "Predicted Failure": failure_map[pred_class],
                "Confidence": f"{confidence:.1%}",
                "Suspected Root Cause": root_cause
            })
        st.session_state.at_risk_machines = pd.DataFrame(at_risk_data)
        st.session_state.selected_machine_udi = st.session_state.at_risk_machines['UDI'].iloc[0]
    else:
        st.session_state.at_risk_machines = pd.DataFrame()
        st.session_state.selected_machine_udi = None


st.sidebar.divider()
st.sidebar.header("Advanced Tools")
if st.sidebar.button("♻️ Retrain Model"):
    if os.path.exists(MODEL_FILE): os.remove(MODEL_FILE)
    st.cache_resource.clear()
    st.rerun()

# --- Main UI with Tabs ---
st.title("🏭 Predictive Maintenance & Anomaly Detection")
tab1, tab2, tab3 = st.tabs(["Fleet Overview", "Machine Deep Dive", "Analytical Tools"])

with tab1:
    st.header("🔴 At-Risk Machines Report")
    col1, col2 = st.columns(2)
    fleet_health = 100 - (len(st.session_state.at_risk_machines) / len(st.session_state.fleet_state) * 100)
    col1.metric("Overall Fleet Health", f"{fleet_health:.1f}%")
    col2.metric("Machines At Risk", len(st.session_state.at_risk_machines))

    if not st.session_state.at_risk_machines.empty:
        st.dataframe(st.session_state.at_risk_machines, use_container_width=True)
        at_risk_udis = st.session_state.at_risk_machines['UDI'].tolist()
        selected_machine = st.selectbox("Select an at-risk machine to analyze:", options=at_risk_udis)
        if selected_machine:
            st.session_state.selected_machine_udi = selected_machine
            st.info(f"Machine {selected_machine} selected. Go to the 'Machine Deep Dive' tab for detailed analysis.")
    else:
        st.success("✅ All systems normal. No machine failures predicted in the last scan.")

    st.header("Fleet-Wide Sensor Distribution")
    fleet_state_display = st.session_state.fleet_state[original_features]
    col1, col2 = st.columns(2)
    with col1:
        fig_temp = px.histogram(fleet_state_display, x='Air temperature [K]', title='Air Temperature Distribution')
        st.plotly_chart(fig_temp, use_container_width=True)
        fig_torque = px.histogram(fleet_state_display, x='Torque [Nm]', title='Torque Distribution')
        st.plotly_chart(fig_torque, use_container_width=True)
    with col2:
        fig_speed = px.histogram(fleet_state_display, x='Rotational speed [rpm]', title='Rotational Speed Distribution')
        st.plotly_chart(fig_speed, use_container_width=True)
        fig_wear = px.histogram(fleet_state_display, x='Tool wear [min]', title='Tool Wear Distribution')
        st.plotly_chart(fig_wear, use_container_width=True)

with tab2:
    st.header("🔧 Machine Deep Dive Analysis")
    if st.session_state.selected_machine_udi:
        st.subheader(f"Showing details for Machine UDI: {st.session_state.selected_machine_udi}")
        machine_data = st.session_state.fleet_state.loc[st.session_state.selected_machine_udi]
        
        st.write("#### Current Sensor Readings")
        st.dataframe(pd.DataFrame(machine_data[original_features]).T)

        st.write("#### Root Cause Analysis (Comparison to Normal)")
        deviations = ((machine_data[original_features] - normal_averages) / normal_averages) * 100
        most_significant_feature = deviations.abs().idxmax()
        
        for feature, dev in deviations.items():
            st.metric(label=f"{feature} Deviation", value=f"{dev:.1f}%",
                      delta=f"{'Higher' if dev > 0 else 'Lower'} than average",
                      delta_color="inverse" if dev > 0 else "normal")
        st.info(f"**Conclusion:** The most significant deviation from normal was in **{most_significant_feature}**, which is the likely root cause for this machine's at-risk status.")

    else:
        st.info("Scan the fleet in the 'Fleet Overview' tab. If any machines are at risk, you can select one from the dropdown to analyze here.")

with tab3:
    st.header("🔬 Analytical Tools")

    # --- NEW: Manual "What-If" Analysis for Engineers ---
    st.subheader("⚙️ Manual 'What-If' Analysis")
    st.markdown("This tool is for engineers to perform sensitivity analysis and test the model's response to specific conditions.")
    
    col1, col2 = st.columns(2)
    with col1:
        air_temp = st.slider(original_features[0], 295.0, 305.0, 300.1, 0.1)
        process_temp = st.slider(original_features[1], 305.0, 315.0, 310.1, 0.1)
        rotational_speed = st.slider(original_features[2], 1100, 3000, 1500)
    with col2:
        torque = st.slider(original_features[3], 3.0, 80.0, 40.5, 0.1)
        tool_wear = st.slider(original_features[4], 0, 260, 108)

    input_data_dict = dict(zip(original_features, [air_temp, process_temp, rotational_speed, torque, tool_wear]))
    input_df_original_names = pd.DataFrame([input_data_dict])
    input_df_cleaned_names = input_df_original_names.copy()
    input_df_cleaned_names.columns = features

    prediction = model.predict(input_df_cleaned_names)
    prediction_proba = model.predict_proba(input_df_cleaned_names)
    
    is_extreme = False
    extreme_conditions = []
    if torque >= 75.0 and rotational_speed <= 1300: is_extreme = True; extreme_conditions.append("High Torque at Low Speed")
    if tool_wear >= 240: is_extreme = True; extreme_conditions.append("Critical Tool Wear")
    if process_temp >= 314.0 and air_temp >= 304.0: is_extreme = True; extreme_conditions.append("Extreme Heat")

    st.write("##### System Diagnosis:")
    if is_extreme:
        st.error(f"🚨 CRITICAL FAILURE (Rule-based override)\n\n**Reason(s):** {', '.join(extreme_conditions)}")
    else:
        predicted_status = failure_map[prediction[0]]
        confidence = prediction_proba[0][prediction[0]] * 100
        if prediction[0] == 0:
            st.success(f"{predicted_status} (Confidence: {confidence:.1f}%)")
        else:
            st.error(f"{predicted_status} (Confidence: {confidence:.1f}%)")

    st.divider()
    st.subheader("Failure Signature Analysis")
    st.markdown("This tool shows the average sensor profile for a specific failure type compared to a healthy machine.")
    
    selected_failure_type = st.selectbox("Select failure type to analyze", options=list(failure_map.keys())[1:], format_func=lambda x: failure_map[x])
    
    failure_df = df[df['Failure Type'] == selected_failure_type]
    if not failure_df.empty:
        failure_averages = failure_df[original_features].mean()
        
        comparison_df = pd.DataFrame({'Normal': normal_averages, 'Failure': failure_averages})
        comparison_df = comparison_df.reset_index().rename(columns={'index': 'Feature'})
        comparison_df = pd.melt(comparison_df, id_vars='Feature', var_name='State', value_name='Average Value')
        
        fig = px.bar(comparison_df, x='Feature', y='Average Value', color='State', barmode='group',
                     title=f'Failure Signature vs. Normal Operation for {failure_map[selected_failure_type]}')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No historical examples found for this failure type.")

    st.divider()
    st.subheader("Model Insights: Global Feature Importance")
    try:
        feature_importances_data = model.get_booster().get_score(importance_type='weight')
        if feature_importances_data:
            feature_importances = pd.DataFrame(list(feature_importances_data.items()), columns=['feature', 'importance']).sort_values('importance', ascending=False)
            fig = px.bar(feature_importances, x='feature', y='importance', title='Feature Importance for Failure Prediction (XGBoost)')
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning("Could not display feature importance for XGBoost model.")
