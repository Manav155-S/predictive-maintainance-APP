import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("🔧 Predictive Maintenance & Anomaly Detection")

# Upload the dataset
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # Anomaly Detection using Z-score
    st.subheader("Anomaly Detection Plot (Torque vs RPM)")
    df["zscore"] = (df["Torque [Nm]"] - df["Torque [Nm]"].mean()) / df["Torque [Nm]"].std()
    anomalies = df[np.abs(df["zscore"]) > 2]

    fig, ax = plt.subplots()
    ax.scatter(df["Rotational speed [rpm]"], df["Torque [Nm]"], color="blue", label="Normal")
    ax.scatter(anomalies["Rotational speed [rpm]"], anomalies["Torque [Nm]"], color="red", label="Anomalies")
    ax.set_xlabel("RPM")
    ax.set_ylabel("Torque")
    ax.legend()
    st.pyplot(fig)
