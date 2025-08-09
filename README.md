# Predictive Maintenance and Anomaly Detection Command Center

This project is an end-to-end web application that simulates a real-world command center for monitoring an entire fleet of industrial machines. It's a professional-grade tool designed to demonstrate how a combination of **predictive maintenance** and **anomaly detection** can prevent costly equipment failures.

The system uses a high-performance **XGBoost** model to analyze simulated fleet data, **predict** the specific type of failure, and **detect the anomalies** causing it. It provides deep, actionable insights to different user personas—from the factory manager to the on-the-ground engineer—with the core objective of preventing unplanned downtime.

---


## 🛠️ Key Features

* **Fleet-Wide Monitoring:** Simulates an entire fleet of 100 machines, providing a high-level "command center" view of overall operational health.
* **Actionable "At-Risk" Report:** The "Scan Fleet" function runs a diagnostic on the entire fleet and generates a stable, persistent report of only the machines predicted to be in a failure state, including the suspected root cause for each.
* **Interactive Machine Deep Dive:** Select any at-risk machine directly from the report to instantly view its specific sensor data and a detailed root cause analysis, showing exactly how its parameters deviate from a healthy machine.
* **Advanced Analytical Tools:** A dedicated tab for engineers, featuring:
    * **Failure Signature Analysis:** A tool to visually compare the average sensor profile of a specific failure type against a healthy machine.
    * **Manual 'What-If' Analysis:** A sensitivity analysis tool for engineers to test the model's response to specific, manually-set sensor conditions.
* **High-Performance ML Model:** The backend is powered by a tuned **XGBoost Classifier**, optimized through hyperparameter tuning (`RandomizedSearchCV`) to ensure high accuracy and reliability.

---

## 🏛️ System Architecture & Personas

The application is designed to serve two key user personas, mirroring a real-world industrial workflow:

1.  **The Manager (Strategic View):** The **"Fleet Overview"** tab is their primary tool. It provides high-level KPIs like Overall Fleet Health and a clear, concise list of at-risk machines, enabling quick, strategic decisions about where to allocate maintenance resources.

2.  **The Engineer (Diagnostic View):** The **"Machine Deep Dive"** and **"Analytical Tools"** tabs are their domain. These tools allow for detailed, low-level investigation into *why* a machine is failing, enabling faster diagnostics and more effective repairs.

---

## 🚀 Deployment

This application is deployed on Streamlit Community Cloud and is publicly accessible.

**Live App URL:** `(https://predictive-maintainance-app-l84cxus3lavjcp3uflvkbn.streamlit.app/)`

---
