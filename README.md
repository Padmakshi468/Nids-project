Network Intrusion Detection System (NIDS)
Overview

This project implements a machine learning-based Network Intrusion Detection System using the NSL-KDD dataset. The system classifies network traffic as normal or malicious using trained models and provides an interactive web interface for analysis and visualization.

The primary goal is to detect anomalous network behavior efficiently and demonstrate a practical application of machine learning in cybersecurity.

Features:
Classification of network traffic into normal and attack categories
Real-time prediction through a web-based interface
Confidence score for each prediction
Visualization of traffic distribution
Filtering of suspicious network records
Supports CSV and TXT network datasets
Machine Learning Approach

Dataset: NSL-KDD

Problem Type: Binary classification

Label Mapping:

Normal → 0
Attack → 1

Models Used:

Decision Tree Classifier (baseline)
XGBoost Classifier (final model)

Preprocessing:

Label encoding for categorical features
Handling missing and invalid values
Feature scaling not required (tree-based models)
System Workflow
Input network traffic dataset
Data preprocessing and feature encoding
Model inference using trained XGBoost classifier
Prediction generation (Normal / Attack)
Visualization of results in dashboard

Tech Stack:
Python,
Streamlit,
Pandas, NumPy
Scikit-learn,
XGBoost,
Matplotlib,
Joblib

Installation:
Clone the repository:
git clone https://github.com/Padmakshi468/Nids-project.git
cd Nids-project

Install dependencies:
pip install -r requirements.txt

If requirements file is not available:
pip install streamlit pandas numpy scikit-learn xgboost matplotlib joblib

Running the Application:
streamlit run app.py

Project Structure:
Nids-project/
│
├── app.py                 # Streamlit application
├── model.pkl              # Trained ML model
├── KDDTrain+.txt          # Dataset
├── README.md              # Documentation

Results:
The system successfully classifies network traffic into normal and attack categories.
XGBoost provides higher accuracy compared to Decision Tree.
The dashboard provides real-time visualization of detection results.

Future Improvements:
Integration with real-time packet capture systems
Deep learning-based intrusion detection models
API-based deployment for external integration
Automated alert generation system
Cloud deployment for scalability

License:
This project is intended for academic and educational purposes.
