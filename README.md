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

## Project Structure

Nids-project/
│
├── app.py              Streamlit web application for intrusion detection dashboard
├── nids.py             Model training and evaluation script (ML pipeline)
├── KDDTrain+.txt       NSL-KDD dataset used for training the model
├── model.pkl           Saved trained XGBoost model
├── README.md           Project documentation and usage guide


---

## How It Works

1. Input Dataset (KDDTrain+.txt)
-> Data Preprocessing (Encoding + Cleaning)
-> Feature Selection
-> Model Training (Decision Tree + XGBoost)
-> Best Model Saved (model.pkl)
-> Streamlit Web Application (app.py)
-> Prediction Output (Normal / Attack)
-> Visualization Dashboard


---

## System Architecture

1. User uploads dataset
-> Streamlit interface (app.py) processes input
-> Pretrained model (model.pkl) is loaded
-> XGBoost prediction engine is executed
-> Results are classified as Normal or Attack
-> Visualization layer displays insights


---

## Results Preview

(Add screenshots of your Streamlit application here)

Recommended sections for screenshots:
- Raw Data Preview
- Detection Summary Dashboard
- Traffic Distribution Pie Chart
- Suspicious Records Table


---

## Workflow Explanation

1. User uploads a network traffic dataset
-> Data is cleaned and preprocessed
-> Categorical values are encoded
-> Model processes feature set
-> Each record is classified as:
   -> Normal
   -> Attack
-> Confidence scores are generated
-> Visual analytics are displayed in the dashboard


---

## Key Highlights

- Real-time intrusion detection system
- Machine learning-based binary classification
- Interactive Streamlit dashboard
- Lightweight and deployable model
- Built using NSL-KDD dataset


---

## Deployment Options

The application can be deployed using:

- Streamlit Cloud
- Render
- Hugging Face Spaces

Run locally using:

streamlit run app.py


---

## Future Improvements

1. Real-time network packet capture integration
2. Deep learning-based intrusion detection models
3. REST API-based deployment for external systems
4. Cloud deployment for scalability
5. Automated alert generation system

License:
This project is intended for academic and educational purposes.
