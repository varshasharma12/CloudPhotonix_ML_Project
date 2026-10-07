# Automated Customer Churn Prediction & Monitoring Platform

## Project Overview

This project is an Automated Machine Learning platform that predicts whether a customer is likely to churn.

The system automates the complete ML workflow:

Dataset Generation → Data Validation → Model Training → Model Evaluation → Best Model Selection → Prediction → Monitoring

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- SQLite
- Joblib
- Matplotlib
- Seaborn

## Machine Learning Models

The system trains and compares three classification models:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The best model is automatically selected using F1 Score.

## Features

- Automated dataset generation
- Data validation
- Missing-value and duplicate checking
- Data preprocessing
- Categorical feature encoding
- Multiple ML model training
- Accuracy, Precision, Recall and F1 Score
- Confusion Matrix
- Automatic best-model selection
- Model saving using Joblib
- Customer churn prediction
- Churn probability
- Prediction history
- SQLite database
- Monitoring dashboard
- Model performance visualization

## Project Structure


CloudPhotonix_ML_Project
│
├── data
│   └── customer_churn.csv
│
├── models
│   └── best_model.pkl
│
├── reports
│   ├── data_validation_report.txt
│   ├── model_metrics.csv
│   └── confusion_matrix.csv
│
├── src
│   ├── generate_dataset.py
│   ├── data_validation.py
│   ├── train.py
│   └── monitoring.py
│
├── app.py
├── run_pipeline.py
├── predictions.db
├── requirements.txt
└── README.md