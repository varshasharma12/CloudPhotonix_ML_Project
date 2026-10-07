import streamlit as st
import pandas as pd
import joblib
import sqlite3

from src.monitoring import (
    get_prediction_history,
    get_monitoring_metrics
)


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


st.title("Customer Churn Prediction & Monitoring Platform")
st.write("Automated Machine Learning Prediction System")


model = joblib.load("models/best_model.pkl")


st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Predict Churn",
        "Model Performance",
        "Prediction History"
    ]
)


if page == "Dashboard":

    st.header("Monitoring Dashboard")

    metrics = get_monitoring_metrics()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Predictions",
            metrics["total_predictions"]
        )

    with col2:
        st.metric(
            "Churn Predictions",
            metrics["churn_predictions"]
        )

    with col3:
        st.metric(
            "Non-Churn Predictions",
            metrics["non_churn_predictions"]
        )

    with col4:
        st.metric(
            "Average Churn Probability",
            f"{metrics['average_probability']:.2f}%"
        )

    history = get_prediction_history()

    if not history.empty:

        st.subheader("Prediction Distribution")

        distribution = history["prediction"].value_counts()

        st.bar_chart(distribution)

        st.subheader("Recent Predictions")

        st.dataframe(
            history.head(10),
            use_container_width=True
        )

    else:

        st.info(
            "No predictions available yet. "
            "Go to Predict Churn to create a prediction."
        )


elif page == "Predict Churn":

    st.header("Customer Churn Prediction")

    st.write(
        "Enter customer information to predict churn probability."
    )

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=1,
            max_value=100,
            value=12
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=5000.0,
            value=700.0
        )

    with col2:

        support_calls = st.number_input(
            "Support Calls",
            min_value=0,
            max_value=50,
            value=2
        )

        contract = st.selectbox(
            "Contract",
            [
                "Monthly",
                "Yearly",
                "Two-Year"
            ]
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber",
                "None"
            ]
        )

    if st.button(
        "Predict Churn",
        type="primary"
    ):

        input_data = pd.DataFrame({
            "age": [age],
            "tenure": [tenure],
            "monthly_charges": [monthly_charges],
            "support_calls": [support_calls],
            "contract": [contract],
            "internet_service": [internet_service]
        })

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(
            input_data
        )[0][1]

        if prediction == 1:

            result = "WILL CHURN"

            st.error(result)

        else:

            result = "WILL NOT CHURN"

            st.success(result)

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

        conn = sqlite3.connect(
            "predictions.db"
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_age INTEGER,
                tenure INTEGER,
                monthly_charges REAL,
                support_calls INTEGER,
                contract TEXT,
                internet_service TEXT,
                prediction TEXT,
                probability REAL,
                prediction_time TEXT
            )
            """
        )

        from datetime import datetime

        conn.execute(
            """
            INSERT INTO predictions
            (
                customer_age,
                tenure,
                monthly_charges,
                support_calls,
                contract,
                internet_service,
                prediction,
                probability,
                prediction_time
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                age,
                tenure,
                monthly_charges,
                support_calls,
                contract,
                internet_service,
                result,
                probability,
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )
        )

        conn.commit()
        conn.close()

        st.success(
            "Prediction saved successfully!"
        )


elif page == "Model Performance":

    st.header("Model Performance")

    try:

        metrics_df = pd.read_csv(
            "reports/model_metrics.csv"
        )

        st.dataframe(
            metrics_df,
            use_container_width=True
        )

        st.subheader("Model Comparison")

        chart_data = metrics_df.set_index(
            "Model"
        )[
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ]
        ]

        st.bar_chart(chart_data)

        st.subheader("Confusion Matrix")

        confusion_matrix = pd.read_csv(
            "reports/confusion_matrix.csv",
            index_col=0
        )

        st.dataframe(
            confusion_matrix,
            use_container_width=True
        )

    except FileNotFoundError:

        st.warning(
            "Model reports not found. "
            "Run the ML pipeline first."
        )


elif page == "Prediction History":

    st.header("Prediction History")

    history = get_prediction_history()

    if not history.empty:

        st.dataframe(
            history,
            use_container_width=True
        )

    else:

        st.info(
            "No prediction history available."
        )