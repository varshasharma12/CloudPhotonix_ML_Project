import sqlite3
import pandas as pd


def get_prediction_history():

    conn = sqlite3.connect("predictions.db")

    history = pd.read_sql_query(
        """
        SELECT *
        FROM predictions
        ORDER BY id DESC
        """,
        conn
    )

    conn.close()

    return history


def get_monitoring_metrics():

    history = get_prediction_history()

    if history.empty:
        return {
            "total_predictions": 0,
            "churn_predictions": 0,
            "non_churn_predictions": 0,
            "average_probability": 0
        }

    total_predictions = len(history)

    churn_predictions = len(
        history[history["prediction"] == "WILL CHURN"]
    )

    non_churn_predictions = len(
        history[history["prediction"] == "WILL NOT CHURN"]
    )

    average_probability = history["probability"].mean() * 100

    return {
        "total_predictions": total_predictions,
        "churn_predictions": churn_predictions,
        "non_churn_predictions": non_churn_predictions,
        "average_probability": average_probability
    }