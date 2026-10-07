import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


df = pd.read_csv("data/customer_churn.csv")

X = df.drop(columns=["customer_id", "churn"])
y = df["churn"].map({"No": 0, "Yes": 1})


categorical_features = [
    "contract",
    "internet_service"
]

numeric_features = [
    "age",
    "tenure",
    "monthly_charges",
    "support_calls"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


results = []


for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1 Score": round(f1 * 100, 2)
    })

    print("\n" + name)
    print("Accuracy :", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall   :", round(recall * 100, 2), "%")
    print("F1 Score :", round(f1 * 100, 2), "%")


results_df = pd.DataFrame(results)

os.makedirs("reports", exist_ok=True)

results_df.to_csv(
    "reports/model_metrics.csv",
    index=False
)


best_model_name = results_df.loc[
    results_df["F1 Score"].idxmax(),
    "Model"
]


best_model = None

for name, model in models.items():

    if name == best_model_name:

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(X_train, y_train)

        best_model = pipeline

        predictions = best_model.predict(X_test)

        matrix = confusion_matrix(
            y_test,
            predictions
        )

        confusion_df = pd.DataFrame(
            matrix,
            index=["Actual No", "Actual Yes"],
            columns=["Predicted No", "Predicted Yes"]
        )

        confusion_df.to_csv(
            "reports/confusion_matrix.csv"
        )


joblib.dump(
    best_model,
    "models/best_model.pkl"
)


print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(results_df.to_string(index=False))

print("\n==============================")
print("BEST MODEL:", best_model_name)
print("==============================")

print("\nReports generated:")
print("reports/model_metrics.csv")
print("reports/confusion_matrix.csv")

print("\nModel saved successfully!")