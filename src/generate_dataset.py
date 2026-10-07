import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

age = np.random.randint(18, 70, n)
tenure = np.random.randint(1, 73, n)
monthly_charges = np.random.randint(300, 1500, n)
support_calls = np.random.randint(0, 11, n)

contract = np.random.choice(
    ["Monthly", "Yearly", "Two-Year"],
    n,
    p=[0.55, 0.30, 0.15]
)

internet_service = np.random.choice(
    ["DSL", "Fiber", "None"],
    n,
    p=[0.35, 0.50, 0.15]
)

churn_probability = (
    0.15
    + (monthly_charges > 900) * 0.20
    + (tenure < 12) * 0.25
    + (support_calls > 5) * 0.20
    + (contract == "Monthly") * 0.15
    + (internet_service == "Fiber") * 0.05
)

churn_probability = np.clip(churn_probability, 0, 0.95)

churn = np.random.binomial(1, churn_probability, n)

df = pd.DataFrame({
    "customer_id": range(1001, 1001 + n),
    "age": age,
    "tenure": tenure,
    "monthly_charges": monthly_charges,
    "support_calls": support_calls,
    "contract": contract,
    "internet_service": internet_service,
    "churn": churn
})

df["churn"] = df["churn"].map({0: "No", 1: "Yes"})

df.to_csv("data/customer_churn.csv", index=False)

print("Dataset generated successfully!")
print(f"Total records: {len(df)}")
print(df.head())