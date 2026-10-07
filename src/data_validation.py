import pandas as pd
from pathlib import Path
import sys

DATA_PATH = "data/customer_churn.csv"
REPORT_PATH = "reports/data_validation_report.txt"

required_columns = [
    "customer_id",
    "age",
    "tenure",
    "monthly_charges",
    "support_calls",
    "contract",
    "internet_service",
    "churn"
]

Path("reports").mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

errors = []
warnings = []

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    errors.append(f"Missing columns: {missing_columns}")

missing_values = int(df.isnull().sum().sum())

if missing_values > 0:
    warnings.append(f"Missing values found: {missing_values}")

duplicates = int(df.duplicated().sum())

if duplicates > 0:
    warnings.append(f"Duplicate rows found: {duplicates}")

if "age" in df.columns:
    invalid_age = int(
        ((df["age"] < 18) | (df["age"] > 100)).sum()
    )
    if invalid_age > 0:
        errors.append(f"Invalid age records: {invalid_age}")

if "tenure" in df.columns:
    invalid_tenure = int(
        ((df["tenure"] < 1) | (df["tenure"] > 100)).sum()
    )
    if invalid_tenure > 0:
        errors.append(f"Invalid tenure records: {invalid_tenure}")

if "monthly_charges" in df.columns:
    invalid_charges = int(
        (df["monthly_charges"] < 0).sum()
    )
    if invalid_charges > 0:
        errors.append(f"Invalid monthly charges: {invalid_charges}")

if "churn" in df.columns:
    invalid_churn = int(
        (~df["churn"].isin(["Yes", "No"])).sum()
    )
    if invalid_churn > 0:
        errors.append(f"Invalid churn values: {invalid_churn}")

report = []

report.append("DATA VALIDATION REPORT")
report.append("======================")
report.append(f"Total Records: {len(df)}")
report.append(f"Total Columns: {len(df.columns)}")
report.append(f"Missing Values: {missing_values}")
report.append(f"Duplicate Rows: {duplicates}")

if "churn" in df.columns:
    report.append("\nChurn Distribution:")
    report.append(str(df["churn"].value_counts()))

if warnings:
    report.append("\nWARNINGS:")
    report.extend(warnings)

if errors:
    report.append("\nERRORS:")
    report.extend(errors)
    status = "FAILED"
else:
    report.append("\nNo critical data errors found.")
    status = "PASSED"

report.append(f"\nVALIDATION STATUS: {status}")

with open(REPORT_PATH, "w") as file:
    file.write("\n".join(report))

print("\n==============================")
print("DATA VALIDATION")
print("==============================")
print(f"Records checked: {len(df)}")

if warnings:
    print("Warnings found:", len(warnings))
else:
    print("Warnings: 0")

if errors:
    print("Errors found:", len(errors))
    print("Validation FAILED")
    sys.exit(1)
else:
    print("Errors: 0")
    print("Validation PASSED")

print(f"Report saved: {REPORT_PATH}")