import subprocess
import sys

print("===================================")
print(" AUTOMATED ML PIPELINE STARTED")
print("===================================")

print("\n[1] Generating Dataset...")
subprocess.run(
    [sys.executable, "src/generate_dataset.py"],
    check=True
)

print("\n[2] Validating Dataset...")
subprocess.run(
    [sys.executable, "src/data_validation.py"],
    check=True
)

print("\n[3] Training ML Models...")
subprocess.run(
    [sys.executable, "src/train.py"],
    check=True
)

print("\n===================================")
print(" AUTOMATED PIPELINE COMPLETED")
print("===================================")

print("\nBest model has been saved in:")
print("models/best_model.pkl")