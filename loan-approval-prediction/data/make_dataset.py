"""
Generates data/loan_data.csv — a dataset matching the schema described in
this project's README (614 rows x 13 columns, same columns as the Kaggle
"Loan Prediction" dataset). Ground-truth relationships are built in (credit
history, income, education, etc. genuinely affect approval) plus noise, so
the model in src/model.py has real signal to learn from.

Run this once before src/model.py if data/loan_data.csv is missing:
    python3 data/make_dataset.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 614  # matches the row count stated in README.md

gender = np.random.choice(["Male", "Female"], size=N, p=[0.78, 0.22])
married = np.random.choice(["Yes", "No"], size=N, p=[0.65, 0.35])
dependents = np.random.choice(["0", "1", "2", "3+"], size=N, p=[0.55, 0.2, 0.16, 0.09])
education = np.random.choice(["Graduate", "Not Graduate"], size=N, p=[0.78, 0.22])
self_employed = np.random.choice(["Yes", "No"], size=N, p=[0.14, 0.86])
property_area = np.random.choice(["Urban", "Semiurban", "Rural"], size=N, p=[0.38, 0.38, 0.24])

applicant_income = np.random.gamma(shape=5, scale=1200, size=N).round(0) + 1500
coapplicant_income = np.where(
    married == "Yes",
    np.random.gamma(shape=3, scale=800, size=N).round(0),
    0
)
loan_amount = (
    (applicant_income + coapplicant_income) * np.random.uniform(0.08, 0.22, size=N)
).round(-2) / 1000
loan_amount = loan_amount.round(0)
loan_amount = np.clip(loan_amount, 9, 600)

loan_term = np.random.choice([360, 180, 120, 300, 60, 84], size=N, p=[0.72, 0.1, 0.06, 0.06, 0.03, 0.03])
credit_history = np.random.choice([1.0, 0.0], size=N, p=[0.82, 0.18])


def add_missing(arr, frac):
    arr = arr.astype(object)
    idx = np.random.choice(len(arr), size=int(len(arr) * frac), replace=False)
    arr[idx] = np.nan
    return arr


gender = add_missing(gender, 0.02)
married = add_missing(married, 0.005)
dependents = add_missing(dependents, 0.025)
self_employed = add_missing(self_employed, 0.05)
loan_amount_missing = add_missing(loan_amount.copy(), 0.035)
loan_term_missing = add_missing(loan_term.copy(), 0.023)
credit_history_missing = add_missing(credit_history.copy(), 0.08)

total_income = applicant_income + coapplicant_income
income_to_loan = total_income / (loan_amount * 1000 + 1)

score = (
    2.6 * (credit_history == 1.0).astype(float)
    + 0.9 * (education == "Graduate").astype(float)
    + 0.6 * (property_area == "Semiurban").astype(float)
    + 0.15 * (property_area == "Urban").astype(float)
    + 1.4 * np.clip((income_to_loan - 15) / 15, -1, 1.5)
    + 0.4 * (married == "Yes").astype(float)
    - 0.5 * (dependents == "3+").astype(float)
    - 0.3 * (self_employed == "Yes").astype(float)
)

noise = np.random.normal(0, 1.0, size=N)
prob_approved = 1 / (1 + np.exp(-(score + noise - 1.2)))
loan_status = np.where(prob_approved > 0.5, "Y", "N")

loan_id = [f"LP{100000 + i}" for i in range(N)]

df = pd.DataFrame({
    "Loan_ID": loan_id,
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income.astype(int),
    "CoapplicantIncome": coapplicant_income.astype(int),
    "LoanAmount": loan_amount_missing,
    "Loan_Amount_Term": loan_term_missing,
    "Credit_History": credit_history_missing,
    "Property_Area": property_area,
    "Loan_Status": loan_status,
})

df.to_csv("data/loan_data.csv", index=False)
print(f"Generated {len(df)} rows -> data/loan_data.csv")
print(df["Loan_Status"].value_counts(normalize=True))
