# model_testing.py
import joblib
import pandas as pd

# Load models
logistic_model = joblib.load(r"C:\Users\BAJWA LAPTOPS\Desktop\Loan_Approval_Project\models\logistic_model.pkl")
decision_tree_model = joblib.load(r"C:\Users\BAJWA LAPTOPS\Desktop\Loan_Approval_Project\models\decision_tree_model.pkl")

# Example new customer data (must match training columns!)
sample_data = pd.DataFrame([{
    'Age': 35,
    'Gender': 1,  # 0 or 1 depending on encoding
    'MaritalStatus': 1,
    'EducationLevel': 2,
    'EmploymentStatus': 3,
    'AnnualIncome': 90000,
    'LoanAmountRequested': 25000,
    'PurposeOfLoan': 1,
    'CreditScore': 650,
    'ExistingLoansCount': 2,
    'LatePaymentsLastYear': 1
}])

# Predict
lr_pred = logistic_model.predict(sample_data)[0]
dt_pred = decision_tree_model.predict(sample_data)[0]

print(f"Logistic Regression Prediction: {'Approved' if lr_pred else 'Not Approved'}")
print(f"Decision Tree Prediction: {'Approved' if dt_pred else 'Not Approved'}")
