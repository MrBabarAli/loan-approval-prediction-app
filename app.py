# app.py
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load models
lr_model = joblib.load(r"C:\Users\BAJWA LAPTOPS\Desktop\Loan_Approval_Project\models\logistic_model.pkl")
dt_model = joblib.load(r"C:\Users\BAJWA LAPTOPS\Desktop\Loan_Approval_Project\models\decision_tree_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form values
    features = [int(request.form.get(f)) for f in [
        'Age', 'Gender', 'MaritalStatus', 'EducationLevel',
        'EmploymentStatus', 'AnnualIncome', 'LoanAmountRequested',
        'PurposeOfLoan', 'CreditScore', 'ExistingLoansCount', 'LatePaymentsLastYear'
    ]]

    # Create DataFrame
    input_data = pd.DataFrame([features], columns=[
        'Age', 'Gender', 'MaritalStatus', 'EducationLevel',
        'EmploymentStatus', 'AnnualIncome', 'LoanAmountRequested',
        'PurposeOfLoan', 'CreditScore', 'ExistingLoansCount', 'LatePaymentsLastYear'
    ])

    # Predict
    lr_pred = lr_model.predict(input_data)[0]
    dt_pred = dt_model.predict(input_data)[0]

    return render_template('index.html',
                           prediction_lr="Approved" if lr_pred else "Not Approved",
                           prediction_dt="Approved" if dt_pred else "Not Approved")

if __name__ == '__main__':
    app.run(debug=True)
