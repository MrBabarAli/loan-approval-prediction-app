# model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\BAJWA LAPTOPS\Desktop\Loan_Approval_Project\data\synthetic_loan_data.csv")

# Drop unnecessary columns
df.drop(['CustomerID', 'Name'], axis=1, inplace=True)

# Handle missing values if any
df.fillna(df.mode().iloc[0], inplace=True)

# Encode target
df['LoanApproved'] = df['LoanApproved'].map({'Yes': 1, 'No': 0})

# Identify categorical columns
categorical_cols = ['Gender', 'MaritalStatus', 'EducationLevel', 'EmploymentStatus', 'PurposeOfLoan']

# Encode categorical columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop('LoanApproved', axis=1)
y = df['LoanApproved']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Train Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_res, y_train_res)
y_pred_lr = lr.predict(X_test)

# Train Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train_res, y_train_res)
y_pred_dt = dt.predict(X_test)

# Evaluation
print("=== Logistic Regression ===")
print(classification_report(y_test, y_pred_lr))

print("=== Decision Tree ===")
print(classification_report(y_test, y_pred_dt))

# Save models
joblib.dump(lr, 'models/logistic_model.pkl')
joblib.dump(dt, 'models/decision_tree_model.pkl')
