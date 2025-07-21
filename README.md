🚀 Loan Approval Prediction System
A Flask web application that predicts whether a loan application will be approved based on user input. It uses two machine learning models — Logistic Regression and Decision Tree Classifier — trained on a preprocessed dataset. The project handles missing values, categorical features, and class imbalance using SMOTE.

📌 Features
🧹 Data Preprocessing

Handles missing values

Encodes categorical variables using Label Encoding

Balances the dataset using SMOTE

# 🚀 Loan Approval Prediction System

A Flask web application that predicts whether a loan application will be **approved or rejected** based on user input. It uses two machine learning models — **Logistic Regression** and **Decision Tree Classifier** — trained on a preprocessed dataset. The project includes full ML workflow support: data cleaning, categorical encoding, **SMOTE** for class imbalance, model training, evaluation, and an interactive web UI for real‑time inference.

---

## 📚 Table of Contents

* [Features](#-features)
* [Demo Screenshots](#-demo-screenshots)
* [Project Structure](#-project-structure)
* [Data Description](#-data-description)
* [Installation](#-installation)
* [Quick Start](#-quick-start)
* [Training the Models](#-training-the-models)
* [Running the Flask App](#-running-the-flask-app)
* [Web App Input Fields](#-web-app-input-fields)
* [Model Evaluation](#-model-evaluation)
* [Troubleshooting](#-troubleshooting)
* [Future Improvements](#-future-improvements)
* [Contributing](#-contributing)
* [Author](#-author)
* [License](#-license)

---

## 📌 Features

### 🧹 Data Preprocessing

* Handles missing values (numeric + categorical strategies)
* Encodes categorical variables using **Label Encoding**
* Balances classes using **SMOTE** (Synthetic Minority Oversampling Technique)

### 🧠 Model Training

* Trains **Logistic Regression** & **Decision Tree** models
* Saves trained estimators to disk as **`.pkl`** files for deployment
* Optionally saves fitted LabelEncoders for consistent inference

### 📊 Evaluation Metrics

* Precision, Recall, F1‑Score, Accuracy (optional)
* Side‑by‑side model comparison report

### 🌐 Flask Web App

* Clean Bootstrap‑styled UI form for loan prediction
* Dropdowns for categorical inputs (no confusing numeric codes!)
* Model selection (Logistic vs Decision Tree)
* Displays **prediction result + probability / confidence**
* Plotly chart visualization (bar/gauge) of approval probability
* Form validation + user alerts (Flask `flash` messages)

---

## 🖼 Demo Screenshots

> *Add screenshots of the running web app here.*

| Home Form                              | Prediction Result                |
| -------------------------------------- | -------------------------------- |
| ![Home Form](static/img/home_form.png) | ![Result](static/img/result.png) |

---

## 📂 Project Structure

```text
Loan_Approval_Project/
│
├── data/
│   └── loan_data.csv              # Raw / cleaned dataset
│
├── models/
│   ├── logistic_model.pkl         # Trained Logistic Regression model
│   ├── decision_tree_model.pkl    # Trained Decision Tree model
│   └── label_encoders.pkl         # (Optional) Saved encoders
│
├── static/
│   ├── style.css                  # Custom CSS overrides (optional)
│   └── img/                       # Screenshots / assets
│
├── templates/
│   ├── index.html                 # Main prediction form
│   └── result.html                # (Optional) separate result page
│
├── model_training.py              # Script: preprocess + train + save models
├── app.py                         # Flask web server (use this to run app)
├── model_testing.py               # Quick local inference test (optional)
├── requirements.txt               # Python dependencies
├── README.md                      # You are here ✅
└── .gitignore                     # (Recommended) Ignore venv, __pycache__, *.pkl?
```

---

## 🧾 Data Description

Below are the **expected columns** for the training dataset. Adjust `model_training.py` if your dataset differs.

| Column                 | Type   | Description                                               | Example            |
| ---------------------- | ------ | --------------------------------------------------------- | ------------------ |
| `CustomerID`           | string | Unique applicant ID (drop)                                | `bdd640fb-0667...` |
| `Name`                 | string | Applicant name (drop)                                     | `Michelle Prince`  |
| `Age`                  | int    | Applicant age                                             | `56`               |
| `Gender`               | cat    | Male / Female                                             | `Male`             |
| `MaritalStatus`        | cat    | Single / Married / Divorced / Widowed                     | `Single`           |
| `EducationLevel`       | cat    | High School / Bachelor / Master / PhD / Other             | `High School`      |
| `EmploymentStatus`     | cat    | Employed / Unemployed / Self-employed / Student / Retired | `Employed`         |
| `AnnualIncome`         | float  | Annual income in local currency                           | `97622`            |
| `LoanAmountRequested`  | float  | Requested loan amount                                     | `46413`            |
| `PurposeOfLoan`        | cat    | Home / Car / Education / Personal                         | `Personal`         |
| `CreditScore`          | int    | Credit score (0–850 style scale)                          | `500`              |
| `ExistingLoansCount`   | int    | Number of current active loans                            | `2`                |
| `LatePaymentsLastYear` | int    | Number of late payments in last 12 months                 | `4`                |
| `LoanApproved`         | target | Yes / No                                                  | `Yes`              |

### 🔐 Target Encoding

`LoanApproved` → `Yes` = 1, `No` = 0.

---

## 🔢 Encoding Maps (Used in Web App)

> These integer codes must match how models were trained. Update if your training encodings differ.

**Gender**: `Female=0`, `Male=1`

**MaritalStatus**: `Divorced=0`, `Married=1`, `Single=2`, `Widowed=3`

**EducationLevel**: `Bachelor=0`, `High School=1`, `Master=2`, `Other=3`, `PhD=4`

**EmploymentStatus**: `Employed=0`, `Retired=1`, `Self-employed=2`, `Student=3`, `Unemployed=4`

**PurposeOfLoan**: `Car=0`, `Education=1`, `Home=2`, `Personal=3`

> ⚠️ If you re‑train the models and encodings change, **update both `model_training.py` and the HTML dropdowns**.

---

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/loan-approval-prediction.git
cd loan-approval-prediction

# (Recommended) create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## ⚙️ Quick Start

1. Ensure your dataset is available at `data/loan_data.csv` (or update the path in `model_training.py`).
2. Train models:

   ```bash
   python model_training.py
   ```
3. Run the app:

   ```bash
   python app.py
   ```
4. Open your browser at: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 🧪 Training the Models

The `model_training.py` script:

* Loads the dataset
* Drops non‑informative columns (`CustomerID`, `Name`)
* Handles missing values (median / mode strategies)
* Encodes categorical features
* Applies **SMOTE** to address class imbalance in the target
* Splits into train/test
* Trains Logistic Regression & Decision Tree models
* Evaluates them (prints classification report)
* Saves models to `models/`

Run:

```bash
python model_training.py
```

After training you should see:

```
models/
├── logistic_model.pkl
└── decision_tree_model.pkl
```

(Optional) encoders saved as `label_encoders.pkl`.

---

## ▶️ Running the Flask App

Start the web server:

```bash
python app.py
```

Then open:

```
http://127.0.0.1:5000/
```

Enter applicant details in the form and submit to see:

* **Prediction**: Approved / Not Approved
* **Confidence Score** from selected model
* **Probability Chart** (Plotly bar / gauge)

---

## 🧾 Web App Input Fields

These correspond to encoded features used in training. Dropdown labels are user‑friendly; code conversion happens in the backend.

| Field (UI)                 | Backend Feature        | Notes                    |
| -------------------------- | ---------------------- | ------------------------ |
| Age                        | `Age`                  | Numeric                  |
| Gender                     | `Gender`               | Dropdown → 0/1           |
| Marital Status             | `MaritalStatus`        | Dropdown                 |
| Education                  | `EducationLevel`       | Dropdown                 |
| Employment                 | `EmploymentStatus`     | Dropdown                 |
| Annual Income (\$)         | `AnnualIncome`         | Numeric                  |
| Loan Amount Requested (\$) | `LoanAmountRequested`  | Numeric                  |
| Purpose of Loan            | `PurposeOfLoan`        | Dropdown                 |
| Credit Score               | `CreditScore`          | Numeric                  |
| Loans Count                | `ExistingLoansCount`   | Integer                  |
| Late Payments              | `LatePaymentsLastYear` | Integer                  |
| Select Model               | (UI only)              | Logistic / Decision Tree |

---

## 📊 Model Evaluation

After training, classification reports are printed for each model. Example metrics:

```
=== Logistic Regression ===
precision    recall  f1-score  support
...

=== Decision Tree ===
precision    recall  f1-score  support
...
```

You can extend `model_training.py` to log metrics to CSV, plot confusion matrices, or export reports to Markdown/HTML.

---

## 🧯 Troubleshooting

### ❌ `_pickle.UnpicklingError: STACK_GLOBAL requires str`

This occurs when loading a `.pkl` model saved in a **different Python / scikit-learn environment**.
**Fix:** Re‑train and re‑save models in your current environment using `model_training.py`.

### ❌ `Model loading error: [Errno 2] No such file or directory`

Check that the `models/` folder exists relative to `app.py` and contains the `.pkl` files.

### ❌ `Form Error: 400 Bad Request`

Usually caused by mismatched form field names. Ensure the input `name="..."` attributes in `index.html` match the keys used in `request.form[...]` in `app.py`.

### ❌ ModuleNotFoundError: No module named 'plotly'

Install it:

```bash
pip install plotly
```

Update `requirements.txt` afterwards:

```bash
pip freeze > requirements.txt
```

---

## 🧠 Future Improvements

* Add more ML models (Random Forest, XGBoost, LightGBM)
* Hyperparameter tuning (GridSearchCV / Optuna)
* Model explainability with SHAP or LIME
* Persist user submissions & predictions in SQLite / PostgreSQL
* Authentication / admin dashboard
* Containerize with Docker
* Deploy to Render, Railway, or AWS Elastic Beanstalk
* Convert to Streamlit for rapid prototyping

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m "Add my feature"`
4. Push branch: `git push origin feature/my-feature`
5. Open a Pull Request

Feel free to open an **Issue** for bugs, questions, or feature requests.

---

## 👤 Author

**Babar Ali**
📧 Email: [usmankkjj@gmail.com](mailto:usmankkjj@gmail.com)
🔗 LinkedIn: [https://www.linkedin.com/in/babar-ali-babar](https://www.linkedin.com/in/babar-ali-babar)

> If you use this project in academic work, please cite the repository and include a link.

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

### ⭐ Star This Repo

If you find this project helpful, please ⭐ star the repository on GitHub — it helps others discover it!

---

**Happy Building & Predicting!** 💻📊🏦

