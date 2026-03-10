# ===============================
# Loan Approval Prediction System
# ===============================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


# ===============================
# 1. Load Dataset
# ===============================

data = pd.read_csv("data/loan_approval_dataset.csv")

# Remove spaces in column names
data.columns = data.columns.str.strip()

print("Dataset Shape:", data.shape)
print(data.head())


# ===============================
# 2. Data Cleaning
# ===============================

# Drop unnecessary column
data = data.drop("loan_id", axis=1)

# Encode categorical columns
data["education"] = data["education"].map({" Graduate": 1, " Not Graduate": 0})
data["self_employed"] = data["self_employed"].map({" Yes": 1, " No": 0})
data["loan_status"] = data["loan_status"].map({" Approved": 1, " Rejected": 0})

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())


# ===============================
# 3. Feature and Target Split
# ===============================

X = data.drop("loan_status", axis=1)
y = data["loan_status"]


# ===============================
# 4. Train Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", X_train.shape)
print("Testing samples:", X_test.shape)


# ===============================
# 5. Logistic Regression Model
# ===============================

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\nLogistic Regression Accuracy:",
      accuracy_score(y_test, lr_pred))


# ===============================
# 6. Random Forest Model
# ===============================

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Accuracy:",
      accuracy_score(y_test, rf_pred))


# ===============================
# 7. XGBoost Model
# ===============================

xgb_model = XGBClassifier(use_label_encoder=False,
                          eval_metric="logloss")

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

print("\nXGBoost Accuracy:",
      accuracy_score(y_test, xgb_pred))


# ===============================
# 8. Model Comparison
# ===============================

print("\nModel Comparison")

print("Logistic Regression:", accuracy_score(y_test, lr_pred))
print("Random Forest:", accuracy_score(y_test, rf_pred))
print("XGBoost:", accuracy_score(y_test, xgb_pred))


# ===============================
# 9. Feature Importance
# ===============================

feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(feature_importance)


# ===============================
# 10. Prediction Function
# ===============================

def predict_loan(input_data):

    input_df = pd.DataFrame([input_data], columns=X.columns)

    prediction = xgb_model.predict(input_df)[0]

    probability = xgb_model.predict_proba(input_df)[0][1]

    if prediction == 1:
        result = "Loan Approved"
    else:
        result = "Loan Rejected"

    return result, probability


# ===============================
# 11. Test Prediction
# ===============================

sample_input = [
    2,      # dependents
    1,      # education
    0,      # self employed
    500000, # income
    150000, # loan amount
    12,     # loan term
    750,    # cibil score
    300000, # residential assets
    200000, # commercial assets
    100000, # luxury assets
    50000   # bank assets
]

result, prob = predict_loan(sample_input)

print("\nPrediction Result:", result)
print("Approval Probability:", round(prob * 100, 2), "%")