

# Loan Approval Prediction using Machine Learning

## Project Description

This project predicts whether a loan application will be **approved or rejected** using machine learning.

The model analyzes financial and personal information of applicants such as income, loan amount, credit score, and assets to determine loan approval.

## Dataset

The dataset contains **4269 loan records** with several financial features.

Important features include:

- Number of dependents
- Education
- Self employment status
- Annual income
- Loan amount
- Loan term
- CIBIL score
- Residential asset value
- Commercial asset value
- Luxury asset value
- Bank asset value

Target variable:
loan status

Values:
- Approved
- Rejected

---

## Project Steps

1. Load the dataset  
2. Clean the data  
3. Convert categorical values to numbers  
4. Split dataset into training and testing data  
5. Train machine learning models  
6. Evaluate model performance  
7. Predict loan approval  

---

## Machine Learning Models Used

The following models were used in this project:

- Logistic Regression
- Random Forest
- XGBoost

---

## Model Performance

| Model | Accuracy |
|------|------|
| Logistic Regression | ~79% |
| Random Forest | ~97% |
| XGBoost | ~98% |

XGBoost achieved the highest accuracy.

---

## Important Insight

The most important feature affecting loan approval is:

**CIBIL Score**

This shows that credit score plays a major role in loan approval decisions.

---

## Technologies Used

- Python
- pandas
- numpy
- scikit-learn
- xgboost

---

## Project Structure

loan_approval_prediction
│
├── data
│ loan_approval_dataset.csv
│
├── main.py
├── requirements.txt
└── README.md

---

## Conclusion

This project demonstrates how machine learning can help predict loan approval based on financial data. Ensemble models like Random Forest and XGBoost provide high prediction accuracy for this dataset.

---

## Author

Sravani Divvela
