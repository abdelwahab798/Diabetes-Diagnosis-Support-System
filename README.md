# 🧠 Diabetes Diagnosis Support System

An intelligent machine learning model designed to assist in **confirming the diagnosis of diabetes** based on patients’ medical test results and demographic data.

## 📌 Project Objective

To build a smart prediction system that supports healthcare professionals in identifying patients with potential diabetes by analyzing:

- Medical test results (e.g., **HbA1c Level**, **Blood Glucose Level**)
- Demographic and health information (e.g., **Age**, **Gender**, **BMI**, **Smoking History**, etc.)

> ⚠️ Note: This model is not a replacement for medical diagnosis but a decision support tool based on real-world data.

---

## 📊 Dataset Overview

- **Source**: [Kaggle - Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
- **Size**: ~100,000 rows
- **Target Variable**: `diabetes` (1 = diabetic, 0 = non-diabetic)

### 🔢 Features

| Feature              | Description                                        |
|----------------------|----------------------------------------------------|
| `gender`             | Male or Female                                     |
| `age`                | Age in years                                       |
| `hypertension`       | Whether the patient has hypertension               |
| `heart_disease`      | Whether the patient has any heart disease          |
| `smoking_history`    | History of smoking (never, former, current, etc.) |
| `bmi`                | Body Mass Index                                    |
| `HbA1c_level`        | Long-term blood sugar average                      |
| `blood_glucose_level`| Blood glucose level at the moment of testing      |
| `diabetes`           | Target (0 = No, 1 = Yes)                           |

---

## 🧪 Exploratory Data Analysis (EDA)

- Count plots and bar charts for categorical features.
- Boxen and scatter plots to analyze outliers and relationships.
- Heatmap correlation matrix to identify influential features.
- Investigation of data quality issues (e.g., ages below 1, missing values, label distributions).

---

## ⚙️ Data Preprocessing

- Label encoding for categorical features (`smoking_history`, `gender`).
- Outlier analysis and data cleaning.
- Addressed class imbalance using:
  - `class_weight="balanced"` in classifiers
  - `SMOTE` oversampling technique
  - `scale_pos_weight` in XGBoost

---

## 🤖 Models & Evaluation

### 🧠 Models Trained:

- **Random Forest Classifier**
- **XGBoost Classifier**
- **Decision Tree Classifier**

### 📈 Best Model Performance (XGBoost):

- **Accuracy**: 94%
- **Recall (for diabetic)**: 85%
- **ROC AUC**: 0.90+
- **SMOTE + scale_pos_weight** were applied

### 📉 Metrics used:

- Accuracy
- Precision / Recall / F1-score
- ROC AUC Score
- Confusion Matrix

---

## 🚀 Deployment

The final model was deployed using **Streamlit**, providing:

- A user-friendly interface for prediction
- Page to input medical test results (`HbA1c` & `Glucose`)
- Real-time diabetes risk prediction
- Model saved using `joblib` (`.pkl` file)

To run the app:

```bash
streamlit run streamlit_app.py
