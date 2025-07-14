import streamlit as st
import pandas as pd
import joblib

# تحميل النموذج و LabelEncoder
model = joblib.load("diabetes_prediction_model.pkl")
smoking_encoder = joblib.load("smoking_history_encoder.pkl")

# إعداد الواجهة
st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🔍 Diabetes Prediction App")
st.markdown("أدخل البيانات الصحية ليقوم النموذج بتوقع وجود مرض السكري.")

# إدخالات المستخدم
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", min_value=1, max_value=100, value=30)
hypertension = st.selectbox("Hypertension (ضغط مرتفع)", [0, 1])
heart_disease = st.selectbox("Heart Disease (أمراض قلب)", [0, 1])
smoking_input = st.selectbox("Smoking History", ['never', 'No Info', 'current', 'former', 'ever', 'not current'])
bmi = st.number_input("BMI (مؤشر كتلة الجسم)", min_value=10.0, max_value=70.0, value=25.0)
hba1c_level = st.number_input("HbA1c Level (الهيموغلوبين السكري)", min_value=3.0, max_value=15.0, value=5.5)
blood_glucose_level = st.number_input("Blood Glucose Level (مستوى السكر)", min_value=50.0, max_value=300.0, value=100.0)

# تحويل القيم
gender_num = 1 if gender == "Male" else 0
smoking_num = smoking_encoder.transform([smoking_input])[0]

# تجميع البيانات في DataFrame (بدل np.array)
input_data = pd.DataFrame([{
    "gender": gender_num,
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "smoking_history": smoking_num,
    "bmi": bmi,
    "HbA1c_level": hba1c_level,
    "blood_glucose_level": blood_glucose_level
}])

# زر التنبؤ
if st.button("🔎 Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ The model predicts the person is **Diabetic**.\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ The model predicts the person is **Not Diabetic**.\nProbability: {probability:.2f}")
