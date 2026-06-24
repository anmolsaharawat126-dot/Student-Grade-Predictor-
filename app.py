import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Student Grade Predictor", page_icon="🎓")

st.title("🎓 Student Grade Predictor")
st.write("Predict final student marks based on attendance, assignment score, and internal marks.")

data = {
    "attendance": [60, 70, 80, 85, 90, 95],
    "assignment": [50, 60, 70, 75, 85, 90],
    "internal": [45, 55, 65, 70, 80, 90],
    "final_marks": [50, 60, 70, 75, 85, 92],
}

df = pd.DataFrame(data)

X = df[["attendance", "assignment", "internal"]]
y = df["final_marks"]

model = LinearRegression()
model.fit(X, y)

name = st.text_input("Student Name")
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=75)
assignment = st.number_input("Assignment Score (%)", min_value=0, max_value=100, value=70)
internal = st.number_input("Internal Marks (%)", min_value=0, max_value=100, value=65)

if st.button("Predict Grade"):
    prediction = model.predict([[attendance, assignment, internal]])
    marks = round(prediction[0], 2)

    st.subheader(f"Result for {name if name else 'Student'}")
    st.success(f"Predicted Final Marks: {marks}%")

    if marks >= 90:
        grade = "A+"
    elif marks >= 80:
        grade = "A"
    elif marks >= 70:
        grade = "B"
    elif marks >= 60:
        grade = "C"
    elif marks >= 50:
        grade = "D"
    else:
        grade = "F"

    st.info(f"Predicted Grade: {grade}")

    if marks >= 50:
        st.success("✅ Student is likely to pass.")
    else:
        st.error("❌ Student is at risk of failing.")
