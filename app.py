import streamlit as st
import pandas as pd
import joblib

st.title("Customer Churn Prediction")

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    scaled_data = scaler.transform(df)
    predictions = model.predict(scaled_data)

    df["Prediction"] = predictions
    st.write(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Results",
        csv,
        "predictions.csv",
        "text/csv",
    )
