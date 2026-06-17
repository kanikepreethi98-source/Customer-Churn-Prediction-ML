import streamlit as st
import pandas as pd
import joblib

st.title("Customer Churn Prediction")

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    feature_columns = [
        'State',
        'Account length',
        'Area code',
        'International plan',
        'Voice mail plan',
        'Number vmail messages',
        'Total day minutes',
        'Total day calls',
        'Total day charge',
        'Total eve minutes',
        'Total eve calls',
        'Total eve charge',
        'Total night minutes',
        'Total night calls',
        'Total night charge',
        'Total intl minutes',
        'Total intl calls',
        'Total intl charge'
    ]

    X = df[feature_columns]

    scaled_data = scaler.transform(X)
    predictions = model.predict(scaled_data)

    df["Prediction"] = predictions

    st.write(df)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Results",
        csv,
        "predictions.csv",
        "text/csv"
    )
