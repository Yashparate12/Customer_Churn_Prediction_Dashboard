import streamlit as st
import pandas as pd
from src.inference.predict import predict_with_details

st.set_page_config(page_title="Churn Prediction", layout="wide")

st.title("Customer Churn Prediction System")
st.subheader("AI-Powered Retention Analysis")

input_data = {}

col1, col2, col3 = st.columns(3)

input_data["tenure"] = st.number_input("Tenure", 0, 100)
input_data["MonthlyCharges"] = st.number_input("Monthly Charges", 0.0)
input_data["TotalCharges"] = st.number_input("Total Charges", 0.0)


if st.button("Predict Churn"):
    prediction, probability = predict_with_details(input_data)

    st.divider()

    if prediction == 1:
        st.error("⚠️ High Risk of Churn")

        st.markdown("### 📉 What this means")
        st.write(
            "The model predicts that this customer shows behavior patterns "
            "similar to customers who previously left the service."
        )

        if probability:
            st.metric("Estimated Churn Risk", f"{probability*100:.2f}%")

        st.markdown("### 🔍 Possible Reasons")
        st.write("""
        • Short customer tenure  
        • Higher recurring charges  
        • Low engagement indicators  
        """)

        st.markdown("### 🛠️ Recommended Actions")
        st.write("""
        • Offer a retention discount  
        • Provide personalized support  
        • Encourage long-term contract plans  
        """)

    else:
        st.success("✅ Customer Likely to Stay")

        st.markdown("### 📈 What this means")
        st.write(
            "The customer shows stable behavior patterns and is unlikely to churn "
            "based on historical data."
        )

        if probability:
            st.metric("Estimated Churn Risk", f"{probability*100:.2f}%")

        st.markdown("### 💡 Business Insight")
        st.write("""
        • Customer has healthy engagement  
        • Revenue is stable  
        • No immediate retention intervention needed  
        """)

        st.markdown("### 🤝 Suggested Strategy")
        st.write("""
        • Maintain current service quality  
        • Encourage upgrades or add-on services  
        • Monitor periodically for changes  
        """)

       
