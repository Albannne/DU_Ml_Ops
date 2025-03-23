# app/app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load the best model
def load_model(model_dir):
    # Read the best model file
    with open(f"{model_dir}/best_model.txt", 'r') as f:
        best_model = f.read().strip()
    
    # Load the model
    model_path = f"{model_dir}/{best_model}.pkl"
    model = joblib.load(model_path)
    
    # Load the scaler
    scaler_path = "data/processed/scaler.pkl"
    scaler = joblib.load(scaler_path)
    
    return model, scaler, best_model

# Make prediction
def predict(model, scaler, features):
    # Scale features
    features_scaled = scaler.transform([features])
    
    # Make prediction
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[0][1]
    
    return prediction[0], probability

# Setup the Streamlit app
def main():
    # Page config
    st.set_page_config(
        page_title="Loan Default Prediction",
        page_icon="💰",
        layout="wide"
    )
    
    # App title and description
    st.title("Loan Default Prediction")
    st.markdown("""
    This app predicts the probability of a customer defaulting on a loan.
    Enter the customer's information below to get a prediction.
    """)
    
    try:
        # Load model and scaler
        model_dir = "../models"
        model, scaler, model_name = load_model(model_dir)
        
        st.sidebar.info(f"Using {model_name.replace('_', ' ').title()} model")
        
        # Create input form for user
        st.subheader("Customer Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            credit_lines = st.number_input("Credit Lines Outstanding", min_value=0, max_value=10, value=1)
            loan_amount = st.number_input("Loan Amount Outstanding", min_value=0.0, max_value=10000.0, value=3000.0)

# app/app.py (continuation)
            total_debt = st.number_input("Total Debt Outstanding", min_value=0.0, max_value=30000.0, value=5000.0)
            income = st.number_input("Annual Income", min_value=10000.0, max_value=150000.0, value=60000.0)
        
        with col2:
            years_employed = st.number_input("Years Employed", min_value=0, max_value=20, value=4)
            fico_score = st.slider("FICO Score", min_value=300, max_value=850, value=650)
        
        # Create a button to make prediction
        if st.button("Predict Default Risk"):
            # Prepare feature array
            features = [credit_lines, loan_amount, total_debt, income, years_employed, fico_score]
            
            # Make prediction
            prediction, probability = predict(model, scaler, features)
            
            # Display prediction
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ High Risk of Default")
                else:
                    st.success("✅ Low Risk of Default")
                
                st.metric("Default Probability", f"{probability:.2%}")
            
            with col2:
                # Create a gauge chart for risk visualization
                fig, ax = plt.subplots(figsize=(4, 3))
                
                # Create color gradient
                cmap = sns.color_palette("RdYlGn_r", as_cmap=True)
                
                # Plot the gauge
                ax.bar(0, 1, color=cmap(probability))
                ax.set_xlim(-0.5, 0.5)
                ax.set_ylim(0, 1)
                ax.text(0, 0.5, f"{probability:.1%}", ha='center', va='center', fontsize=20, fontweight='bold')
                ax.set_title("Default Risk")
                ax.set_xticks([])
                ax.set_yticks([])
                
                st.pyplot(fig)
            
            # Show risk factors
            st.subheader("Risk Analysis")
            
            risk_factors = []
            if credit_lines > 3:
                risk_factors.append("High number of credit lines outstanding")
            if total_debt > loan_amount * 2:
                risk_factors.append("High total debt compared to loan amount")
            if income < total_debt * 3:
                risk_factors.append("Debt-to-income ratio is concerning")
            if years_employed < 2:
                risk_factors.append("Short employment history")
            if fico_score < 600:
                risk_factors.append("Low credit score")
            
            if risk_factors:
                st.write("Risk factors identified:")
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.write("No significant risk factors identified.")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Please make sure the model files exist and are accessible.")

if __name__ == "__main__":
    main()
