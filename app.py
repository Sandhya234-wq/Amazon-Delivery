import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline model
model = joblib.load("model_4.pkl")

st.set_page_config(page_title="Amazon Delivery Time Predictor", layout="centered")
st.title("📦 Amazon Delivery Time Prediction App")
st.markdown("This app predicts the expected delivery time for an Amazon order based on distance, traffic, product category, and more.")

st.header("Enter Delivery Details")

# Collecting user input 
input_data = {
    'Weather': st.selectbox('Weather', ['Sunny', 'Rainy', 'Stormy']),
    'Traffic': st.selectbox('Traffic Conditions', ['Low', 'Medium', 'High', 'Jam']),
    'Vehicle': st.selectbox('Vehicle Type', ['Scooter', 'Motorcycle', 'Bike']),
    'Area': st.selectbox('Delivery Area Type', ['Urban', 'Metropolitian', 'Other']),
    'Category': st.selectbox('Product Category', ['Grocery', 'Electronics', 'Clothing']),
    'Agent_Age': st.slider('Agent Age', 18, 60, 28),
    'Agent_Rating': st.slider('Agent Rating', 1.0, 5.0, 4.5),
    'Distance_km': st.number_input('Delivery Distance (km)', min_value=0.1, value=5.0),
    'Order_Hour': st.slider('Order Hour (0-23)', 0, 23, 13),
    'Order_Day': st.slider('Order Day (1=Monday, 7=Sunday)', 1, 7, 2),
    'Pickup_Delay_Minutes': st.slider('Pickup Delay (minutes)', 0, 180, 5),
}

# Convert to DataFrame for prediction
input_df = pd.DataFrame([input_data])

if st.button("🚚 Predict Delivery Time"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"✅ Estimated Delivery Time: {round(prediction, 2)} minutes")
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")

st.markdown("---")
st.markdown("Developed as part of an MLOps project using MLflow and Streamlit.")
