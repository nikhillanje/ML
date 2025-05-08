import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('decision_tree_model.pkl')

# Create encoders again (same as before)
from sklearn.preprocessing import LabelEncoder

# Initialize encoders
le_outlook = LabelEncoder()
le_temperature = LabelEncoder()
le_humidity = LabelEncoder()
le_wind = LabelEncoder()
le_play = LabelEncoder()

# Fit encoders with original data (classes must match your training)
le_outlook.fit(['Sunny', 'Overcast', 'Rainy'])
le_temperature.fit(['Hot', 'Mild', 'Cool'])
le_humidity.fit(['High', 'Normal'])
le_wind.fit(['Weak', 'Strong'])
le_play.fit(['No', 'Yes'])

# Streamlit UI
st.title("🎾 Play Tennis Predictor")

st.write("Enter the weather conditions below to check if you should play tennis today:")

# User inputs
outlook = st.selectbox("Outlook", ['Sunny', 'Overcast', 'Rainy'])
temperature = st.selectbox("Temperature", ['Hot', 'Mild', 'Cool'])
humidity = st.selectbox("Humidity", ['High', 'Normal'])
wind = st.selectbox("Wind", ['Weak', 'Strong'])

if st.button("Predict"):
    # Prepare the input for prediction
    test_input = [[le_outlook.transform([outlook])[0],
                   le_temperature.transform([temperature])[0],
                   le_humidity.transform([humidity])[0],
                   le_wind.transform([wind])[0]]]
    
    # Make prediction
    prediction = model.predict(test_input)
    result = le_play.inverse_transform(prediction)[0]
    
    # Show result
    if result == 'Yes':
        st.success("✅ Yes! You can play tennis today.")
    else:
        st.warning("🚫 No, not a good day for tennis.")
