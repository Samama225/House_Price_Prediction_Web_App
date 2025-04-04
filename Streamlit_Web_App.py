import numpy as np
import pickle
import pandas as pd
import streamlit as st


with open("model.pickle", "rb") as f:
    loaded_model = pickle.load(f)






st.title("ML Model Deployment")
st.write("Enter the input features to get the prediction")


# Collect user input
feature1 = st.number_input("longitude")
feature2 = st.number_input("latitute")
feature3 = st.number_input("housing_median_age")
feature4 = st.number_input("total_rooms")
feature5 = st.number_input("total_bedrooms")
feature6 = st.number_input("population")
feature7 = st.number_input("households")
feature8 = st.number_input("median_income")
feature9 = st.number_input("median_house_value")

# Prediction
if st.button("Predict"):
    prediction = loaded_model.predict([[feature1, feature2, feature3,feature4,feature5,feature6,feature7,feature8,feature9]])
    st.success(f"Predicted Value: {prediction[0]}")
