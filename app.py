import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import streamlit as st


##importing the model
model = load_model("my_model.keras")

##Importing the preprocessing tools
with open("annclassification/annclassification/onehot_encoder_geo.pkl",'rb') as file:
    geo_encoder = pickle.load(file)

with open("annclassification/annclassification/label_encoder_gender.pkl",'rb') as file:
    gender_encoder = pickle.load(file)

with open("scaler.pkl",'rb') as file:
    scaler = pickle.load(file)

##title for app
st.title('Custom Churn Perdiction')

#User title
geography = st.selectbox("Geography",geo_encoder.categories_[0])
gender = st.selectbox("Gender", gender_encoder.classes_)
age = st.slider("Age",18,92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure",0,10)
number_of_pproducts = st.slider("Number of Products",0,10)
has_cr_card = st.selectbox("Has credit card",[0,1])
is_active_number = st.selectbox("Is Active Member",[0,1])

##prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [number_of_pproducts],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_number],
    'EstimatedSalary': [estimated_salary],
})

geo_encoded = geo_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out(['Geography']))

#Combine the one columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

#Scale the input data
input_data_scaled = scaler.transform(input_data)

#Predict churn
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

if prediction_prob >0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("He's not likely to churn.")