import streamlit as st
import numpy as np 
import pickle

# Load the model
rfc = pickle.load(open('rfc.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Define prediction function
def prediction(credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary):
    country_dict = {'France': 0, 'Germany': 1, 'Spain': 2}
    gender_dict = {'Male': 0, 'Female': 1}

    country_val = country_dict.get(country.strip().title(), 0)
    gender_val = gender_dict.get(gender.strip().title(), 0)

    features = np.array([[credit_score, country_val, gender_val, age, tenure, balance, products_number, credit_card, active_member, estimated_salary]])
    return rfc.predict(features)[0]

# Web app title
st.title('Bank Customer Churn Prediction')

# Show image
st.image('img.jpg', caption='Customer Churn Analysis', use_container_width=True)


# Input fields
credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=600)
country = st.selectbox('Country', ['France', 'Germany', 'Spain'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=18, max_value=100, value=35)
tenure = st.number_input('Tenure (Years with bank)', min_value=0, max_value=10, value=3)
balance = st.number_input('Balance', value=50000.0)
products_number = st.number_input('Number of Products', min_value=1, max_value=4, value=1)
credit_card = st.selectbox('Has Credit Card?', [1, 0])
active_member = st.selectbox('Active Member?', [1, 0])
estimated_salary = st.number_input('Estimated Salary', value=50000.0)

# Predict button
if st.button('Predict'):
    result = prediction(credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary)

    if result == 1:
        st.error("The customer is likely to leave.")
    else:
        st.success("The customer is likely to stay.")
