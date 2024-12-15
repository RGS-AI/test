# Importing the necessary libraries
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np
from PIL import Image

st.write(f"## Disease Prediction Application")

st.write("Author: Raghunandan M S") 


# Loading the model
hdmodel = pickle.load(open('Models/hdmodel.sav', 'rb'))
diabetesmodel = pickle.load(open('Models/diabetes_model.sav', 'rb'))
parkinsonsmodel = pickle.load(open('Models/parkinsons_model.sav', 'rb'))

with st.sidebar:
    selected = option_menu("Choose the disease", ['Heart Disease', 'Diabetes Prediction', 'Parkinsons', 'About'], default_index=0)

if selected == 'Heart Disease':
    st.write(f"## Heart Disease Prediction")

    # Heart Disease
    st.write("Please provide the details")
    # Dividing into 2 columns
    col1, col2, col3 = st.columns(3)
    # Taking user inputs
    with col1:

        age = st.text_input("Enter your age:")
        gender = st.text_input("For male, write 1 or 0 for female")
        cp = st.text_input("Do you have Chest pain?")
        trestbps = st.text_input("Enter your resting blood pressure:")
        chol = st.text_input("Enter your cholestrol level:")
        fbs = st.text_input("Enter your fasting blood sugar:")
        restecg = st.text_input("Enter your resting electrocardiographic results:")

    with col2:
        thalach = st.text_input("Enter your maximum heart rate:")
        exang = st.text_input("Do you have exercise induced angina?")
        oldpeak = st.text_input("Enter your old peak:")
        slope = st.text_input("Enter your slope:")
        ca = st.text_input("Enter your number of major vessels:")
        thal = st.text_input("Enter your thalassemia:")

    with col3:
        #iamge = Image.open('Images/heart.png')
        st.image('Images/heart.png', width=200)

    if st.button('Predict'):
        data = [age, gender, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        data_array = np.array(data, dtype=float).reshape(1, -1)
        prediction = hdmodel.predict(data_array)
        st.write(f"## The prediction is {prediction}")

if selected == 'Diabetes Prediction':
    st.write(f"## Diabetes Prediction")
    
    # Take User inputs
    col1, col2, col3= st.columns(3)
   
    with col1:
        Pregnancies = st.text_input('No of Pregnancies')
        SkinThickness = st.text_input('Skin Thickness value')
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Glucose = st.text_input('Glucose Level')
        Insulin = st.text_input('Insulin Level')
        Age = st.text_input('Age')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
        BMI = st.text_input('BMI')

    if st.button('Predict'):
        data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        data_array = np.array(data, dtype=float).reshape(1, -1)
        prediction = diabetesmodel.predict(data_array)
        st.write(f"## The prediction is {prediction}")

#if selected == 'General':
    #st.write(f"## General Prediction")


if selected == 'Parkinsons':
    st.write(f"## Parkinsons Prediction")
    # Take the user input
    col1, col2, col3, col4, col5 = st.columns(5)  
    with col1:
        fo = st.text_input('Fo(Hz)')
        RAP = st.text_input('RAP')
        APQ3 = st.text_input('APQ3')
        HNR = st.text_input('HNR')
        D2 = st.text_input('D2')
    with col2:
        fhi = st.text_input('Fhi(Hz)')
        PPQ = st.text_input('PPQ')
        APQ5 = st.text_input('APQ5')
        RPDE = st.text_input('RPDE')
        PPE = st.text_input('PPE')
       
    with col3:
        flo = st.text_input('Flo(Hz)')
        DDP = st.text_input('DDP')
        APQ = st.text_input('APQ')
        DFA = st.text_input('DFA')
 
       
    with col4:
        Jitter_percent = st.text_input('Jitter(%)')
        Shimmer = st.text_input('Shimmer')
        DDA = st.text_input('DDA')
        spread1 = st.text_input('spread1')
       
    with col5:
        Jitter_Abs = st.text_input('Jitter(Abs)')
        Shimmer_dB = st.text_input('Shimmer(dB)')
        NHR = st.text_input('NHR')
        spread2 = st.text_input('spread2')

    if st.button('Predict'):
        data = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
        data_array = np.array(data, dtype=float).reshape(1, -1)
        prediction = parkinsonsmodel.predict(data_array)
        st.write(f"## The prediction is {prediction}")

if selected == 'About':
    st.write(f"## About")
    st.write("This is a disease prediction application. It is a simple application that predicts the disease based on the user's input. The application is built using Streamlit and the model is built using Machine Learning. This was done during KT-session part of ICBP 2024 by Raghunandan M S ")

