# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 01:57:46 2024

@author: Xabi
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

scaler = StandardScaler()
df = pd.read_csv('C:/Users/kabbe/Desktop/Hibuna/ckd new model/ckd_Encoded_data.csv')
scaler = StandardScaler()

x = df.drop(columns='stages', axis=1)
y = df['stages']
scaler.fit(x)
standardized_data = scaler.transform(x)

ckd_model = load_model("C:/Users/kabbe/Desktop/Hibuna/ckd new model/1d_CNNLSTM_98.keras")
#pickle.load(open('C:/Users/kabbe/Desktop/DiabStreamlit/diabetes_model.sav','rb'))
with st.sidebar:
    selected = option_menu('CKD Prediction by Using DL',
                    ['CKD Prediction', 'eGFR','About CKD','Home'],
                    default_index = 0)
    
if (selected == 'CKD Prediction' ):
    
    st.title('CKD Prediction with DL')
    age = st.text_input('Age')
    sex = st.text_input('Sex')
    sg = st.text_input('Specific Gravity')
    microalbumin = st.text_input('Microalbumin')
    glucose = st.text_input('Glucose')
    leukocytes = st.text_input('Leukocytes')
    calc = st.text_input('Calcium')
    urea = st.text_input('Urea')
    nitrites = st.text_input('Nitrites')
    sod = st.text_input('Sodium')
    pot = st.text_input('Potassium')
    sc = st.text_input('Creatinine')
    bgr = st.text_input('Blood Glucose Random')
    bu = st.text_input('Blood Urea')
    hemo = st.text_input('Hemoglobin')
    cystC= st.text_input('Cystatin C')

    # code for pred
    
    CKD_dignosis = ''
    
    #creatint button for prediction
    
    if st.button('CKD Test Result'):
        input_data = [[age,sex,sg,microalbumin,glucose,leukocytes,calc,urea,nitrites,sod,pot,sc,bgr,bu,hemo,cystC]]
        input_data_array = np.asarray(input_data)

        #reshape
        input_data_reshaped = input_data_array.reshape(1,-1)
        std_data = scaler.transform(input_data_reshaped)
        
        ckd_cnnlstmPred = ckd_model.predict(std_data)
        prediction = ckd_cnnlstmPred[0]
        predicted_index = np.argmax(prediction)
        stage_labels = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'Stage 5']
        predicted_stage = stage_labels[predicted_index]
        CKD_dignosis = predicted_stage

    st.success(CKD_dignosis)
    
if (selected == 'eGFR' ):
    
    st.title('eGFR Caliculation')
    age = st.text_input('Age')
    sex = st.text_input('Sex')
    sc = st.text_input('Creatinine')
    cystC= st.text_input('Cystatin C')
    eGFR_result = ''
    
    if st.button('eGFR result'):
        #if sex_values == 1:
        #gfr = 141*((sc/0.9)**-1.209)* (0.993**age_values[i])
        gfr = (100/float(cystC))-14
        #gfr = 141 * min((seri_cret_values[i] / 0.9) ** -0.411, 1) * max((seri_cret_values[i] / 0.9) ** -1.209, 1) * (0.993 ** age_values[i])*1.159
        #elif sex_values[i] == 0:
       # gfr = 144*((seri_cret_values[i]/0.7)**-1.209)* (0.993**age_values[i])
        #else:
        #gfr = None
        eGFR_result = gfr
    st.success(eGFR_result)