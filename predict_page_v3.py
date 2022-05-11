import streamlit as st
import pickle 
import pandas as pd
import numpy as np
from PIL import Image
import time

def load_model1():
    with open('saved_steps_rf1.pkl', 'rb') as file:
        data1 = pickle.load(file)

    return data1

def load_model2():
    with open('saved_steps_rf2.pkl', 'rb') as file:
        data2 = pickle.load(file)

    return data2

data1 = load_model1()
data2 = load_model2()
model1 = data1['model1']
model2 = data2['model2']

def page():

    st.set_page_config(page_title = 'Predictive maintenance', layout='centered')
    image1 = Image.open('image.png')
    st.image(image1)

    st.title('Predictive maintenance prediction // by Abhishek Chaudhary')
    st.write('### Provide the input to predict th requirment of maintenance')
    Q1 = st.number_input('Air temperature in K', step=10, value=298)
    Q2 = st.number_input('Process temperature in K', step=10, value=308)
    Q3 = st.number_input('Torque', step=10, value=43)
    Q4 = st.number_input('Rotational speed in RPM', step=10, value=1500)
    Q5 = st.number_input('Tool wear in min', step=1, value=0)
    #Q_Power = Q3*Q4
    Q6 = Q3*Q4
    st.write('Power in Watt', Q6)

    

    

    colms = ['Air temperature [K]', 'Process temperature [K]', 'Torque [Nm]',
       'Tool wear [min]', 'Power']

    x = np.array([[Q1, Q2, Q3, Q5, Q6]])
    test = pd.DataFrame(x, columns=colms)

    return test

def clf1(x):
    
    with col1:
        ok = st.button('Predict the requirment of maintenance')
        if ok:
            with st.spinner('Wait for it...'):
                time.sleep(1.1)
                st.success('Done!')
            ans = model1.predict(x)
            if ans[0] == 0:
                st.write('No chance of failure')
            elif ans[0] == 1:
                st.write('Possible chance of failure') 

def clf2(x):
    
    with col2:
        ok = st.button('Predict the type of failure')
        if ok:
            with st.spinner('Wait for it...'):
                time.sleep(1.1)
                st.success('Done!')
            ans = model2.predict(x)

            if model1.predict(x) == 1:
                if ans[0] == 0:
                    st.write('Heat Dissipation Failure')
                elif ans[0] == 1 :                          #(elif ans[0] == 1 : #st.write('No Failure'))
                    st.write('Unknown Failure type')
                elif ans[0] == 2:
                    st.write('Overstrain Failure')
                elif ans[0] == 3:
                    st.write('Power Failure')
                elif ans[0] == 4:
                    st.write('Random Failures')
                elif ans[0] == 5:
                    st.write('Tool Wear Failure')
                else:
                    st.write( 'Can not  detect the failure type')
            
            elif model1.predict(x) == 0:
                st.write('No Failure')
            else:
                st.write( 'Can not  detect the failure type')


result = page()
#page()

col1, col2 = st.columns([1,1])
#print(result)
#check(result)

clf1(result)
clf2(result)
#st.markdown('##')