#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 12:16:10 2020

@author: elliotgross
"""

import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from decimal import Decimal
from sklearn.preprocessing import MinMaxScaler

import Visualisations_Helper_Script as hs
from Data_Transformer_Pipeline import Data_Transformer


df = hs.clean_data(hs.load_data()[0], hs.load_data()[1])
model = hs.get_xgb_model(df)



def generate_test_df(seconds_threshold, min_proton, max_proton, min_neutron, max_neutron):
    z, n = [],[]
    for i in range(min_proton,max_proton+1):
        for j in range(min_neutron,max_neutron+1):
            z.append(i)
            n.append(j)
            
    test_df = pd.DataFrame({'Z':z, 'N':n})
    test_df['Mass'] = test_df['N'] + test_df['Z']
    test_df['Mass'] = [Decimal(test_df.loc[i,'Z']*0.00054386734)+test_df.loc[i,'Mass'] for i in range(test_df.shape[0])]
    #fix mass
    
    data_transformer = hs.quick_transformer_generator(df, 'Seconds',
                                                      s_threshold=seconds_threshold)
    
    prepared_X_test = data_transformer.transform(test_df)
    
    norm = MinMaxScaler().fit(prepared_X_test)
    X_norm_test = pd.DataFrame(norm.transform(prepared_X_test), columns=prepared_X_test.columns)
    
    return X_norm_test, prepared_X_test
    

def plot_model(df, X_test, predictions, seconds_threshold,
               min_proton, max_proton, min_neutron, max_neutron,
               alpha_predictions, show_known_values, alpha_known_values):
    
    fig = plt.figure()
    
    
    if show_known_values:
        data_transformer = hs.quick_transformer_generator(df, 'Seconds',
                                                          s_threshold=seconds_threshold)
        X = df.drop(['Half Life'], axis=1)
        y = df['Half Life']
        X,y = data_transformer.transform(X,y)
        X = X[(X['Z']>=min_proton) & (X['Z']<=max_proton)]
        X = X[(X['N']>=min_neutron) & (X['N']<=max_neutron)]
        y = y[X.index]
    
        
        true_stable = X[y==1]
        plt.scatter(true_stable['N'],true_stable['Z'],
                    color='blue', alpha=alpha_known_values, s=12, label='True')


    
    predicted_stable = X_test[predictions==1]
    plt.scatter(predicted_stable['N'], predicted_stable['Z'],
                color='green', alpha=alpha_predictions, s=12, label='Predicted')
    
    
    plt.xlim(min_neutron-2, max_neutron+5)
    plt.ylim(min_proton-2, max_proton+5)

    plt.xlabel('N')
    plt.ylabel('Z')
    
    plt.legend()
    plt.show()
    st.pyplot()



st.sidebar.markdown('**Data Options:**')
min_proton = int(st.sidebar.text_input("Min Protons: ", '1'))
max_proton = int(st.sidebar.text_input("Max Protons: ", '118'))
min_neutron = int(st.sidebar.text_input("Min Neutrons: ", '1'))
max_neutron = int(st.sidebar.text_input("Max Neutrons: ", '117'))

seconds_threshold = float(st.text_input("Half-Life Threshold (Seconds): ", '3600'))

st.sidebar.markdown('**Graph Options:**')
alpha_predictions = st.sidebar.slider('Alpha - Predictions:', 0.0,1.0,.37)

show_known_values = st.sidebar.checkbox("Show Known Data",True)
alpha_known_values = st.sidebar.slider('Alpha - Known Data:', 0.0,1.0,.5)

norm_X, X_test = generate_test_df(seconds_threshold, min_proton, max_proton, 
                                  min_neutron, max_neutron)
new_model = hs.quick_model_generator(df, model, 'Seconds', 
                                             s_threshold=seconds_threshold)

predictions_test = new_model.predict(norm_X)


plot_model(df, X_test, predictions_test, seconds_threshold, min_proton,
           max_proton, min_neutron, max_neutron, alpha_predictions,
           show_known_values, alpha_known_values)

#hs.update_df(df)

